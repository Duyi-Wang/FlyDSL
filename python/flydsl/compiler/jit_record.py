# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Automatic records of successful JIT misses and startup compilation replay."""

import gzip
import hashlib
import json
import os
import threading
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from weakref import WeakValueDictionary

from ..utils import env
from ..utils.file import atomic_write
from .jit_plan import _unique_object

# The returned JitReplay owns these artifacts; normal JIT functions adopt them
# on a matching trace. Dropping an unused replay releases its engines/binaries.
_replayed = WeakValueDictionary()
_replay_lock = threading.RLock()
_auto_replay_lock = threading.RLock()
_auto_replays = {}
_AUTO_REPLAY_RUNNING = object()


def _reset_replay_after_fork():
    # A child must not inherit completion flags or a lock held by a vanished
    # thread. Automatic replay only compiles; it never initializes GPU engines.
    global _replayed, _replay_lock, _auto_replay_lock, _auto_replays
    _replayed = WeakValueDictionary()
    _replay_lock = threading.RLock()
    _auto_replay_lock = threading.RLock()
    _auto_replays = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_replay_after_fork)


class JitReplay:
    """Keep startup-compiled artifacts alive for subsequent normal JIT calls.

    This is not a callable launcher. Retain it in the serving worker, especially
    with disk caching disabled. Normal JIT calls still trace to verify a match.
    """

    def __init__(self):
        self._artifacts = {}

    def __len__(self):
        return len(self._artifacts)


def _get_replayed_artifact(key):
    with _replay_lock:
        return _replayed.get((os.getpid(), key))


def _maybe_replay_jit_records(backend):
    """One automatic batch per process/directory/compilation configuration.

    Called outside tracing, before the current JIT request looks up its artifact.
    Missing/empty directories are remembered too: a recording run must not
    repeatedly replay the records it is itself producing. Failures remain fatal
    for this scope; subsequent calls do not retry a partially completed batch.
    """
    from .jit_function import _cache_invalidating_env_values

    directory = _records_dir().absolute()
    scope = (
        os.getpid(),
        str(directory),
        backend.target,
        backend.hash(),
        _cache_invalidating_env_values(backend.target),
        env.runtime.enable_cache,
        env.runtime.run_only,
        env.debug.dump_ir,
        env.debug.dump_asm,
    )
    with _auto_replay_lock:
        state = _auto_replays.get(scope)
        if state is _AUTO_REPLAY_RUNNING:
            raise RuntimeError("Recursive automatic JIT replay")
        if isinstance(state, str):
            raise RuntimeError(state)
        if state is not None:
            return
        _auto_replays[scope] = _AUTO_REPLAY_RUNNING
        try:
            # Retain the handle inside FlyDSL, including with ENABLE_CACHE=0.
            # Defer engine initialization to the worker's actual kernel calls.
            _auto_replays[scope] = replay_jit_records(directory) if directory.exists() else JitReplay()
        except Exception as exc:
            message = f"Automatic JIT replay failed for {directory}: {exc}"
            _auto_replays[scope] = message
            raise RuntimeError(message) from exc


def _records_dir(directory=None):
    if directory is not None:
        return Path(directory)
    if not env.runtime.cache_dir:
        raise ValueError("JIT recording/replay requires a nonempty FLYDSL_RUNTIME_CACHE_DIR")
    return Path(env.runtime.cache_dir) / "jit-records"


def _compiler_fingerprint():
    from .jit_function import _flydsl_key

    # _flydsl_key is a long concatenation of source/native-library digests.
    # Keep its full invalidation coverage without repeating that list per record.
    return hashlib.sha256(_flydsl_key().encode()).hexdigest()


def _record_compilation(key, asm, config, backend, artifact):
    """Called only after successful lowering, never on memory/disk/lock hits."""
    from .jit_function import FileLock, _cache_invalidating_env_values, _disk_cache_key

    try:
        reason = None
        if artifact._post_load_processors:
            reason = "post-load initializers require an application JitPlan factory"
        elif artifact._link_libs:
            reason = "external link libraries require an application JitPlan factory"
        elif artifact._ktrace_traced:
            reason = "ktrace event registration requires an application JitPlan factory"
        directory = _records_dir()
        directory.mkdir(parents=True, exist_ok=True)
        disk_key = _disk_cache_key(key)
        path = directory / f"{disk_key}.json.gz"
        with FileLock(directory / f"{disk_key}.lock", exclusive=True, timeout=30):
            if path.exists():
                return
            record = {
                "schema_version": 1,
                "artifact_key": key,
                "compiler": _compiler_fingerprint(),
                "target": asdict(backend.target),
                "environment": dict(_cache_invalidating_env_values(backend.target)),
                "backend_hash": backend.hash(),
                "pipeline": asdict(config),
                "entry": artifact._entry,
                "ir": asm if reason is None else None,
                "replay_error": reason,
            }
            data = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            with atomic_write(path, mode="wb") as output:
                output.write(gzip.compress(data, compresslevel=6, mtime=0))
        if reason:
            warnings.warn(
                f"Recorded JIT {artifact._entry}, but automatic replay is unavailable: {reason}", RuntimeWarning
            )
    except Exception as exc:
        # Recording is diagnostic; a full/unwritable filesystem must not undo
        # successful compilation or stop a serving workload.
        warnings.warn(f"Failed to record JIT {artifact._entry}: {exc}", RuntimeWarning)


def _read_record(path, backend, compiler):
    from .jit_function import PipelineConfig, _artifact_cache_key, _cache_invalidating_env_values

    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            record = json.load(source, object_pairs_hook=_unique_object)
        fields = {
            "schema_version",
            "artifact_key",
            "compiler",
            "target",
            "environment",
            "backend_hash",
            "pipeline",
            "entry",
            "ir",
            "replay_error",
        }
        if type(record) is not dict or set(record) != fields:
            raise ValueError("invalid record fields")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("unsupported record schema")
        if record["compiler"] != compiler:
            raise ValueError("compiler/toolchain mismatch; record again with this FlyDSL installation")
        if record["target"] != asdict(backend.target) or record["backend_hash"] != backend.hash():
            raise ValueError("GPU target/backend mismatch")
        if record["environment"] != dict(_cache_invalidating_env_values(backend.target)):
            raise ValueError("compilation environment mismatch")
        if record["replay_error"] is not None:
            raise ValueError(record["replay_error"])
        if not isinstance(record["entry"], str) or not record["entry"].isidentifier():
            raise ValueError("invalid launcher name")
        if not isinstance(record["ir"], str):
            raise ValueError("missing compilation input IR")
        config = PipelineConfig(**record["pipeline"])
        key = _artifact_cache_key(record["ir"], config, backend)
        if key != record["artifact_key"]:
            raise ValueError("compilation input checksum mismatch")
        return record, config
    except (ValueError, TypeError, KeyError, OSError, EOFError) as exc:
        raise ValueError(f"Cannot replay JIT record {path}: {exc}") from exc


def replay_jit_records(directory=None, *, preload=False) -> JitReplay:
    """Recompile automatically recorded IR before serving; never launch kernels.

    ``directory`` defaults to ``FLYDSL_RUNTIME_CACHE_DIR/jit-records``. Copy
    that subdirectory alone to a compatible installation; no compiled cache or
    application factory is needed. All records are checked before compilation.
    Existing local binary caches are honored, and ENABLE_CACHE=0 keeps results
    in memory only. RUN_ONLY forbids missing-artifact compilation as usual.

    ``preload=True`` also initializes engines after the worker selects its GPU,
    unless COMPILE_ONLY=1. Retain the returned handle for in-memory reuse.
    Records are trusted compilation inputs, like kernel source code.
    """
    from .._mlir import ir
    from .backends import get_backend
    from .jit_executor import CompiledArtifact
    from .jit_function import JitCacheManager, MlirCompiler, _create_mlir_context, _disk_cache_key

    if ir.Context.current is not None:
        raise RuntimeError("JIT record replay cannot run during DSL tracing")
    directory = _records_dir(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if env.debug.dump_ir or env.debug.dump_asm:
        raise ValueError("JIT record replay is incompatible with IR/ASM dumping")
    backend = get_backend()
    compiler = _compiler_fingerprint()
    # Validate before any lowering, engine loading, or cache writes.
    records = [_read_record(path, backend, compiler) for path in sorted(directory.glob("*.json.gz"))]
    result = JitReplay()
    for record, config in records:
        key = record["artifact_key"]
        manager = None
        if env.runtime.enable_cache or env.runtime.run_only:
            if not env.runtime.cache_dir:
                raise ValueError("JIT replay disk caching requires a nonempty FLYDSL_RUNTIME_CACHE_DIR")
            manager = JitCacheManager(Path(env.runtime.cache_dir) / record["entry"])
        disk_key = _disk_cache_key(key) if manager else None
        artifact = _get_replayed_artifact(key)
        if artifact is None and manager:
            artifact = manager.get(disk_key)
        if artifact is None:
            if env.runtime.run_only:
                raise RuntimeError(f"FLYDSL_RUNTIME_RUN_ONLY=1 but no usable cache for {record['entry']}")
            lock = manager.compile_lock(disk_key) if manager else nullcontext((None, None))
            with lock as (existing, writer):
                artifact = existing
                if artifact is None:
                    with _create_mlir_context():
                        module = MlirCompiler.compile(record["ir"], func_name=record["entry"], config=config)
                        artifact = CompiledArtifact(module, record["entry"], record["ir"])
                    if writer:
                        writer(artifact)
        if preload and not env.compile.compile_only:
            from ..runtime.device_runtime import ensure_compile_runtime_pairing_from_env

            ensure_compile_runtime_pairing_from_env(backend.target.backend)
            artifact._get_func_exe()
        result._artifacts[key] = artifact
        with _replay_lock:
            _replayed[(os.getpid(), key)] = artifact
    return result

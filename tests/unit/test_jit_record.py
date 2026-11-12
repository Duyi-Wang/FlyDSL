# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Automatic recording must observe actual compilation, not cache lookup."""

import gc
import gzip
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function as jf
from flydsl.compiler import jit_record as recording

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def make_launcher(delta=1.0):
    # Real nested closures, as used by application kernel factories. Automatic
    # replay must work without importing or registering this Python factory.
    @flyc.kernel
    def add(x: fx.Pointer):
        x[fx.thread_idx.x] = x[fx.thread_idx.x] + delta

    @flyc.jit
    def launch(x: fx.Pointer):
        add(x).launch(grid=(1, 1, 1), block=(64, 1, 1))

    return launch


def compile_launcher(launcher):
    return flyc.compile(launcher, flyc.from_c_void_p(fx.Float32, 0), compile_only=True)


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    for key, value in {
        "ARCH": "gfx950",
        "COMPILE_ONLY": "1",
        "FLYDSL_COMPILE_BACKEND": "rocm",
        "FLYDSL_COMPILE_LLVM_DIR": "",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
        "FLYDSL_RUNTIME_RUN_ONLY": "0",
        "FLYDSL_RUNTIME_RECORD_JIT": "1",
        "FLYDSL_RUNTIME_REPLAY_JIT": "0",
        "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "cache"),
        "FLYDSL_DEBUG_ENABLE_DEBUG_INFO": "0",
        "FLYDSL_KTRACE_ENABLE": "0",
        "FLYDSL_DUMP_IR": "0",
        "FLYDSL_DEBUG_DUMP_ASM": "0",
    }.items():
        monkeypatch.setenv(key, value)
    recording._replayed.clear()
    recording._auto_replays.clear()
    yield
    recording._auto_replays.clear()
    recording._replayed.clear()


def records(tmp_path):
    return sorted((tmp_path / "cache" / "jit-records").glob("*.json.gz"))


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def test_only_successful_misses_are_recorded(tmp_path, monkeypatch):
    launcher = make_launcher()
    compile_launcher(launcher)
    assert len(records(tmp_path)) == 1
    record = read(records(tmp_path)[0])
    assert "gpu.binary" not in record["ir"] and "gpu.func" in record["ir"]
    assert record["replay_error"] is None
    assert len(record["compiler"]) == 64  # digest, not the full per-source toolchain inventory
    shutil.rmtree(tmp_path / "cache" / "jit-records")

    def unexpected(*args, **kwargs):
        raise AssertionError("Cache hit must not lower or record")

    monkeypatch.setattr(jf.MlirCompiler, "compile", unexpected)
    compile_launcher(launcher)  # memory hit
    compile_launcher(make_launcher())  # new JitFunction, disk hit
    assert records(tmp_path) == []


def test_default_disabled_does_not_record(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYDSL_RUNTIME_RECORD_JIT")
    monkeypatch.delenv("FLYDSL_RUNTIME_REPLAY_JIT")
    compile_launcher(make_launcher())
    assert not (tmp_path / "cache" / "jit-records").exists()
    assert not recording._auto_replays


def test_failed_compilation_is_not_recorded(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("compiler failure")

    monkeypatch.setattr(jf.MlirCompiler, "compile", fail)
    with pytest.raises(RuntimeError, match="compiler failure"):
        compile_launcher(make_launcher())
    assert records(tmp_path) == []


def test_hit_after_compile_lock_is_not_recorded(tmp_path, monkeypatch):
    compiled = compile_launcher(make_launcher())
    shutil.rmtree(tmp_path / "cache" / "jit-records")
    monkeypatch.setattr(jf.JitCacheManager, "get", lambda *args: None)

    @contextmanager
    def late_hit(*args, **kwargs):
        yield compiled._keepalive, None

    monkeypatch.setattr(jf.JitCacheManager, "compile_lock", late_hit)
    compile_launcher(make_launcher())
    assert records(tmp_path) == []


def test_unique_closure_variants_without_binary_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    compile_launcher(make_launcher(1.0))
    compile_launcher(make_launcher(2.0))
    compile_launcher(make_launcher(1.0))  # real compile, record deduplicates
    assert len(records(tmp_path)) == 2
    assert not list((tmp_path / "cache").rglob("*.pkl"))


def test_record_write_failure_preserves_compilation(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    path = tmp_path / "not-a-directory"
    path.write_text("occupied")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(path))
    with pytest.warns(RuntimeWarning, match="Failed to record JIT"):
        assert isinstance(compile_launcher(make_launcher()), flyc.CompiledFunction)


@pytest.mark.parametrize("disk_cache", [False, True])
def test_fresh_process_replay_needs_only_records(tmp_path, disk_cache):
    compile_launcher(make_launcher())
    source = tmp_path / "portable-records"
    shutil.copytree(tmp_path / "cache" / "jit-records", source)
    shutil.rmtree(tmp_path / "cache")
    code = """
import json, os, runpy, sys
sys.path.insert(0, sys.argv[1])
import flydsl.compiler as flyc
from flydsl.compiler import jit_function as jf
handle = flyc.replay_jit_records(sys.argv[3])
assert len(handle) == 1
assert all(a._engine is None for a in handle._artifacts.values())
namespace = runpy.run_path(sys.argv[2])  # application imports AFTER replay
def unexpected(*args, **kwargs):
    raise AssertionError('Serving must reuse the startup compilation')
jf.MlirCompiler.compile = unexpected
compiled = namespace['compile_launcher'](namespace['make_launcher']())
assert compiled._keepalive is next(iter(handle._artifacts.values()))
print(json.dumps({'reused': True}))
"""
    child_env = dict(os.environ)
    child_env["FLYDSL_RUNTIME_ENABLE_CACHE"] = str(int(disk_cache))
    child = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(flydsl.__file__).parents[1]), __file__, str(source)],
        env=child_env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == {"reused": True}
    assert bool(list((tmp_path / "cache").rglob("*.pkl"))) == disk_cache
    assert records(tmp_path) == []  # replay/hits do not create new recordings


def test_concurrent_processes_merge_records_without_lost_variants(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    code = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
namespace = runpy.run_path(sys.argv[2])
for delta in [1.0, float(sys.argv[3])]:
    namespace['compile_launcher'](namespace['make_launcher'](delta))
"""

    def worker(delta):
        return subprocess.run(
            [sys.executable, "-B", "-c", code, str(Path(flydsl.__file__).parents[1]), __file__, str(delta)],
            text=True,
            capture_output=True,
            timeout=120,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        children = list(pool.map(worker, [2.0, 3.0]))
    for child in children:
        assert child.returncode == 0, child.stdout + child.stderr
    paths = records(tmp_path)
    assert len(paths) == 3
    assert len({read(path)["artifact_key"] for path in paths}) == 3
    assert not list((tmp_path / "cache").rglob("*.pkl"))
    assert not list((tmp_path / "cache").rglob("*.tmp"))


@pytest.mark.parametrize("field", ["compiler", "target", "environment", "ir", "schema_version", "replay_error"])
def test_incompatible_records_fail_before_compilation(tmp_path, monkeypatch, field):
    compile_launcher(make_launcher())
    path = records(tmp_path)[0]
    record = read(path)
    record[field] = "changed"
    path.write_bytes(gzip.compress(json.dumps(record).encode()))

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid records must fail validation first")

    monkeypatch.setattr(jf.MlirCompiler, "compile", unexpected)
    with pytest.raises(ValueError, match="Cannot replay JIT record"):
        flyc.replay_jit_records()


def test_replay_honors_run_only_and_compile_only(tmp_path, monkeypatch):
    compile_launcher(make_launcher())
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    handle = flyc.replay_jit_records(preload=True)
    assert all(a._engine is None for a in handle._artifacts.values())
    del handle
    gc.collect()
    for path in (tmp_path / "cache").rglob("*.pkl"):
        path.unlink()
    with pytest.raises(RuntimeError, match="RUN_ONLY=1"):
        flyc.replay_jit_records()


def test_replay_handle_lifetime_and_engine_option(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    compile_launcher(make_launcher())
    monkeypatch.setenv("COMPILE_ONLY", "0")
    loaded = []
    monkeypatch.setattr(jf.CompiledArtifact, "_get_func_exe", lambda self: loaded.append(self._entry))
    handle = flyc.replay_jit_records(preload=True)
    assert loaded == ["launch"]
    assert len(handle) == len(recording._replayed) == 1
    del handle
    gc.collect()
    assert len(recording._replayed) == 0


def test_unsupported_initializers_are_recorded_explicitly(tmp_path):
    compiled = compile_launcher(make_launcher())
    # Use the real recorded compile input, then exercise unsupported metadata.
    original = read(records(tmp_path)[0])
    shutil.rmtree(tmp_path / "cache" / "jit-records")
    compiled._keepalive._post_load_processors = [lambda: None]
    with pytest.warns(RuntimeWarning, match="post-load initializers"):
        recording._record_compilation(
            original["artifact_key"],
            original["ir"],
            jf.PipelineConfig(**original["pipeline"]),
            jf.get_backend(),
            compiled._keepalive,
        )
    assert "post-load initializers" in read(records(tmp_path)[0])["replay_error"]
    with pytest.raises(ValueError, match="post-load initializers"):
        flyc.replay_jit_records()


def test_env_only_replay_in_fresh_process(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    for delta in [1.0, 2.0]:
        compile_launcher(make_launcher(delta))
    code = """
import gc, json, os, runpy, sys
sys.path.insert(0, sys.argv[1])
namespace = runpy.run_path(sys.argv[2])
jf = namespace['jf']
recording = namespace['recording']
assert not recording._auto_replays  # import did not replay
compile_impl = jf.MlirCompiler.compile
lowered = []
def count(*args, **kwargs):
    lowered.append(kwargs['func_name'])
    return compile_impl(*args, **kwargs)
jf.MlirCompiler.compile = count
before = dict(os.environ)
first = namespace['compile_launcher'](namespace['make_launcher'](1.0))
assert len(lowered) == 2  # entire recorded batch, not only requested variant
assert len(recording._auto_replays) == 1
assert len(next(iter(recording._auto_replays.values()))) == 2
gc.collect()
second = namespace['compile_launcher'](namespace['make_launcher'](2.0))
assert len(lowered) == 2
assert first._keepalive._engine is second._keepalive._engine is None
assert dict(os.environ) == before
namespace['compile_launcher'](namespace['make_launcher'](3.0))  # unseen variant
assert len(lowered) == 3
print(json.dumps({'compiled': len(lowered), 'batches': len(recording._auto_replays)}))
"""
    child_env = dict(os.environ, FLYDSL_RUNTIME_REPLAY_JIT="1")
    child = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(flydsl.__file__).parents[1]), __file__],
        env=child_env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == {"compiled": 3, "batches": 1}
    assert len(records(tmp_path)) == 3
    assert not list((tmp_path / "cache").rglob("*.pkl"))


@pytest.mark.parametrize("existing_empty_directory", [False, True])
def test_auto_replay_does_not_rescan_its_own_new_records(tmp_path, monkeypatch, existing_empty_directory):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_REPLAY_JIT", "1")
    if existing_empty_directory:
        (tmp_path / "cache" / "jit-records").mkdir(parents=True)
    replay = recording.replay_jit_records
    batches = []

    def count(*args, **kwargs):
        batches.append(1)
        return replay(*args, **kwargs)

    monkeypatch.setattr(recording, "replay_jit_records", count)
    compile_launcher(make_launcher(1.0))
    compile_launcher(make_launcher(2.0))
    assert len(records(tmp_path)) == 2
    assert len(batches) == int(existing_empty_directory)
    assert len(recording._auto_replays) == 1
    assert len(next(iter(recording._auto_replays.values()))) == 0


def test_auto_replay_serializes_concurrent_callers(tmp_path, monkeypatch):
    import threading

    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    compile_launcher(make_launcher())
    replay = recording.replay_jit_records
    batches = []
    start = threading.Barrier(4)
    backend = jf.get_backend()

    def count(*args, **kwargs):
        batches.append(1)
        return replay(*args, **kwargs)

    def worker(_):
        start.wait(timeout=10)
        recording._maybe_replay_jit_records(backend)

    monkeypatch.setattr(recording, "replay_jit_records", count)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    assert len(batches) == len(recording._auto_replays) == 1


def test_auto_replay_failure_is_not_silently_retried(tmp_path, monkeypatch):
    compile_launcher(make_launcher())
    path = records(tmp_path)[0]
    record = read(path)
    record["compiler"] = "incompatible"
    path.write_bytes(gzip.compress(json.dumps(record).encode()))
    monkeypatch.setenv("FLYDSL_RUNTIME_REPLAY_JIT", "1")
    replay = recording.replay_jit_records
    batches = []

    def count(*args, **kwargs):
        batches.append(1)
        return replay(*args, **kwargs)

    monkeypatch.setattr(recording, "replay_jit_records", count)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Automatic JIT replay failed.*compiler/toolchain mismatch"):
            compile_launcher(make_launcher())
    assert len(batches) == 1


def test_auto_replay_checks_changed_directory_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    compile_launcher(make_launcher())
    second_root = tmp_path / "other-cache"
    shutil.copytree(tmp_path / "cache", second_root)
    monkeypatch.setenv("FLYDSL_RUNTIME_REPLAY_JIT", "1")
    compile_launcher(make_launcher())
    assert len(recording._auto_replays) == 1
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(second_root))
    compile_launcher(make_launcher())
    assert len(recording._auto_replays) == 2
    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "3")
    with pytest.raises(RuntimeError, match="compilation environment mismatch"):
        compile_launcher(make_launcher())


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_forked_child_resets_replay_state_and_locks(tmp_path, monkeypatch):
    import select
    import signal
    import threading

    (tmp_path / "cache" / "jit-records").mkdir(parents=True)
    backend = jf.get_backend()
    recording._maybe_replay_jit_records(backend)
    assert recording._auto_replays
    acquired, release = threading.Event(), threading.Event()

    def hold_locks():
        with recording._auto_replay_lock, recording._replay_lock:
            acquired.set()
            release.wait(timeout=30)

    holder = threading.Thread(target=hold_locks)
    holder.start()
    assert acquired.wait(timeout=10)
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                assert not recording._auto_replays and not recording._replayed
                with recording._replay_lock:
                    pass
                recording._maybe_replay_jit_records(backend)
                assert len(recording._auto_replays) == 1
                os.write(write_fd, b"ok")
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(write_fd)
        ready, _, _ = select.select([read_fd], [], [], 15)
        if not ready:
            os.kill(pid, signal.SIGKILL)
        _, status = os.waitpid(pid, 0)
        assert ready and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        assert os.read(read_fd, 2) == b"ok"
    finally:
        os.close(read_fd)
        release.set()
        holder.join(timeout=10)

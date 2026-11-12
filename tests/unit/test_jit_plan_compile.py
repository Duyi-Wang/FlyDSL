# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile a saved recipe list on a CPU host, without disk artifacts or dispatch."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@flyc.kernel
def _add_kernel(x: fx.Pointer, n: fx.Int32, delta: fx.Constexpr[float]):
    idx = fx.thread_idx.x
    if idx < n:
        x[idx] = x[idx] + delta


@flyc.jit
def _add(x: fx.Pointer, n: fx.Int32, delta: fx.Constexpr[float], stream: fx.Stream):
    _add_kernel(x, n, delta).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def _rebuild_add(n, delta):
    return flyc.JitCall(_add, (flyc.from_c_void_p(fx.Float32, 0), n, delta, fx.Stream(None)))


def test_saved_plan_recompiles_without_gpu_or_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_DUMP_IR", "0")
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "0")
    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "0")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    plan = flyc.JitPlan()
    plan.add("add.v1", n=64, delta=1.0)
    plan.add("add.v1", n=64, delta=2.0)
    plan.add("add.v1", n=64, delta=1.0)
    path = tmp_path / "jit-plan.json"
    plan.save(path)

    # Only the JSON crosses into the fresh process; no compiled object or cache
    # from this process is needed. Factories come from application Python code.
    code = """
import json, runpy, sys
sys.path.insert(0, sys.argv[1])
namespace = runpy.run_path(sys.argv[2])
flyc = namespace['flyc']
compiled = flyc.JitPlan.load(sys.argv[3]).compile({'add.v1': namespace['_rebuild_add']})
assert len(compiled) == 2
assert all(value._keepalive._engine is None for value in compiled)
print(json.dumps({'compiled': len(compiled)}))
"""
    child = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(flydsl.__file__).parents[1]), __file__, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == {"compiled": 2}
    assert not (tmp_path / "cache").exists()

    compiled = flyc.JitPlan.load(path).compile({"add.v1": _rebuild_add})
    assert len(compiled) == 2
    assert all(isinstance(value, flyc.CompiledFunction) for value in compiled)
    assert all(value._keepalive._engine is None and value._keepalive._func_exe is None for value in compiled)
    assert not (tmp_path / "cache").exists()
    assert os.environ["COMPILE_ONLY"] == "1"

    monkeypatch.setenv("COMPILE_ONLY", "0")
    again = plan.compile({"add.v1": _rebuild_add})
    assert all(a._keepalive is b._keepalive for a, b in zip(compiled, again))
    assert all(value._keepalive._engine is None for value in again)
    assert not (tmp_path / "cache").exists()

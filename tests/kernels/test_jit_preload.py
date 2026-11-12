# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device regressions for JitFunction.preload; use real compilation and dispatch."""

import os

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_record as recording

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("requires a ROCm GPU", allow_module_level=True)


@pytest.fixture(autouse=True)
def isolated_preload_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", torch.cuda.get_device_properties().gcnArchName.split(":", 1)[0])
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RECORD_JIT", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_REPLAY_JIT", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "0")
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "0")
    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "0")
    monkeypatch.setenv("COMPILE_ONLY", "0")
    recording._auto_replays.clear()
    recording._replayed.clear()
    yield
    recording._auto_replays.clear()
    recording._replayed.clear()


def make_increment():
    """Fresh launcher/kernel objects keep in-memory specialization caches isolated."""

    @flyc.kernel
    def kernel(x: fx.Tensor, delta: fx.Constexpr[float]):
        idx = fx.thread_idx.x
        if idx < 64:
            x[idx] = x[idx] + delta

    @flyc.jit
    def increment(x: fx.Tensor, delta: fx.Constexpr[float], stream: fx.Stream):
        kernel(x, delta).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return increment


def only_artifact(launcher):
    assert len(launcher._mem_cache) == 1
    return next(iter(launcher._mem_cache.values()))


def test_preload_live_tensor_does_not_launch_and_reuses_engine():
    increment = make_increment()
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    compiled = increment.preload(x, 7.0, stream)
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0
    assert os.environ["COMPILE_ONLY"] == "0"
    assert compiled is not None
    artifact = only_artifact(increment)
    engine = artifact._engine
    assert engine is not None
    assert artifact._func_exe is not None
    assert compiled._keepalive is artifact

    again = increment.preload(x, 7.0, stream)
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0
    assert again._keepalive is artifact
    assert artifact._engine is engine

    increment(x, 7.0, stream)
    increment(x, 7.0, stream)
    compiled(x, 7.0, stream)
    state = compiled._call_state
    compiled(x, 7.0, stream)
    torch.cuda.synchronize()
    assert torch.all(x == 28.0).item()
    assert only_artifact(increment) is artifact
    assert artifact._engine is engine
    assert compiled._call_state is state


def test_compile_only_preload_does_not_create_execution_engine(monkeypatch):
    increment = make_increment()
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    monkeypatch.setenv("COMPILE_ONLY", "1")

    assert increment.preload(x, 3.0, stream) is None
    assert increment.preload(x, 3.0, stream) is None
    artifact = only_artifact(increment)
    assert artifact._engine is None
    assert artifact._func_exe is None
    assert os.environ["COMPILE_ONLY"] == "1"
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0

    monkeypatch.setenv("COMPILE_ONLY", "0")
    compiled = increment.preload(x, 3.0, stream)
    assert compiled is not None
    assert compiled._keepalive is artifact
    assert artifact._engine is not None
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0
    increment(x, 3.0, stream)
    torch.cuda.synchronize()
    assert torch.all(x == 3.0).item()


def test_bound_method_preload_preserves_normal_calls():
    class Program:
        @flyc.kernel
        def kernel(self, x: fx.Tensor, delta: fx.Constexpr[float]):
            idx = fx.thread_idx.x
            if idx < 64:
                x[idx] = x[idx] + delta

        @flyc.jit
        def run(self, x: fx.Tensor, delta: fx.Constexpr[float], stream: fx.Stream):
            self.kernel(x, delta).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    program = Program()
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    compiled = program.run.preload(x, 5.0, stream)
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0
    artifact = only_artifact(Program.run)
    engine = artifact._engine
    assert engine is not None
    assert compiled._keepalive is artifact

    program.run(x, 5.0, stream)
    compiled(x, 5.0, stream)
    # Existing flyc.compile must still recognize the descriptor's bound partial.
    compiled_again = flyc.compile(program.run, x, 5.0, stream)
    torch.cuda.synchronize()
    assert torch.all(x == 15.0).item()
    assert compiled_again._keepalive is artifact
    assert only_artifact(Program.run) is artifact
    assert artifact._engine is engine


def test_saved_plan_preloads_without_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    increment = make_increment()
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    plan = flyc.JitPlan()
    plan.add("increment.v1", delta=3.0)
    path = tmp_path / "jit-plan.json"
    plan.save(path)

    def factory(delta):
        return flyc.JitCall(increment, (x, delta, stream))

    ready = flyc.JitPlan.load(path).preload({"increment.v1": factory})
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0
    artifact = only_artifact(increment)
    assert ready[0]._keepalive is artifact and artifact._engine is not None
    assert not (tmp_path / "cache").exists()
    ready[0](x, 3.0, stream)
    torch.cuda.synchronize()
    assert torch.all(x == 3.0).item()


def test_automatic_replay_reuses_engine_without_launching(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RECORD_JIT", "1")
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    increment = make_increment()
    flyc.compile(increment, x, 4.0, stream, compile_only=True)
    assert len(list((tmp_path / "cache" / "jit-records").glob("*.json.gz"))) == 1
    del increment

    startup = flyc.replay_jit_records(preload=True)
    assert len(startup) == 1
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0
    artifact = next(iter(startup._artifacts.values()))
    engine = artifact._engine
    assert engine is not None
    increment = make_increment()
    increment(x, 4.0, stream)
    torch.cuda.synchronize()
    assert torch.all(x == 4.0).item()
    assert only_artifact(increment) is artifact and artifact._engine is engine
    assert not list((tmp_path / "cache").rglob("*.pkl"))


def test_env_only_replay_defers_engines_until_kernel_use(tmp_path, monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler

    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RECORD_JIT", "1")
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    flyc.compile(make_increment(), x, 4.0, stream, compile_only=True)
    monkeypatch.setenv("FLYDSL_RUNTIME_REPLAY_JIT", "1")
    increment = make_increment()
    compiled = flyc.compile(increment, x, 4.0, stream, compile_only=True)
    assert compiled._keepalive._engine is None
    assert len(recording._auto_replays) == 1
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0

    def unexpected(*args, **kwargs):
        raise AssertionError("An automatically replayed kernel must not compile again")

    monkeypatch.setattr(MlirCompiler, "compile", unexpected)
    increment(x, 4.0, stream)
    torch.cuda.synchronize()
    assert torch.all(x == 4.0).item()
    assert compiled._keepalive._engine is not None
    assert not list((tmp_path / "cache").rglob("*.pkl"))

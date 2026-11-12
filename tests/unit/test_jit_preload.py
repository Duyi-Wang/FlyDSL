# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Preload dispatch policy, including compile_only's callable return semantics."""

import os
from types import SimpleNamespace

import pytest

from flydsl._mlir import ir
from flydsl.compiler.jit_function import JitFunction
from flydsl.runtime import device_runtime

pytestmark = [pytest.mark.l0_backend_agnostic]


def launcher_body(*args, **kwargs):
    raise AssertionError("preload must not invoke the launcher")


@pytest.mark.parametrize("compile_only", ["0", "1"])
def test_preload_materializes_only_when_runtime_is_enabled(monkeypatch, compile_only):
    monkeypatch.setenv("COMPILE_ONLY", compile_only)
    launcher = JitFunction(launcher_body)
    events = []
    compiled = SimpleNamespace(
        _backend="rocm",
        _keepalive=SimpleNamespace(_get_func_exe=lambda: events.append("materialize")),
    )

    def compile_specialization(args, kwargs, *, execute, return_compiled):
        assert not execute, "preload must never dispatch kernels"
        assert return_compiled
        assert args == (17,) and kwargs == {"size": 64}
        assert os.environ["COMPILE_ONLY"] == compile_only
        events.append("compile")
        # Upstream returns a callable for this path even with COMPILE_ONLY=1.
        return compiled

    def check_pairing(backend):
        assert backend == "rocm"
        events.append("pairing")

    monkeypatch.setattr(launcher, "_compile_or_execute", compile_specialization)
    monkeypatch.setattr(device_runtime, "ensure_compile_runtime_pairing_from_env", check_pairing)

    result = launcher.preload(17, size=64)
    assert os.environ["COMPILE_ONLY"] == compile_only
    if compile_only == "1":
        assert result is None
        assert events == ["compile"]
    else:
        assert result is compiled
        assert events == ["compile", "pairing", "materialize"]


def test_preload_rejects_nested_dsl_tracing(monkeypatch):
    launcher = JitFunction(launcher_body)

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("nested preload must fail before compilation")

    monkeypatch.setattr(launcher, "_compile_or_execute", unexpected_compile)
    with ir.Context(), pytest.raises(RuntimeError, match="cannot run during DSL tracing"):
        launcher.preload()


def test_bound_preload_forwards_the_instance(monkeypatch):
    monkeypatch.setenv("COMPILE_ONLY", "0")

    class Program:
        @JitFunction
        def run(self, value, *, size):
            raise AssertionError("preload must not invoke the bound launcher")

    program = Program()
    calls = []
    compiled = SimpleNamespace(_backend="rocm", _keepalive=SimpleNamespace(_get_func_exe=lambda: None))

    def compile_specialization(args, kwargs, *, execute, return_compiled):
        assert not execute and return_compiled
        calls.append((args, kwargs))
        return compiled

    monkeypatch.setattr(Program.run, "_compile_or_execute", compile_specialization)
    monkeypatch.setattr(device_runtime, "ensure_compile_runtime_pairing_from_env", lambda backend: None)
    assert program.run.preload(17, size=64) is compiled
    assert calls == [((program, 17), {"size": 64})]

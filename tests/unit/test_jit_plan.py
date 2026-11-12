# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Portable recipe round trips and non-executing startup dispatch."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from flydsl._mlir import ir
from flydsl.compiler import JitCall, JitPlan
from flydsl.compiler.jit_function import JitFunction

pytestmark = [pytest.mark.l0_backend_agnostic]


def launcher_body(*args, **kwargs):
    raise AssertionError("A plan must never invoke the compiled launcher")


def test_round_trip_keeps_only_parameters_and_deduplicates(tmp_path):
    plan = JitPlan()
    parameters = {"shape": [64, 128], "dtype": "f16", "options": {"transposed": False}}
    plan.add("gemm.v1", **parameters)
    plan.add("gemm.v1", options={"transposed": False}, dtype="f16", shape=[64, 128])
    parameters["shape"][0] = 999
    plan.add("gemm.v1", dtype="f16", shape=[128, 128])
    path = tmp_path / "plan.json"
    plan.save(path)

    restored = JitPlan.load(path)
    assert len(restored) == 2
    assert restored.to_dict()["entries"][0]["parameters"]["shape"] == [64, 128]
    assert restored.to_dict() == plan.to_dict()
    snapshot = restored.to_dict()
    snapshot["entries"][0]["parameters"]["shape"][0] = 777
    assert restored.to_dict() == plan.to_dict()
    assert len(path.read_bytes()) < 300


def test_concurrent_recording_keeps_unique_entries():
    plan = JitPlan()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: plan.add("launch", n=n % 4), range(100)))
    assert len(plan) == 4
    assert {e["parameters"]["n"] for e in plan.to_dict()["entries"]} == {0, 1, 2, 3}


@pytest.mark.parametrize("value", [object(), b"binary", (1, 2), float("nan"), float("inf"), {1: "value"}])
def test_opaque_or_nonportable_parameters_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        JitPlan().add("launch", value=value)


def test_circular_parameters_are_rejected():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="Circular"):
        JitPlan().add("launch", value=value)


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":2,"entries":[]}',
        '{"schema_version":true,"entries":[]}',
        '{"schema_version":1,"entries":[],"code":"ignored"}',
        '{"schema_version":1,"entries":[],"entries":[]}',
        '{"schema_version":1,"entries":[{"recipe":"launch","parameters":{"n":NaN}}]}',
        '{"schema_version":1,"entries":[{"recipe":"launch","parameters":[]}]}',
    ],
)
def test_invalid_manifest_is_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        JitPlan.load(path)


def test_missing_factory_is_rejected_before_any_work():
    plan = JitPlan()
    plan.add("known", n=64)
    plan.add("missing", n=128)
    calls = []
    with pytest.raises(ValueError, match="missing"):
        plan.compile({"known": lambda **kwargs: calls.append(kwargs)})
    assert not calls


@pytest.mark.parametrize("preload", [False, True])
def test_replay_preserves_bound_self_and_does_not_change_cache_settings(tmp_path, monkeypatch, preload):
    class Program:
        @JitFunction
        def run(self, value, *, compile_only):
            raise AssertionError("Must not execute")

    program = Program()
    plan = JitPlan()
    # A recipe parameter can itself be named recipe.
    plan.add("bound.v1", recipe="parameter", value=17)
    path = tmp_path / "plan.json"
    plan.save(path)
    calls = []
    result = object()
    monkeypatch.setenv("COMPILE_ONLY", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    before = dict(os.environ)

    def factory(recipe, value):
        assert recipe == "parameter"
        return JitCall(program.run, (value,), {"compile_only": "launcher-argument"})

    def compile_call(args, kwargs, *, execute, return_compiled):
        assert not execute and return_compiled
        calls.append((args, kwargs, "compile"))
        return result

    def preload_call(*args, **kwargs):
        calls.append((args, kwargs, "preload"))
        return result

    monkeypatch.setattr(Program.run, "_compile_or_execute", compile_call)
    monkeypatch.setattr(Program.run, "preload", preload_call)
    restored = JitPlan.load(path)
    output = restored.preload({"bound.v1": factory}) if preload else restored.compile({"bound.v1": factory})
    assert output == [result]
    assert calls == [((program, 17), {"compile_only": "launcher-argument"}, "preload" if preload else "compile")]
    assert os.environ == before


def test_factory_mutation_does_not_change_the_plan(monkeypatch):
    plan = JitPlan()
    plan.add("shape", shape=[64])
    launcher = JitFunction(launcher_body)
    monkeypatch.setattr(launcher, "_compile_or_execute", lambda *a, **k: None)

    def factory(shape):
        shape[0] = 128
        return JitCall(launcher)

    plan.compile({"shape": factory})
    assert plan.to_dict()["entries"][0]["parameters"] == {"shape": [64]}


def test_preload_plan_honors_compile_only_without_skipping_compilation(monkeypatch):
    monkeypatch.setenv("COMPILE_ONLY", "1")
    launcher = JitFunction(launcher_body)
    calls = []

    def compile_call(args, kwargs, *, execute, return_compiled):
        assert not execute and return_compiled
        calls.append(True)
        return object()  # No runtime fields: touching an engine would fail.

    monkeypatch.setattr(launcher, "_compile_or_execute", compile_call)
    plan = JitPlan()
    plan.add("launch")
    assert plan.preload({"launch": lambda: JitCall(launcher)}) == [None]
    assert calls == [True]
    assert os.environ["COMPILE_ONLY"] == "1"


@pytest.mark.parametrize("bad", [None, JitCall(lambda: None), JitCall(JitFunction(launcher_body), kwargs={1: 2})])
def test_invalid_factory_result_is_rejected(bad):
    plan = JitPlan()
    plan.add("bad")
    with pytest.raises(TypeError):
        plan.compile({"bad": lambda: bad})


def test_plan_rejects_nested_tracing_before_resolving_factories():
    plan = JitPlan()
    plan.add("launch")
    with ir.Context(), pytest.raises(RuntimeError, match="during DSL tracing"):
        plan.compile({"launch": lambda: pytest.fail("Factory must not run")})


def test_empty_plan_is_portable(tmp_path):
    path = tmp_path / "empty.json"
    JitPlan().save(path)
    assert json.loads(path.read_text()) == {"schema_version": 1, "entries": []}
    assert JitPlan.load(path).compile({}) == []

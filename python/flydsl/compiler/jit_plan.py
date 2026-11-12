# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Portable startup compilation recipes, without tensors or compiled artifacts."""

import json
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..utils.file import atomic_write

if TYPE_CHECKING:
    from .jit_function import CompiledFunction


def _check_json(value, path="parameters", parents=()):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return
    if kind is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if kind not in (dict, list):
        raise TypeError(f"{path} must contain JSON values, got {kind.__name__}")
    if id(value) in parents:
        raise ValueError(f"Circular reference at {path}")
    parents = (*parents, id(value))
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} must have string keys")
            _check_json(item, f"{path}.{key}", parents)
    else:
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]", parents)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class JitCall:
    """A recipe factory's runtime result. This object is never serialized.

    ``launcher`` is a @jit function or bound @jit method. The factory supplies
    example arguments or compile placeholders, including any tensor layout and
    constexpr information needed to recreate the specialization.
    """

    launcher: Any
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


class JitPlan:
    """An ordered, deduplicated list of named factories and JSON parameters.

    Applications explicitly record recipes with add(); factories reconstruct
    launchers and arguments from application code at replay time. Only the
    supplied JSON parameters are saved; runtime JitCall objects and compiled
    artifacts are never serialized. Record metadata, not tensor contents or
    process-specific device addresses, in the parameters.
    """

    def __init__(self):
        self._entries: list[dict[str, Any]] = []
        self._keys: set[str] = set()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, recipe: str, /, **parameters: Any) -> None:
        """Record one factory call; identical recipe/parameter pairs coalesce."""
        if type(recipe) is not str or not recipe.strip():
            raise ValueError("A recipe must have a nonempty string name")
        _check_json(parameters)
        key = _encode({"recipe": recipe, "parameters": parameters})
        with self._lock:
            if key not in self._keys:
                self._entries.append(json.loads(key))
                self._keys.add(key)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible snapshot."""
        with self._lock:
            return {"schema_version": 1, "entries": json.loads(_encode(self._entries))}

    def save(self, path: str | Path) -> None:
        """Atomically write only the compact recipe list."""
        data = (_encode(self.to_dict()) + "\n").encode("utf-8")
        with atomic_write(Path(path), mode="wb") as output:
            output.write(data)

    @classmethod
    def load(cls, path: str | Path) -> "JitPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if type(data) is not dict or set(data) != {"schema_version", "entries"}:
            raise ValueError("Expected a JitPlan schema_version and entries")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported JitPlan schema_version")
        if type(data["entries"]) is not list:
            raise ValueError("JitPlan entries must be a list")
        plan = cls()
        for entry in data["entries"]:
            if type(entry) is not dict or set(entry) != {"recipe", "parameters"}:
                raise ValueError("Each JitPlan entry needs recipe and parameters")
            if type(entry["parameters"]) is not dict:
                raise ValueError("Recipe parameters must be an object")
            plan.add(entry["recipe"], **entry["parameters"])
        return plan

    def compile(self, factories: Mapping[str, Callable[..., JitCall]]) -> list["CompiledFunction"]:
        """Compile recipes without initializing execution engines or launching.

        For compilation on a CPU-only host, select the target architecture and
        have factories supply compile placeholders. Local disk caching follows
        the existing FlyDSL environment configuration. This returns callables
        even with COMPILE_ONLY=1, like flyc.compile(..., compile_only=True).
        """
        return self._execute(factories, preload=False)

    def preload(self, factories: Mapping[str, Callable[..., JitCall]]) -> list[Optional["CompiledFunction"]]:
        """Compile and initialize engines in the worker after device selection.

        No compiled launcher is invoked. COMPILE_ONLY=1 retains compilation-only
        behavior and returns None entries, just like JitFunction.preload().
        """
        return self._execute(factories, preload=True)

    def _execute(self, factories, *, preload):
        from .._mlir import ir
        from .jit_function import _unwrap_jit_function

        if ir.Context.current is not None:
            raise RuntimeError("JitPlan cannot run during DSL tracing")
        if not isinstance(factories, Mapping):
            raise TypeError("JitPlan factories must be a mapping")
        entries = self.to_dict()["entries"]
        resolved = []
        # Resolve names before calling any factory or beginning compilation.
        for entry in entries:
            name = entry["recipe"]
            factory = factories.get(name)
            if not callable(factory):
                raise ValueError(f"Missing callable JIT recipe factory: {name!r}")
            resolved.append(factory)
        result = []
        for entry, factory in zip(entries, resolved):
            call = factory(**entry["parameters"])
            if not isinstance(call, JitCall):
                raise TypeError(f"JIT recipe {entry['recipe']!r} must return a JitCall")
            jf, bound_self = _unwrap_jit_function(call.launcher)
            if jf is None:
                raise TypeError(f"JIT recipe {entry['recipe']!r} must supply a @jit launcher or bound method")
            if not isinstance(call.args, (tuple, list)) or not isinstance(call.kwargs, Mapping):
                raise TypeError("JitCall needs positional arguments and a keyword mapping")
            if any(type(key) is not str for key in call.kwargs):
                raise TypeError("JitCall keyword names must be strings")
            args = tuple(call.args)
            if bound_self is not None:
                args = (bound_self, *args)
            kwargs = dict(call.kwargs)
            if preload:
                result.append(jf.preload(*args, **kwargs))
            else:
                result.append(jf._compile_or_execute(args, kwargs, execute=False, return_compiled=True))
        return result

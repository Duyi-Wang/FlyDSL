Compiler and pipeline
=====================

FlyDSL includes a JIT compiler that traces Python kernel functions into MLIR
and lowers them through the Fly dialect pipeline to GPU binaries.

``@flyc.kernel`` and ``@flyc.jit``
------------------------------------

The primary API for defining and compiling kernels:

.. code-block:: python

   import flydsl.compiler as flyc
   import flydsl.expr as fx

   @flyc.kernel
   def my_kernel(A: fx.Tensor, B: fx.Tensor, n: fx.Constexpr[int]):
       tid = fx.thread_idx.x
       bid = fx.block_idx.x
       # ... kernel body using layout ops ...

   @flyc.jit
   def launch(A: fx.Tensor, B: fx.Tensor, n: fx.Constexpr[int],
              stream: fx.Stream = fx.Stream(None)):
       grid_x = (n + 255) // 256
       my_kernel(A, B, n).launch(
           grid=(grid_x, 1, 1),
           block=(256, 1, 1),
           stream=stream,
       )

- ``@flyc.kernel`` compiles the function body into a ``gpu.func`` inside a
  ``gpu.module``. It uses AST rewriting to trace Python code into MLIR IR.
- ``@flyc.jit`` wraps a host-side function that constructs and launches kernels.
  On first call it triggers JIT compilation; subsequent calls with the same type
  signature use a cached compiled artifact.
- ``launch.preload(*args)`` compiles or reads the cached specialization and
  initializes its execution engine without invoking the launcher. Call it in
  the worker after selecting its GPU; later calls reuse the loaded artifact.
  It returns a ``CompiledFunction`` and does not change process-wide flags.
  Use the returned callable to also skip tracing on subsequent calls.
  With ``COMPILE_ONLY=1``, it compiles without creating an engine and returns
  ``None`` instead.

``JitPlan`` saves a small JSON list of recipe names and parameters, rather than
shipping compiled cache files. Record recipes explicitly where the application
constructs its kernel variants. At startup, application-provided factories
reconstruct the launchers and their arguments as ``JitCall`` objects:

.. code-block:: python

   import torch

   plan = flyc.JitPlan()
   plan.add("my_launch.v1", n=256)
   plan.add("my_launch.v1", n=512)
   plan.save("jit-plan.json")

   def rebuild_launch(n):
       # Reconstruct all tensor type/layout and constexpr information here.
       # These CPU placeholders are never passed to a kernel invocation.
       a = torch.empty((n,), dtype=torch.float32, device="cpu")
       b = torch.empty((n,), dtype=torch.float32, device="cpu")
       return flyc.JitCall(launch, (a, b, n, fx.Stream(None)))

   plan = flyc.JitPlan.load("jit-plan.json")
   factories = {"my_launch.v1": rebuild_launch}
   compiled = plan.compile(factories)  # no execution engine or kernel launch
   # Or, inside each worker after selecting its GPU:
   ready = plan.preload(factories)     # also initialize execution engines

Identical recipe/parameter entries are deduplicated. Parameters must be plain
JSON values; tensor data, pointers, Python code and closures are not saved.
Factories must come from compatible application code and reproduce all
compilation inputs, including bound instances or closures when needed. The list
must also account for compile hints used by the application. It does not
automatically discover kernel variants. Missing factory names are
rejected before compilation, and compilation errors propagate to the caller.

A plan moves compilation to startup; it does not eliminate compilation on a
fresh machine. Existing local cache hits still work. To avoid writing local
binary caches too, set ``FLYDSL_RUNTIME_ENABLE_CACHE=0`` and
``FLYDSL_RUNTIME_RUN_ONLY=0`` before starting the worker. The plan never changes
these process-wide settings. Keep and reuse the returned callables, or have
factories return the same JIT function instances used later by the model;
separate launcher instances cannot share in-memory artifacts without disk cache.

For automatic recording without application factories, set
``FLYDSL_RUNTIME_RECORD_JIT=1`` before running the workload. Only successful
JIT compilation misses create records; memory hits, disk hits (including hits
after waiting for another compiler), and failed compilations do not. Each
launcher specialization is deduplicated by its compilation key and atomically
written to ``$FLYDSL_RUNTIME_CACHE_DIR/jit-records/<key>.json.gz``. The default
directory is ``~/.flydsl/cache/jit-records``. Multiple workers can share it.
Recording also works with ``FLYDSL_RUNTIME_ENABLE_CACHE=0``: only the records,
not compiled binary caches, are written in that case.

These automatic records contain compressed, pre-lowering MLIR and compilation
settings, rather than just recipe parameters. This captures closures and kernel
variants without importing the original application during replay. It contains
no compiled device binary or runtime tensor storage, but does include constants
embedded in the compilation input. Its size depends on the IR and is not
guaranteed to be smaller than a compiled cache. The explicit ``JitPlan`` API
above remains the smaller option when application factories are available.
Copy only the ``jit-records`` directory to the next worker. For automatic replay,
set ``FLYDSL_RUNTIME_REPLAY_JIT=1``. Each process checks the directory before its
first outer JIT/compile call and compiles the recorded variants before continuing
that call. No replay runs at Python import time. FlyDSL keeps the replay handle
alive internally, so ``FLYDSL_RUNTIME_ENABLE_CACHE=0`` needs no application code.
Automatic replay only compiles; execution engines are initialized on use.

The check runs once per process, cache directory, target and compilation/cache
configuration. A missing or empty directory continues with ordinary JIT and is
not rescanned on later calls in that scope. Consequently ``RECORD_JIT=1`` and
``REPLAY_JIT=1`` can be used together: the first cold run records, and the next
process replays. New variants still compile and can be recorded normally.
Concurrent callers wait for the same startup batch. Forked children reset the
automatic replay state and locks. Invalid records or compilation failures stop
the call and remain failed for that scope; fix the cause and restart the worker.
This moves batch compilation to the first FlyDSL use, usually model loading or
warmup; if the first use occurs in a serving request, that request pays the cost.

Alternatively, explicitly trigger replay at a chosen startup point:

.. code-block:: python

   # Defaults to the configured cache directory's jit-records subdirectory.
   startup_jit = flyc.replay_jit_records()
   # Or, after GPU selection, also initialize execution engines:
   # startup_jit = flyc.replay_jit_records("/path/to/jit-records", preload=True)

Keep ``startup_jit`` alive for the worker's lifetime. Normal JIT calls trace and
reuse matching replayed artifacts, even with disk caching disabled and with
newly reconstructed launcher instances. Replay respects existing local cache
hits, ``RUN_ONLY`` and ``COMPILE_ONLY``; it does not launch kernels or change
environment variables. Recompilation still costs time at startup.

Replay requires matching compiler/toolchain, target and compilation environment;
it validates every record before beginning compilation. Treat records as trusted
kernel source code. Extern-linked kernels, post-load initializers and ktrace
event registration require application factories: their JIT occurrences are
recorded with an explicit unsupported reason, and automatic replay rejects them
instead of omitting initialization. Failed record writes emit a warning without
discarding the successful compilation. Recording is off by default; enabling it
after a cache is already warm does not backfill entries.

Compilation flow
-----------------

On first call, ``@flyc.jit`` runs the following pipeline:

1. **AST rewriting**: The Python source is parsed and rewritten to emit MLIR ops.
2. **MLIR module construction**: The kernel body is traced into ``fly``, ``gpu``,
   ``arith``, ``scf``, ``memref``, and ``vector`` dialect ops.
3. **Fly pass pipeline**: The module is lowered through three pass stages,
   defined in ``RocmBackend._pipeline_parts()``
   (``python/flydsl/compiler/backends/rocm.py``). See
   :doc:`../architecture_guide` §3 for the per-pass table.

   A. ``pre_binary_fragments`` (Fly → ROCDL):

      - ``fly-canonicalize``
      - ``fly-rocdl-expand-ops``
      - ``fly-layout-lowering``
      - ``fly-int-swizzle-simplify``
      - ``canonicalize``
      - ``fly-convert-atom-call-to-ssa-form``
      - ``fly-promote-regmem-to-vectorssa``
      - ``convert-fly-to-rocdl``
      - ``canonicalize``
      - ``gpu.module(convert-scf-to-cf, cse, convert-rocdl-fastmath-ops, convert-gpu-to-rocdl{chipset=gfxNNN ...}, fly-rocdl-cluster-attr)``

   B. ``binary_prep_fragments`` (→ LLVM):

      - ``rocdl-attach-target{chip=gfxNNN ...}``
      - ``convert-scf-to-cf``
      - ``convert-cf-to-llvm``
      - ``gpu-to-llvm{use-bare-pointers-...=true}``
      - ``convert-vector-to-llvm``
      - ``convert-arith-to-llvm``
      - ``convert-func-to-llvm``
      - ``reconcile-unrealized-casts``
      - ``ensure-debug-info-scope-on-llvm-func`` (optional, gated by ``FLYDSL_DEBUG_ENABLE_DEBUG_INFO``)

   C. ``binary_fragment``:

      - ``gpu-module-to-binary{format=fatbin opts="..."}``

4. **Cached artifact**: The compiled binary is cached to disk
   (``~/.flydsl/cache/``) keyed by the compiler toolchain hash and kernel
   type signature.

Ahead-of-time C export
-----------------------

``CompiledFunction.export_to_c(file_path, file_name, function_prefix="")``
writes a target-specific object file and a C/C++ header. The object contains
the GPU binary and a small backend adapter, so the deployed executable or
shared library does not need FlyDSL or Python. The generated header records the
backend and system link flags; for ROCm these are
``-lamdhip64 -pthread -ldl``. The linker must also be able to find the HIP SDK
library. For a nonstandard ROCm installation,
add its library directory, for example::

   cc -shared -o libkernel.so kernel.o -L/path/to/rocm/lib -lamdhip64 -pthread -ldl

The metadata records ``libamdhip64.so`` as a link-time name. The versioned
runtime dependency (the ELF ``DT_NEEDED`` name) is determined by the library
selected when the final executable or shared library is linked. The deployed
system's dynamic loader must be able to find that versioned HIP library.

The header exposes two calling styles:

* ``<symbol>__module_init``, ``<symbol>__module_load`` and
  ``<symbol>__module_unload`` provide explicit lifecycle and device control.
  After loading, ``<symbol>_call`` provides a typed wrapper around the packed
  entry point.
* ``<symbol>_call_auto`` is the simple path. It idempotently initializes the
  module and loads it on the current device before each call. It intentionally
  does not unload automatically, so asynchronous launches remain safe.

The current exporter targets 64-bit little-endian Linux ELF hosts and uses
GNU-compatible relocatable linking and ``objcopy`` tools. The ROCm adapter
requires HIP and pthread. Exported objects are tied to the host ABI and GPU
target used during compilation. They can be moved and linked independently,
but must run on a compatible backend and GPU architecture.

On Windows x64, ``export_to_c`` writes a COFF ``<file_name>.obj`` instead and
needs no external tools: the adapter is compiled into the object itself. Link
it with MSVC ``cl`` (C callers need ``/std:c11``) or ``clang-cl`` against the
HIP SDK's ``amdhip64.lib``, the link flag the header records::

   cl /std:c11 main.c vector_add.obj amdhip64.lib /link /LIBPATH:<rocm>\lib

``<rocm>`` is the ROCm SDK root, for example ``rocm-sdk path --root`` for
TheRock wheels. The program loads ``amdhip64_7.dll`` at run time. Like objects
compiled with ``/Zl``, the exported object names no C runtime library and takes
the one the program's other objects use; a DLL linked from exported objects
alone needs one named, for example ``msvcrt.lib``. Exports into one directory
take turns on an empty ``.flydsl_export.lock`` file that stays in that directory.

Tensor arguments
-----------------

Use ``flyc.from_dlpack`` to convert PyTorch tensors into FlyDSL tensor
descriptors with layout metadata:

.. code-block:: python

   import torch

   import flydsl.compiler as flyc

   A = torch.randn(1024, device="cuda", dtype=torch.float32)
   B = torch.empty_like(A)
   tA = flyc.from_dlpack(A).mark_layout_dynamic(
       leading_dim=0, divisibility=4
   )
   tB = flyc.from_dlpack(B)
   launch(tA, tB, A.numel(), stream=torch.cuda.Stream())

Precompilation and C export
---------------------------

``flyc.compile(launcher, *specialization_args, **specialization_kwargs)``
compiles a ``@flyc.jit`` launcher for one argument signature and returns a
``CompiledFunction``. The result is callable with new runtime values of the
same signature, and ``Constexpr`` values remain fixed to the specialization
used at compile time.

Calls to the returned ``CompiledFunction`` dispatch directly without constructing
or looking up a JIT cache key. ``flyc.compile`` itself uses the existing JIT
specialization cache.

The same compiled specialization can be exported as a position-independent
host object with a generated C header:

.. code-block:: python

   from pathlib import Path

   output = Path("build/aot")
   output.mkdir(parents=True, exist_ok=True)

   compiled = flyc.compile(launch, tA, tB, A.numel())
   compiled.export_to_c(
       file_path=output,
       file_name="vector_add",
       function_prefix="flydsl_vector_add",
   )

``export_to_c`` writes ``vector_add.o`` and ``vector_add.h`` into the existing
output directory. The object embeds the FlyDSL backend adapter; link the final
binary against the backend system libraries listed in the header. The header
declares the packed entry point, typed inline call helpers, module
initialization/load/unload functions, and embedded ABI metadata. When
``function_prefix`` is omitted, ``file_name`` is also used as the exported C
symbol.

The lowered IR is parsed and its export ABI validated on the first
``export_to_c`` call. Each ``CompiledFunction`` keeps its own argument metadata
for that export without retaining the example tensors passed to ``compile``.

``flyc.compile(launcher, *args, compile_only=True)`` compiles without executing
the launcher for any arguments, including live device arguments, and always
returns a ``CompiledFunction``, also under ``COMPILE_ONLY=1``. Call the returned
function to execute it. ``compile_only`` is reserved by ``flyc.compile`` and is
not forwarded to the launcher. Without it, ``flyc.compile`` behaves as described
below.

Passing live device arguments to ``flyc.compile`` preserves its normal initial
launch, including when other pointer arguments are null to disable optional
inputs. For an offline or CPU-only build host, use null pointer wrappers created
with ``flyc.from_c_void_p`` (or compatible non-device tensor placeholders) and
set the target architecture explicitly, for example with ``ARCH=gfx950``.
Null pointers suppress the initial launch only when no live device buffers are
present. Non-device tensor placeholders still suppress execution even alongside
device buffers. ``COMPILE_ONLY=1`` explicitly disables execution for all argument
combinations and returns ``None`` instead of a callable.
Export currently supports tensor, pointer, scalar, structure, and stream
arguments whose lowered types have a supported C ABI. Unsupported launchers
report the export error when ``export_to_c`` is called.

ROCDL operations
-----------------

The ``flydsl.expr.rocdl`` module provides AMD-specific operations:

The universal exported entries follow :doc:`../api_stability`. The gfx1250
``WMMAScale`` and TDM helpers below are target-specific implementation APIs and
remain unstable.

- **fx.rocdl.make_buffer_tensor** -- create buffer resource descriptor from tensor (CDNA buffer copy)
- **fx.rocdl.BufferCopy32b** / **BufferCopy128b** -- buffer copy atoms
- **fx.rocdl.MFMA** -- MFMA instruction atoms (CDNA3/CDNA4; for example, ``MFMA(16, 16, 4, fx.Float32)``)
- **fx.rocdl.WMMA** / **fx.rocdl.WMMAScale** -- wave32 WMMA MMA atoms; ``WMMA`` is arch-dispatched (gfx11 / gfx120x RDNA4 / gfx1250), ``WMMAScale`` is the gfx1250 E8M0 MX-scaled form
- **fx.rocdl.cdna5.make_tiled_tdm_atom** / **fx.rocdl.cdna5.tdm_partition** -- gfx1250 TDM tiled copy atom (preferred for new kernels); built over the whole tensor, cut per warp, K-loop advanced by an i32 tile index on the coordinate rest
- **fx.rocdl.make_tdm_atom** / **fx.rocdl.TDM** -- **legacy** gfx1250 TDM async Global↔LDS whole-tile copy atom (1–5D; base from the copy operand, per-dim extent/stride/imm_offset/mask as atom state)

fly-opt CLI
------------

The ``fly-opt`` tool is a command-line interface for running MLIR passes on
``.mlir`` files:

.. code-block:: bash

   fly-opt --fly-canonicalize input.mlir
   fly-opt --fly-layout-lowering input.mlir
   fly-opt --help

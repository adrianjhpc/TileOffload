# TileOffload

TileOffload is an directive-based programming model for moving
Fortran loop kernels to GPUs and other accelerators. Its source model is
deliberately similar to OpenACC and OpenMP offload, while its compiler path is
built directly into LLVM Flang.

TileOffload lowers recognised Fortran computations to tile operations. The
Triton backend generates NVIDIA CUDA and AMD HIP/ROCm device code. An additional
experimental **direct CUDA Tile backend** emits CUDA Tile IR and cubin images
without routing those kernels through Triton. Unsupported CUDA Tile kernels
can fall back to Triton within the same source compilation.

Kernel recognition, launch planning, metadata, and the host runtime interface
are shared across backends. Backend support is deliberately checked per kernel;
not every construct supported by Triton is available through CUDA Tile.

The TileOffload compiler changes currently live on the
[TileOffload branch of the LLVM fork](https://github.com/adrianjhpc/llvm-project/tree/TileOffload).
The compiler driver lives in a separate repository.

> [!IMPORTANT]
> TileOffload is a prototype. The accepted Fortran subset, directive syntax, runtime
> ABI, and generated JSON may change. The compiler source and regression tests
> are the authoritative contract.

## Contents

- [Current capabilities](#current-capabilities)
- [Implementation and validation status](#implementation-and-validation-status)
- [Compilation pipeline](#compilation-pipeline)
- [GPU targets](#gpu-targets)
- [Building the toolchain](#building-the-toolchain)
- [Quick start](#quick-start)
- [Programming model](#programming-model)
- [Directive reference](#directive-reference)
- [Data movement and lifetime](#data-movement-and-lifetime)
- [Supported kernels](#supported-kernels)
- [Expressions and scalar values](#expressions-and-scalar-values)
- [Reductions](#reductions)
- [Types, ranks, and storage](#types-ranks-and-storage)
- [Compilation and linking](#compilation-and-linking)
- [Compiler-driver reference](#compiler-driver-reference)
- [Runtime configuration](#runtime-configuration)
- [Runtime device selection and MPI](#runtime-device-selection-and-mpi)
- [Compiler and runtime architecture](#compiler-and-runtime-architecture)
- [Backend artifact contract](#backend-artifact-contract)
- [Performance tuning and profiling](#performance-tuning-and-profiling)
- [Testing](#testing)
- [Extending TileOffload](#extending-tileoffload)
- [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting)
- [Known limitations](#known-limitations)

## Current capabilities

TileOffload currently provides:

- `parallel` lowering for recognised one- and two-dimensional Fortran loops;
- arbitrary runtime loop lower and upper bounds for elementwise, stencil, and
  reduction kernels, including recognised nonzero constant and runtime steps;
- descriptor-based host launches through ABI v3 by default, with explicit v2
  compatibility and variadic array/scalar bindings;
- one or more output assignments in a recognised loop;
- `real(4)`, `real(8)`, and signed `integer(1|2|4|8)` device expressions;
- affine induction-variable expressions, including the common
  integer-to-real conversion used by mesh initialisation loops;
- two-dimensional stencils with halo offsets, mixed-rank coordinate arrays,
  conditional affine indices, and supported fixed nested-loop expansions;
- specialised single- and double-precision two-dimensional matrix
  multiplication with arbitrary recoverable loop bounds and supported signed steps;
- one-dimensional sum, dot-product, product, minimum, and maximum reductions;
- fused two-dimensional multi-result reductions using one common reduction
  operator;
- hierarchical, multi-kernel device reductions with reusable workspaces;
- one- and multi-warp Triton reduction lowering;
- persistent data regions with nested ownership;
- `copyin`, `create`, `copyout`, `delete`, `present`, `update`, and `release`
  data operations;
- derived-component data designators such as
  `chunk%tiles(1)%field%density0`;
- `no_copyback` launch behavior for device-resident outputs;
- opt-in asynchronous resident launches with `TILEOFF_ASYNC_RESIDENT=1`;
- explicit synchronization with `!$tileoff wait`;
- IEEE-default FP32 matmul and explicit `matmul_precision(tf32|tf32x3)`;
- explicit-shape, assumed-shape, pointer/heap-backed, and allocatable arrays
  when their storage is contiguous;
- multiple separately compiled embedded TileOffload bundles in one executable;
- initial end-to-end NVIDIA CUDA and AMD HIP/ROCm target support;
- embedded PTX or CUDA Tile cubin images for CUDA and HSACO images for HIP;
- runtime device enumeration and selection through the `TileOffload` module;
- precision-aware CUDA matmul pipeline selection and explicit packed-A recognition;
- per-device and per-accelerator-context runtime state; and
- a backend-neutral kernel plan and artifact contract, with Triton as the
  broadest backend and a narrower direct CUDA Tile implementation.

Compiler-generated TF32 A packing is available as an **experimental opt-in
patch**, described below. Its CPU checks pass; full compiler/toolchain and GPU
validation have not yet been reported for that patch. Do not confuse it with
the already measured explicit packing benchmarks.

## Implementation and validation status

| Area | Current status |
| --- | --- |
| CUDA/Triton kernels, residency, reductions, v3 launches | Exercised by the benchmark suite and CloverLeaf; correctness must still be checked for each compiler/target configuration. |
| Arbitrary bounds and constant non-unit/negative steps | Implemented; bounds, tails, empty ranges and untouched-output tests have passed in development. |
| Runtime step values | Implemented in the supplied compiler/runtime sources; steps must be launch-invariant, nonzero signed 32-bit values. Backend restrictions still apply. |
| Direct CUDA Tile | Implemented for a limited pointwise and rank-1 reduction subset; BabelStream has run through this path. Not a replacement for all Triton lowering. |
| HIP/ROCm | End-to-end implementation is present; the NVIDIA measurements here do not validate AMD correctness or performance. |
| Runtime device selection | Public device-count/get/set interfaces and per-context state are implemented. This is not automatic MPI distribution or data migration. |
| Explicit packed-A matmul | Recognises AP(k,i) times B(k,j); measured pack-plus-matmul tests produced correct results and demonstrated potential benefits. |
| Automatic A packing | New opt-in compiler/runtime/driver patch; CPU mock and emitter checks pass. Full Flang/Triton build and GPU execution remain to be validated. |

The most recent pre-automatic-packing benchmark snapshot reported **zero
nonzero-error rows across 80 two-dimensional and 112 one-dimensional results**.
These are results for the measured configurations, not a guarantee for every
supported type, target, backend or loop shape. A tuned long CloverLeaf run was
reported at approximately 192 s for TileOffload, 199 s for CUDA and 201 s for
OpenACC; retain the workload and toolchain settings when reproducing or citing
that comparison.

The supplied sources and patch packages may be ahead of an installed build.
Check that a feature's compiler, driver and runtime changes are all applied;
this README does not assert that every development patch has been merged or
released.

## Compilation pipeline

The implemented end-to-end paths are:

```text
Fortran + !$tileoff
  -> Flang parse tree and semantics
  -> FIR with TileOffload.launch and data operations
  -> kernel recognition and backend-neutral planning
  -> per-kernel backend selection
       Triton:
         TTIR -> TTGIR -> LLVM MLIR -> LLVM IR
         CUDA: NVPTX/libdevice -> PTX
         HIP:  AMDGPU/ROCm device libraries -> object -> HSACO
       Direct CUDA Tile:
         CUDA Tile IR -> cuda-tile-translate -> Tile IR bytecode
         -> tileiras -> cubin
  -> embedded typed device-image/JSON bundle
  -> host object + matching CUDA Driver API or HIP runtime
```

A CUDA bundle can contain both Triton PTX and CUDA Tile cubin kernels. Fallback
is a compile-time backend choice, not a silent retry of an incorrectly running
GPU kernel. Host ABI v3 is independent of that choice.

The recogniser is intentionally fail-closed. A `parallel` region is compiled
only when every relevant operation can be represented in the selected kernel
plan. Unsupported work is not silently discarded or left to execute on the
host.

## GPU targets

The code-generation backend and accelerator target are separate choices.
`--tileoff-backend triton` selects the kernel code generator;
`--tileoff-target cuda|hip` selects the device toolchain, image format, and host
runtime.

| Target | Architecture example | Subgroup width | Embedded image | Runtime library |
| --- | --- | --- | --- | --- |
| `cuda` / Triton | `sm_90a` | `32` | PTX | `FortranTileOffloadRuntime` + CUDA Driver API |
| `cuda` / CUDA Tile | Architecture accepted by installed `tileiras` | Backend-managed | cubin | `FortranTileOffloadRuntime` + CUDA Driver API |
| `hip` | `gfx90a`, `gfx942` | `64` for `gfx9*` by default; otherwise `32` | HSACO | `FortranTileOffloadRuntimeHIP` + `libamdhip64` |

CUDA remains the default target for compatibility. Select AMD explicitly:

```sh
tileoffload-flang --tileoff-target hip --tileoff-gpu-arch gfx942 ...
```

`--tileoff-target` also accepts `nvidia`, `amd`, and `rocm` as normalized
spellings. `--tileoff-sm` remains the CUDA compatibility option, while
`--tileoff-amd-arch` is an AMD compatibility spelling for
`--tileoff-target hip --tileoff-gpu-arch ARCH`.

All tileoffload-bearing objects linked into one executable must target the same
accelerator platform. The final link invocation must use the same
`--tileoff-target` so that the driver selects the matching runtime library.

## Building the toolchain

### Required tools

The Triton path needs a TileOffload-enabled LLVM/Flang build and a matching
Triton/LLVM lowering toolchain:

```sh
export LLVM_BUILD=/path/to/llvm-project/build
export TRITON_OPT=/path/to/triton-opt
export MLIR_TRANSLATE=/path/to/triton-llvm/build/bin/mlir-translate
export LLC=/path/to/triton-llvm/build/bin/llc
export LLVM_LINK="$LLVM_BUILD/bin/llvm-link"
export OPT="$LLVM_BUILD/bin/opt"
```

For CUDA, make the CUDA Driver API and libdevice installation discoverable:

```sh
# Optional when libcuda is not in the default linker search path.
export CUDA_LIB_DIR=/usr/local/cuda/lib64

# Usually auto-detected; set this only when necessary.
export TILEOFF_CUDA_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
```

For HIP, provide a ROCm installation and `ld.lld`:

```sh
export ROCM_PATH=/opt/rocm
export LD_LLD="$ROCM_PATH/llvm/bin/ld.lld"
```

Configure one or both runtime targets:

```sh
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;flang" \
  -DFLANG_TILEOFF_RUNTIME=ON \
  -DFLANG_TILEOFFLOAD_RUNTIME_BACKEND=BOTH
```

`FLANG_TILEOFFLOAD_RUNTIME_BACKEND` accepts `CUDA`, `HIP`, or `BOTH` and defaults to
`CUDA`. A HIP or `BOTH` build must find `hip/hip_runtime_api.h` and
`libamdhip64`; set `ROCM_PATH`, `HIP_PATH`, `HIP_DRIVER_INCLUDE_DIR`, or
`HIP_DRIVER_LIBRARY` when they are outside normal locations. A CUDA or `BOTH`
build must find the CUDA driver headers and library.

The top-level runtime-enable option can vary between fork revisions; retain
the enable option used by your working checkout. The current runtime CMake
file uses `FLANG_TILEOFFLOAD_RUNTIME_BACKEND` for backend selection. Do not
assume similarly named legacy cache entries affect the current build.

Typical rebuild targets are:

```sh
cmake --build "$LLVM_BUILD" \
  --target fir-opt flang FortranTileOffloadRuntime FortranTileOffloadRuntimeHIP
```

When only one target was configured, omit the other runtime target from the
build command. These are configuration fragments: preserve the target,
assertion, compiler, and dependency settings of your working LLVM/Triton build.
With an existing Ninja build, the equivalent rebuild is:

```sh
ninja -C "$LLVM_BUILD" fir-opt flang FortranTileOffloadRuntime
```

After changing device lowering, recompile the affected Fortran source objects
and relink the application. Relinking alone does not regenerate embedded GPU
code. Changing the default host launch ABI also requires recompiling the host
launch sites. Rebuild and relink the matching runtime when its implementation
changes.

NVTX instrumentation is optional and disabled by default. A normal runtime
build does not require `nvtx3/nvtx3.hpp`. For the current runtime CMake file:

```sh
cmake -S llvm -B build \
  -DFLANG_TILEOFFLOAD_ENABLE_NVTX=ON \
  -DTILEOFFLOAD_NVTX_INCLUDE_DIR=/path/containing/nvtx3
```

CMake defines `TILEOFFLOAD_ENABLE_NVTX` on the runtime target and adds the
include/link requirements. Regenerate Ninja rules instead of editing
`build.ninja` directly.

### Direct CUDA Tile tools

For `--tileoff-backend cuda-tile`, also provide:

```sh
export TILEOFF_CUDA_TILE_TRANSLATE=/path/to/cuda-tile-translate
export TILEOFF_TILEIRAS=/path/to/tileiras
"$TILEOFF_TILEIRAS" --version
"$TILEOFF_TILEIRAS" --help
```

The defaults are `cuda-tile-translate` and `tileiras` on `PATH`. The supplied
driver requests Tile IR bytecode version `13.1`; translator, assembler and
installed driver must support the selected path. Check the assembler's actual
`--gpu-name` list. A newer compatibility document does not add architectures
to an older local assembler, and `sm_90a` is not interchangeable with `sm_90`
for every tool. Do not silently compile for `sm_80` to bypass an unsupported
architecture diagnostic.

Triton tools remain necessary when any kernel falls back to Triton. Pure CUDA
Tile kernels use the direct path, but still need Flang and host compilation
and linking tools.

## Quick start

### Vector addition with a persistent data region

```fortran
module vector_kernels
  implicit none
contains
  subroutine vector_add(n, a, b, c)
    integer, intent(in) :: n
    real, intent(in) :: a(n), b(n)
    real, intent(out) :: c(n)
    integer :: i

    !$tileoff parallel tile(256) no_copyback
    do i = 1, n
      c(i) = a(i) + b(i)
    end do
  end subroutine
end module

program example
  use vector_kernels
  implicit none
  integer, parameter :: n = 1000000
  real, allocatable :: a(:), b(:), c(:)

  allocate(a(n), b(n), c(n))
  a = 1.0
  b = 2.0

  !$tileoff enter data copyin(a, b) create(c)
  !$tileoff present(a, b, c)

  call vector_add(n, a, b, c)

  !$tileoff exit data copyout(c) delete(a, b, c)

  if (any(c /= 3.0)) error stop "validation failed"
end program
```

Compile, link, and run it on NVIDIA CUDA (the default target):

```sh
/path/to/TileOffload/bin/tileoffload-flang \
  --tileoff-target cuda --tileoff-gpu-arch sm_90a \
  example.f90 -O3 -o example
TILEOFF_DEVICE=0 ./example
```

Compile the same source for AMD HIP/ROCm:

```sh
/path/to/TileOffload/bin/tileoffload-flang \
  --tileoff-target hip --tileoff-gpu-arch gfx942 \
  example.f90 -O3 -o example
TILEOFF_DEVICE=0 ./example
```

For separate compilation:

```sh
tileoffload-flang -O3 -c kernels.f90 -o kernels.o
tileoffload-flang -O3 -c main.f90 -o main.o
tileoffload-flang main.o kernels.o -o example
```

TileOffload objects are conventional relocatable objects. Each can contain its own
embedded device bundle, and multiple bundles may be linked into the same
executable. For HIP, repeat `--tileoff-target hip` on every TileOffload compilation and
on the final link so the driver selects `FortranTileOffloadRuntimeHIP`.

## Programming model

### Tile-valued expressions

TileOffload starts from a loop nest but represents supported expressions across
a logical tile: loads, arithmetic, predicates and stores operate on collections
of iteration values. This exposes tile layouts, data reuse, dot operations and
software-pipeline opportunities to the backend.

CUDA exposes thread/block programming directly. OpenMP and OpenACC primarily
express iteration-space distribution and data placement through directives;
their compilers can also vectorise, tile and pipeline computations. The
TileOffload distinction is its explicit tile-expression lowering strategy,
not a claim that other models execute expressions serially or cannot perform
those optimisations. Cross-iteration dependencies are not made safe merely by
using a tile.

### Parallel loops

`!$tileoff parallel` applies to the immediately following `do` construct. There
is no matching end directive.

```fortran
!$tileoff parallel tile(128)
do i = lower, upper
  c(i) = a(i) + b(i)
end do
```

For a rank-two kernel, the outer loop normally represents the second Fortran
dimension and the inner loop the first:

```fortran
!$tileoff parallel tile(16, 16)
do k = y_min, y_max
  do j = x_min, x_max
    c(j, k) = alpha * a(j, k) + beta * b(j, k)
  end do
end do
```

### Loop requirements

Recognised kernels require a recoverable lower bound, upper bound and step,
one logical loop for rank one or a supported nest for rank two, and provable
array accesses. Loop bounds need not start at one, including matmul's M/N/K
loops. Array declaration bounds are tracked independently of loop bounds.

Constant steps may be positive or negative and non-unit. The runtime-step
implementation also accepts **launch-invariant signed 32-bit integer steps**:

```fortran
integer :: lower, upper, stride, i
! stride is computed on the host and must be nonzero.
!$tileoff parallel tile(256)
do i = lower, upper, stride
  c(i) = a(i) + b(i)
end do
```

Logical lane `q` maps to Fortran index `lower + q*stride`. Trip counts and
masks must therefore be based on iteration ordinals, not a raw `upper-lower`
range. A step whose sign cannot reach the bound produces an empty iteration
space. A constant zero step is rejected; a runtime zero step is diagnosed
before launch. Runtime steps depending on an offloaded induction variable or
on values computed inside the kernel are not supported.

This is not general support for arbitrary loop-carried dependencies, ragged
nests, noncontiguous host array sections or post-loop induction-variable use.
The recogniser must still prove the complete kernel safe. CUDA Tile currently
requires unit steps and can fall back to Triton for other supported loops.
Automatic A packing also retains a unit-step restriction.

### Logical tile versus hardware block size

`tile(...)` describes the logical number of elements handled by one device
program in each dimension. It is not the CUDA thread-block size or HIP workgroup
size.

The requested warp count is preserved by the updated scheduling analysis rather
than unconditionally reducing ordinary kernels to one effective warp. The default
request remains one warp unless overridden; eight warps is a tuning choice, not
a new global default. Check the driver's `effective warps` message and generated
JSON after rebuilding.

For the Triton CUDA path, the hardware block size comes from the selected
schedule:

```text
threads_per_cta = num_warps * threads_per_warp
```

CUDA currently requires `threads_per_warp=32`. A one-dimensional tile of 1024
with one warp therefore means that 32 CUDA threads cooperate to process 1024
logical elements. HIP accepts subgroup widths of `32` or `64`; the driver
defaults `gfx9*` architectures to wave64 and later architectures to wave32.

CUDA Tile manages its execution mapping separately. Triton warp settings and
`effective warps` logging do not by themselves establish CUDA Tile hardware
occupancy or thread count.

Default logical tiles in the current recogniser are:

| Kernel | Default logical tile |
| --- | --- |
| 1-D elementwise or reduction | `1024` |
| 2-D elementwise, stencil, or reduction | `16, 16` |
| f32 matrix multiplication | `16, 16, 32` |
| f64 matrix multiplication | `16, 16, 8` |

Use `TILEOFF_DEBUG=1` to print the grid, logical tile, subgroup, and hardware
block selected for each launch. The generated per-kernel JSON is the
authoritative schedule.

## Directive reference

TileOffload directives are case-insensitive. The recommended free-form sentinel is
`!$tileoff`. The frontend also accepts `!@tileoff`; fixed-form sentinel handling
depends on the Flang source form in use. The driver primarily scans
conventional TileOffload sentinels when deciding whether to run the accelerator
pipeline.

### `parallel`

```fortran
!$tileoff parallel [tile(...)] [pack(...)] [reduction(...)] [matmul_precision(...)] [no_copyback]
do ...
  ...
end do
```

| Clause | Purpose |
| --- | --- |
| `tile(x[, y[, z]])` | Compile-time logical tile shape. `z` is the reduction/K tile for matrix multiplication. |
| `pack(name:host, ...)` | Explicitly use host-visible temporary-buffer behavior for the named array arguments. |
| `pack(name:device, ...)` | Use or create cached device storage for the named array arguments. |
| `reduction(+:s)` | Additive reduction into `s`. |
| `reduction(*:s)` | Multiplicative reduction into `s`. |
| `reduction(min:s)` | Minimum reduction into `s`. |
| `reduction(max:s)` | Maximum reduction into `s`. |
| `matmul_precision(ieee\|tf32\|tf32x3)` | FP32 matmul input precision; default `ieee`. Explicit reduced-precision modes require a supported CUDA target. |
| `no_copyback` | Do not automatically copy arrays written by this launch back to the host. Keep their results in cached device storage. |

`pack` currently names simple variables, not arbitrary designators or array
sections. Data directives have broader designator support. **This placement
clause is not a matrix transpose or the automatic matmul A-packing optimisation.**

### Data and synchronization directives

| Directive | Behavior |
| --- | --- |
| `!$tileoff enter data copyin(a)` | Begin a nested data region, acquire persistent device storage for `a`, and copy its current host value on a new allocation. |
| `!$tileoff enter data create(c)` | Begin a nested data region and acquire persistent storage for `c` without initializing a new device allocation from the host. |
| `!$tileoff present(a, c)` | Assert that every listed object already has a live device allocation. It neither allocates nor transfers data. |
| `!$tileoff update device(a)` | Create or resize persistent storage as required and copy the current host value to the device. |
| `!$tileoff update host(c)` | Copy a present device allocation to the host. Fails when the object is not present. |
| `!$tileoff exit data copyout(c)` | Request a host update when the innermost region is the final owner, then end that region. |
| `!$tileoff exit data delete(a, b, c)` | Mark listed objects as belonging to the innermost region and release that region's ownership on exit. |
| `!$tileoff release(a, b)` | Release unowned persistent allocations. It cannot release an allocation still owned by an active data region. |
| `!$tileoff release all` | Release all persistent allocations when no data region is active. `release_all` is an alias. |
| `!$tileoff wait` | Wait for prior work on the active TileOffload context's stream. It performs no transfer. |

An `enter data` directive creates one ownership frame even when it contains
several clauses. Its matching `exit data` ends the innermost frame.

### Continued directives

Use normal free-form directive continuation. Repeat the sentinel on each
continued line and place `&` after the sentinel on continuation lines:

```fortran
!$tileoff enter data &
!$tileoff& copyin(chunk%tiles(1)%field%density0) &
!$tileoff& copyin(chunk%tiles(1)%field%energy0) &
!$tileoff& create(chunk%tiles(1)%field%pressure)
```

Do not continue a directive using an ordinary, non-directive source line.

### Data designators

Data directives accept Fortran variables and derived-component designators,
including component chains with scalar subscripts:

```fortran
!$tileoff enter data &
!$tileoff& copyin(chunk%tiles(1)%field%density0) &
!$tileoff& create(chunk%tiles(1)%field%energy0)

!$tileoff present(chunk%tiles(1)%field%density0)
!$tileoff update host(chunk%tiles(1)%field%energy0)

!$tileoff exit data &
!$tileoff& copyout(chunk%tiles(1)%field%energy0) &
!$tileoff& delete(chunk%tiles(1)%field%density0, &
!$tileoff&        chunk%tiles(1)%field%energy0)
```

Allocatable and pointer components are lowered through their descriptors.
Storage must still be contiguous and sizeable by the data-operation ABI.
General array-section mapping is not yet a supported contract.

## Data movement and lifetime

### Default launch behavior

When an array has no cached allocation and no device placement is requested,
the launch uses temporary device storage:

- read and read/write arrays are copied host to device;
- the kernel runs;
- written host-target arrays are copied device to host; and
- temporary storage is freed.

This is convenient for isolated kernels but expensive for repeated launches.

### Persistent placement

`pack(...:device)`, `enter data`, `create`, and `update device` can establish a
persistent allocation keyed by the host data address in the active accelerator
context.

Once an allocation is present, subsequent launches use it automatically even
when their `parallel` directives omit `pack(...:device)`. This
present-if-cached rule allows data placement to be controlled outside the
kernel procedure.

During a persistent lifetime:

- a cache miss for a read target creates storage and copies host data;
- a cache miss for a write-only target creates uninitialized storage;
- later launches reuse the allocation;
- host writes are invisible until `update device`;
- device writes are invisible on the host until an automatic copyback,
  `update host`, or final-owner `copyout`; and
- the allocation remains local to its CUDA or HIP context.

### `no_copyback`

Use `no_copyback` when the values written by a launch will be consumed by
later device work:

```fortran
!$tileoff parallel tile(16,16) no_copyback
do k = y_min, y_max
  do j = x_min, x_max
    pressure(j,k) = (1.4_8 - 1.0_8) * density(j,k) * energy(j,k)
  end do
end do
```

The clause affects written arrays only. Required inputs are still made
available on the device. Written arrays are treated as device targets, are
retained in the cache, and are not copied back automatically. Use
`update host(pressure)` or a final `copyout(pressure)` before host code reads
the result.

The selected behavior is recorded in kernel JSON as
`"copy_back_writes": false`.

### `present`

`present` is an assertion and has no OpenACC-style fallback allocation:

```fortran
!$tileoff present(density, energy, pressure)
```

It is useful at procedure boundaries to catch a missing enclosing data region.
Every object must already be present in the active accelerator context;
otherwise the runtime reports an error. `present` neither acquires
nested-region ownership nor performs a transfer.

### Nested data regions

Data regions are stack structured. An inner region may acquire new objects or
share objects already owned by an outer region:

```fortran
!$tileoff enter data copyin(a) create(c)

! Outer work may use a and c.

!$tileoff enter data copyin(a, b) create(tmp)
!$tileoff present(a, b, c, tmp)

! Inner work may use a, b, c, and tmp.

!$tileoff exit data copyout(b) delete(a, b, tmp)

! a and c are still controlled by the outer region.

!$tileoff exit data copyout(c) delete(a, c)
```

The runtime reference-counts region ownership:

- acquiring an already present object adds ownership to the inner frame
  without reallocating it;
- `copyout` on an inner reference is deferred when an enclosing owner remains;
- the host copy occurs when the final owning region requests `copyout`;
- exiting the inner region releases only ownership acquired by that frame;
- allocation storage is freed only after the final region reference is
  released; and
- `copyout` and `delete` operands must belong to the innermost region, so an
  inner exit cannot accidentally release an outer-only object.

An allocation cannot be resized while it is owned by a data region.
`release` and `release all` are for allocations outside active region
ownership; close the owning region first.

### Recommended long-lived pattern

```fortran
!$tileoff enter data copyin(a, b) create(c, work)

do step = 1, timesteps
  call kernel_a(n, a, b, work)
  call kernel_b(n, work, c)
end do

!$tileoff exit data copyout(c) delete(a, b, c, work)
```

If host code changes `a` during the region, add:

```fortran
!$tileoff update device(a)
```

before its next device consumer.

### Synchronization

The runtime owns one CUDA or HIP stream and one completion event
per accelerator context. Work submitted through the same context is ordered in
that stream.

Use `!$tileoff wait`:

- before host code that needs completion but not a host transfer;
- before external CUDA or HIP work whose stream has no explicit dependency on
  the TileOffload stream;
- as a phase boundary before changing device or context ownership; or
- in tests that require completion at a precise source point.

Host updates, final copyouts, reductions that return host scalars, releases,
and host-target launch paths are synchronization points in the current
runtime. `wait` covers only the active TileOffload context and its runtime stream; it
does not synchronize unrelated CUDA/HIP contexts or caller-created streams.

## Supported kernels

### Elementwise and multi-output kernels

Rank-one and rank-two loops may contain recognised expressions and one or more
array stores:

```fortran
!$tileoff parallel tile(128)
do i = lower, upper
  sum(i) = a(i) + b(i)
  difference(i) = a(i) - b(i)
end do
```

Both the v3 host descriptor and the retained v2 binding interface support
variadic arguments. They are not restricted to the older
three-input/three-scalar shape. Practical limits instead come from the
recogniser, generated kernel signature, and backend.

Straight-line stores may feed later expressions in the same logical
iteration. Mutable scalar temporaries are accepted only when proven private to
that iteration.

### Stencils and affine indexing

A rank-two loop is recognised as `stencil2d` when its accesses can be reduced
to supported functions of the logical induction variables and captured index
scalars. Supported patterns include:

- constant halo offsets such as `a(j-1,k)`, `a(j+1,k)`, and `a(j,k+1)`;
- arbitrary loop and array lower bounds;
- independent per-array lower bounds and element strides;
- rank-one coordinate arrays projected onto either loop dimension;
- reversed affine indices such as `x_min-j`;
- conditional affine choices for donor/upwind/downwind indices;
- supported `min`/`max` clamps in index expressions;
- conditional stores and multiple output arrays; and
- fixed, compile-time-expandable nested stencil loops such as a small
  `j:j+1`, `k:k+1` neighborhood.

Example:

```fortran
!$tileoff parallel tile(16,16)
do k = y_min, y_max
  do j = x_min, x_max+2
    if (flux(j,k) < 0.0_8) then
      upwind = j+2
      donor = j+1
      downwind = j
    else
      upwind = j-1
      donor = j
      downwind = j+1
    end if

    output(j,k) = field(donor,k) + field(upwind,k) + &
                  field(downwind,k)
  end do
end do
```

TileOffload masks the output domain. The program is responsible for declaring and
maintaining sufficient halo storage for every input offset.

Indirect gathers through arbitrary array-valued indices, data-dependent loop
bounds, and loop-carried stencil dependences remain unsupported.

### Matrix multiplication

The matrix-multiplication recogniser accepts the canonical three-loop form:

```fortran
!$tileoff parallel tile(16,16,32)
do j = 1, n
  do i = 1, m
    acc = 0.0
    do p = 1, k
      acc = acc + a(i,p) * b(p,j)
    end do
    c(i,j) = acc
  end do
end do
```

`real(4)` and `real(8)` matrices are supported. All three canonical loops may
have arbitrary recoverable lower bounds and supported nonzero signed steps.
Physical array bounds/leading dimensions remain independent of those loop
coordinates. Empty M/N ranges perform no stores; an empty K reduction in this
zero-initialised pattern writes zero to the selected C elements.

FP32 matmul uses `ieee` input precision by default. Masked matmul inputs are
zero-padded so partial tiles do not contribute undefined values. To opt into
reduced-precision tensor-core input arithmetic on supported NVIDIA GPUs:

```fortran
!$tileoff parallel tile(64,64,32) matmul_precision(tf32)
```

| Mode | Behavior |
| --- | --- |
| `ieee` | Default FP32 input precision; no opt-in to TF32 input rounding. |
| `tf32` | Explicit reduced-precision input arithmetic. |
| `tf32x3` | Three-product TF32 decomposition for improved accuracy over TF32; not a guarantee of bitwise IEEE equivalence. |

The clause applies to one recognised FP32 matmul launch. It is rejected on
FP64 matmul, elementwise kernels, and reductions. The implemented HIP backend
rejects TF32 modes. The CUDA driver requires SM80 or newer for TF32 modes;
TF32x3 also requires the corresponding Triton decomposition pass. Custom
TTIR-to-TTGIR pipelines must retain the required decomposition before matmul
acceleration. Validate numerical error for the application's inputs and
acceptance criteria; do not change validation tolerances merely to hide errors.

Select the f64 code-generation strategy with:

```sh
tileoffload-flang --tileoff-f64-matmul-strategy reduce ...
tileoffload-flang --tileoff-f64-matmul-strategy fma ...
tileoffload-flang --tileoff-f64-matmul-strategy dot ...
```

Performance and toolchain compatibility are GPU- and Triton-version
dependent. Validate numerical results and benchmark on the target system.

### Direct CUDA Tile coverage

Select it explicitly:

```sh
tileoffload-flang --tileoff-target cuda --tileoff-backend cuda-tile \
  --tileoff-gpu-arch ARCH -O3 -c kernels.f90
```

Use an `ARCH` supported by both the device and installed `tileiras`. The current
emitter supports these restricted forms:

| Pattern | Direct CUDA Tile constraints |
| --- | --- |
| Pointwise rank-1/rank-2 | Homogeneous f32/f64, unit steps, one output, supported array/scalar loads and finite constants with addition, subtraction, multiplication or squaring. |
| Rank-1 sum, dot, min, max | f32/f64, unit step, supported plain array inputs, power-of-two reduction tile from 2 through 1024, with a generated reduction stage. |
| Pointwise tile shape | Power-of-two dimensions, at most 1024 total elements. |
| Other recognised kernels | Triton fallback when enabled; otherwise a diagnostic. |

This is not blanket support for every intrinsic, conversion, multi-output
expression, stencil or reduction. The backend's `querySupport` determines
eligibility. Matmul and automatic A packing currently use Triton.

Confirm the backend in the build log and per-kernel JSON:

```text
selected backend: cuda-tile (device IR=cuda-tile-ir, image=cubin, fallback=false)
CUDA Tile: tileoff_kernel_... -> cubin
```

`selected backend: mixed` means some kernels use Triton. To require all kernels
to compile directly, add `--tileoff-no-backend-fallback`. Generic log headings
such as “Lower Triton kernels” may still appear when there are no Triton kernels
to process; they are not evidence of fallback by themselves.

### Matmul pipeline policy

`TILEOFF_MATMUL_PIPELINE` is a **build-time** driver control, separate from
runtime async execution:

| Value | CUDA matmul lowering |
| --- | --- |
| `0` | Disable the optional software-pipeline pass sequence. |
| `1` | Request that sequence using the configured stage count. |
| `auto` | Current driver default. On SM90, use baseline lowering for IEEE FP32 and request three stages for FP64 dot and TF32. Other paths retain baseline unless explicitly selected. |

The driver also checks for dot operations: FP64 reduce/FMA strategies do not
become pipelined dot kernels simply by setting a stage count. One effective
stage selects baseline lowering. `TILEOFF_MATMUL_F64_PIPELINE_STAGES` and
`TILEOFF_MATMUL_TF32_PIPELINE_STAGES` override the corresponding auto choices.
These are policies informed by the measured H100 cases, not universal optimum
settings. The current experimental driver pipeline is limited to SM80–SM90;
check the driver before selecting it for other architectures.

```sh
TILEOFF_MATMUL_PIPELINE=auto tileoffload-flang \
  --tileoff-gpu-arch sm_90a --tileoff-num-warps 4 \
  --tileoff-num-stages 3 -O3 -c matmul_kernel.f90
```

Check `CUDA matmul policy:` in the log for precision class, configured stages
and effective stages. Changing a runtime environment variable after compilation
does not regenerate the pipeline. A `-DTILEOFF_NUM_WARPS=4` or similar Fortran
preprocessor definition does not select the driver warp count; use
`--tileoff-num-warps` or the build-system setting that passes that option.

The SM90a TF32 correctness issue observed during development was resolved in
the revised lowering. Retain the required fencing/pipeline pass order and test
both full and partial tiles when changing it. Large structured mismatches are
not an acceptable consequence of TF32 rounding.

### Explicit packed-A matmul

The Triton recogniser also accepts the product `ap(p,i)*b(p,j)` (and the reversed
multiply operands) for a matmul whose A input is already transposed into
AP(K,M). Physical descriptor dimensions are preserved; this is a storage-layout
choice, not a change to the mathematical multiplication. Users managing this
layout explicitly must populate AP and keep it current themselves.

`TILEOFF_MATMUL_ALIGNED_LOADS=1` enables the experimental guarded alignment
specialisation at build time. The guard checks relevant alignment/layout
conditions before applying load hints; the general path remains available.
Do not substitute unconditional contiguity or divisibility assertions for
those checks.

### Automatic A packing — experimental patch

The latest development patch removes the need for a user-written pack loop
for eligible **CUDA/Triton TF32 matmul with constant unit steps**. It emits:

1. the original matmul for fallback and baseline comparison;
2. a tiled A-packing kernel; and
3. a matmul consuming AP(Kpad,M), with Kpad rounded up to 16 elements.

B/C layouts and requested numerical precision are unchanged. Packing runs
before every selected matmul on the same stream. Only context-local scratch
capacity is cached; packed contents are never assumed current across calls.
The driver records the exact shared-memory requirements of the new kernels.
The compiler, runtime and driver patches must be used together.

**Status:** patch application checks, runtime mock tests, emitter-fragment
compilation and driver metadata checks pass. Full LLVM/Flang compilation,
real Triton lowering and GPU execution have not yet been validated for this
patch. Candidate generation is opt-in, not a new global default.

```sh
# Build-time: emit candidates. Use the updated compiler, runtime and driver.
TILEOFF_MATMUL_PACK_A=auto TILEOFF_MATMUL_PIPELINE=auto \
  cmake --build build --clean-first --target tileoffload-matmul-2d-tf32

# Runtime: compare the two policies using the SAME candidate-enabled binary.
exe=build/benchmarks/tileoffload/tileoffload-matmul-2d-tf32/tileoffload-matmul-2d-tf32
TILEOFF_ASYNC_RESIDENT=1 TILEOFF_MATMUL_PACK_A=0 "$exe" 1000 1000 100
TILEOFF_ASYNC_RESIDENT=1 TILEOFF_MATMUL_PACK_A=auto "$exe" 1000 1000 100
```

At build time, `auto` or `1` emits eligible candidates; unset/`0` does not.
For a candidate-enabled binary, runtime policy is:

| Policy | Behavior |
| --- | --- |
| `0` | Original matmul only. |
| `auto` or unset | Pack supported shapes when M, N and K are all at least 512. |
| `1` | Force packing for supported nonempty shapes, retaining safety checks. |

The threshold is provisional. Unsupported layouts, overlapping host allocation
ranges and scratch allocation failure fall back to the original path. Extra
scratch is `4*M*round_up(K,16)` bytes per context. This initial implementation
does not cover IEEE FP32, FP64, TF32x3, cuTile, HIP or non-unit/runtime steps.
Setting the runtime variable cannot add missing kernels to an old binary.

Use `TILEOFF_DEBUG=1` once to confirm an `auto-pack A kernel=...` message; disable
debug output for timings. Run the patch package's `tests/run_gpu.sh` first.
Earlier manually packed results demonstrate potential, not a measured speedup
for this new automatic implementation.

## Expressions and scalar values

### Floating-point operations

| Category | Supported forms |
| --- | --- |
| Arithmetic | `+`, `-`, `*`, `/`, unary negation, supported integer powers lowered into the expression tree |
| Unary intrinsics | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh` |
| Min/max | `min`, `max`, and recognised numeric min/max FIR forms |
| Comparisons | `<`, `<=`, `>`, `>=`, `==`, `/=` using ordered floating-point comparisons |
| Logical combination | Recognised boolean combinations used by conditional expressions and stores |
| Selection | `merge` and supported `if`/select expression trees |

### Integer operations

Signed `integer(1)`, `integer(2)`, `integer(4)`, and `integer(8)` expressions
support:

- addition, subtraction, multiplication, and signed division;
- `abs`;
- signed `min` and `max`;
- signed relational comparisons and equality/inequality; and
- selection over supported integer expressions.

### Type conversion

All array element types must be supported, and all output arrays in one kernel
must currently have the same element type. Read arrays of different supported
types can participate where the expression preserves their types—for example,
an integer material or region array used in a condition alongside real output
arrays.

General type-changing `fir.convert` operations in arithmetic expressions are
rejected. A deliberate exception recognises the integer-to-real conversion of
an affine induction-variable expression, for example:

```fortran
vertexx(j) = xmin + dx * float(j - x_min)
```

This distinction avoids silently dropping a real precision conversion while
still supporting common mesh-initialisation code.

### Read-only scalar capture

A scalar that is read but not written by the kernel is loaded on the host and
passed by value, similar to `firstprivate`:

```fortran
c(i) = alpha * a(i) + beta * b(i)
```

Real or integer expression captures must be compatible with the expression
that consumes them. Integer values used only to build affine subscripts are
classified separately as index captures.

### Iteration-private temporaries

TileOffload has no source-level `private` clause. A mutable scalar reference is
promoted to device SSA and treated as private to one logical iteration only
when the compiler proves that it:

- is defined within the logical iteration before every use;
- is not read before being written;
- is not loop-carried;
- is not conditionally left undefined; and
- is not observed by host code after the launch.

```fortran
!$tileoff parallel tile(128)
do i = lower, upper
  tmp = alpha * a(i)
  c(i) = tmp + b(i)
end do
```

Unsafe mutable references fail compilation with a diagnostic such as
`mutable scalar reference is neither iteration-private nor a reduction`.

## Reductions

### One-dimensional reductions

TileOffload recognises sum, dot product, product, minimum, and maximum patterns:

```fortran
sum = 0.0
!$tileoff parallel tile(256) reduction(+:sum)
do i = lower, upper
  sum = sum + a(i)
end do

dot = 0.0
!$tileoff parallel tile(256) reduction(+:dot)
do i = lower, upper
  dot = dot + a(i) * b(i)
end do

product = 1.0
!$tileoff parallel tile(256) reduction(*:product)
do i = lower, upper
  product = product * a(i)
end do

smallest = huge(smallest)
!$tileoff parallel tile(256) reduction(min:smallest)
do i = lower, upper
  smallest = min(smallest, a(i))
end do
```

The original host scalar participates in the result. An empty range leaves it
unchanged. Choose an initial value suitable for the operator and element type.

### Fused two-dimensional reductions

A two-dimensional traversal may produce several reductions in one launch when
all results have the same type and use one common operator:

```fortran
!$tileoff parallel tile(16,16) &
!$tileoff& reduction(+:volume_sum, +:mass_sum, +:energy_sum)
do k = y_min, y_max
  do j = x_min, x_max
    volume_sum = volume_sum + volume(j,k)
    mass_sum = mass_sum + volume(j,k) * density(j,k)
    energy_sum = energy_sum + volume(j,k) * density(j,k) * energy(j,k)
  end do
end do
```

Fixed neighborhood expansions may be used inside a recognised two-dimensional
reduction when their bounds can be proven and expanded at compile time.

### Hierarchical implementation

The primary kernel writes one partial result per logical device program. A
synthetic `reduction_stage1d` kernel then reduces those partials recursively
until one result remains. Fused reductions use a corresponding variadic result
layout.

Partial and scratch buffers are cached as grow-only workspaces per accelerator
context. Repeated reductions reuse those allocations.

```sh
TILEOFF_REDUCTION_STATS=1 ./program
```

prints allocation, growth, reuse, capacity, primary-launch, and stage-launch
counters at process exit. These workspace counters cover the original partial
and scratch buffers, not every auxiliary result allocation. `TILEOFF_DEBUG=1` prints individual stages.

The revised multi-result finalization path enqueues the final reduction stages
and preserves each result in a packed device buffer. The device-stage path uses
one final explicit wait and one host transfer for the combined results rather
than waiting and transferring separately for every result. Original host seeds
still participate in each reduction. Host-returned scalar reductions remain
synchronous even when asynchronous resident array launches are enabled.

## Types, ranks, and storage

| Feature | Supported today |
| --- | --- |
| Device expression elements | `real(4)`, `real(8)`, signed `integer(1\|2\|4\|8)` |
| Elementwise/stencil kernel rank | 1 or 2 |
| Matrix multiplication rank | 2 |
| Reduction rank | 1, plus recognised fused rank-2 reductions |
| Descriptor data operations | Rank 1 through 3 |
| Storage | Contiguous explicit-shape, assumed-shape, pointer/heap-backed, and allocatable arrays/components |
| Runtime extent and index ABI | Signed 32-bit values |

Data-operation lowering has two principal paths:

1. FIR descriptors are lowered to an ABI containing the data pointer, element
   size, rank, extents, and byte strides.
2. Sized scalars and explicit-shape references are lowered to a raw pointer
   and byte count.

The runtime validates descriptor contiguity. Noncontiguous and strided
sections, unsupported ranks, or objects whose size cannot be established are
rejected rather than copied with an invented extent.

Release a cached allocatable before deallocating or reallocating it. The cache
is keyed by the host data address; retaining an entry across host reallocation
would otherwise leave stale identity and size information.

## Compilation and linking

### Driver pipeline

For a TileOffload source, `tileoffload-flang` performs:

1. a syntax-only Flang invocation to generate module files;
2. FIR emission;
3. the `tileoff-pipeline`, producing host FIR, device IR, and JSON;
4. per-kernel device-IR splitting;
5. per-backend lowering (CUDA Tile IR to bytecode/cubin, or TTIR to TritonGPU);
6. TritonGPU-to-LLVM-MLIR lowering;
7. LLVM-MLIR-to-LLVM-IR translation;
8. target-specific device-library linking and image generation:
   - CUDA links libdevice when required and emits PTX with NVPTX `llc`;
   - HIP links OCML/OCKL and ROCm control bitcode, emits an AMDGPU object,
     and links HSACO with `ld.lld`;
9. device-image and JSON embedding; and
10. host-object generation followed by a relocatable link.

The result of `-c` is a conventional relocatable object containing host code,
the embedded device bundle, metadata, and its registration constructor. No
sidecar PTX, cubin, HSACO, or JSON files are required at run time.

### Data-only TileOffload sources

A source containing TileOffload data or synchronization directives but no
`parallel` kernel is valid. The driver runs the frontend and host runtime
lowering, detects that the generated kernel list is empty, skips device code
generation and embedding, and emits a host-only TileOffload object.

`TILEOFF_ALLOW_EMPTY_KERNELS=1` is not required for this case. It is only an
escape hatch when a source contains a TileOffload `parallel` launch but the pipeline
unexpectedly emits no kernel; the default is to diagnose that inconsistency.

### Ordinary Fortran sources

Sources without a recognised TileOffload sentinel are delegated to the configured
Flang driver. `-E`, `-S`, and `-fsyntax-only` are also delegated and do not run
accelerator code generation. Use `--tileoff-force` or `--tileoff-disable` to
override source auto-detection.

### Separate compilation and multiple bundles

Multiple TileOffload sources, objects, and archives may contribute embedded bundles
to one executable. The driver assigns a stable per-source bundle key so kernel
IDs remain disjoint across separately compiled inputs, while direct
`fir-opt` tests retain predictable sequential IDs.

The runtime validates bundle registration and diagnoses kernel identity/name
collisions rather than silently choosing one definition.

TileOffload host objects also use normal Flang external procedure ABI names when
calling procedures defined in ordinary Flang objects. Top-level TileOffload
definitions expose the corresponding trailing-underscore compatibility entry
point, allowing mixed TileOffload/plain object links.

### Runtime linking

When it detects TileOffload code, the driver adds the runtime matching
`--tileoff-target`:

```text
cuda: -L$LLVM_BUILD/lib -lFortranTileOffloadRuntime    -lcuda     -lstdc++
hip:  -L$LLVM_BUILD/lib -lFortranTileOffloadRuntimeHIP -lamdhip64 -lstdc++
```

with appropriate runtime search paths. Object and archive inputs are scanned
for TileOffload symbols. Use `--tileoff-runtime` when TileOffload code is visible only
through `-lNAME` and cannot be detected from a named input file. The link
target must match the target used to compile every embedded bundle.

## Compiler-driver reference

### Traditional options

The wrapper accepts normal compile/link options including `-c`, `-o`, `-I`,
`-J`, `-L`, `-l`, `-D`, `-U`, `-O`, `-g`, `-f...`, `-m...`, `-Wl,...`, and `--`.
Flags are routed to the relevant frontend, host-codegen, or final-link stage.

### TileOffload options

| Option | Meaning |
| --- | --- |
| `--tileoff-force` | Run TileOffload lowering for every Fortran source. |
| `--tileoff-disable` | Delegate every source to Flang. |
| `--tileoff-runtime` | Force TileOffload runtime libraries into the final link. |
| `--tileoff-no-runtime` | Do not add the runtime automatically. |
| `--tileoff-launch-abi N` | Host launch ABI: `3` by default, or explicit `2` compatibility. |
| `--tileoff-target NAME` | Accelerator platform: `cuda` (default) or `hip`. The spellings `nvidia`, `amd`, and `rocm` are normalized. |
| `--tileoff-gpu-arch ARCH` | Target architecture such as `sm_90a`, `gfx90a`, or `gfx942`. |
| `--tileoff-sm N` | CUDA compatibility option accepting `80`, `sm_80`, `cc80`, or `sm_90a`. |
| `--tileoff-amd-arch ARCH` | Compatibility spelling that selects HIP and an AMD `gfx...` architecture. |
| `--tileoff-backend NAME` | Preferred backend; default `auto`. |
| `--tileoff-fallback-backend NAME` | Backend used when the preferred backend rejects a kernel; default `triton`. |
| `--tileoff-backend-fallback` | Enable fallback; currently the default. |
| `--tileoff-no-backend-fallback` | Fail instead of using the fallback backend. |
| `--tileoff-num-warps N` | Requested warps per CTA; a power of two and at most 32. |
| `--tileoff-threads-per-warp N` | Subgroup width. CUDA requires `32`; HIP accepts `32` or `64`. |
| `--tileoff-num-stages N` | Triton pipeline stages, currently 1 through 16. |
| `--tileoff-f64-matmul-strategy NAME` | `reduce`, `fma`, or `dot`. |
| `--tileoff-cuda-lib-dir DIR` | CUDA Driver API library directory. |
| `--tileoff-cuda-libdevice FILE` | CUDA `libdevice.10.bc`; normally auto-detected. |
| `--tileoff-rocm-path DIR` | ROCm installation prefix; default `ROCM_PATH` or `/opt/rocm`. |
| `--tileoff-rocm-device-lib-dir DIR` | Directory containing ROCm device bitcode such as `ocml.bc` and `ockl.bc`. |
| `--tileoff-hip-lib-dir DIR` | Directory containing `libamdhip64`. |
| `--tileoff-workdir DIR` | Parent directory for a unique intermediate tree. |
| `--tileoff-keep` | Keep intermediate files. |
| `--tileoff-verbose` | Print commands before executing them. |
| `--tileoff-dry-run` | Print commands without executing them. |
| `--tileoff-stop-after STAGE` | Stop one TileOffload source after an internal stage. |

Compatibility aliases without the `tileoff-` prefix are accepted for schedule,
work-directory, verbosity, and stop controls.

Useful stop stages include `modgen`, `fir`, `tileoff-pipeline`, `ttgir`,
`llvm-mlir`, `llvm-ir`, `ptx`, `hsaco`, `device-image`, `embed`, `host-ll`,
`host-obj`, `object`, `objects`, and `link`.

```sh
tileoffload-flang --tileoff-verbose --tileoff-keep \
  --tileoff-stop-after tileoff-pipeline -c kernels.f90
```

### Driver environment variables

| Variable | Purpose |
| --- | --- |
| `LLVM_BUILD` | TileOffload-enabled LLVM/Flang build directory. |
| `TRITON_OPT` | `triton-opt` executable; also needed for CUDA Tile fallback kernels. |
| `TILEOFF_CUDA_TILE_TRANSLATE` | CUDA Tile translator; default `cuda-tile-translate` on PATH. |
| `TILEOFF_TILEIRAS` | Tile IR assembler; default `tileiras` on PATH. |
| `MLIR_TRANSLATE` | Matching `mlir-translate`. |
| `LLC` | Matching `llc` used to emit PTX or an AMDGPU object. |
| `LLVM_LINK` | Matching `llvm-link`; defaults to `$LLVM_BUILD/bin/llvm-link`. |
| `OPT` | Matching `opt`; defaults to `$LLVM_BUILD/bin/opt`. |
| `LD_LLD` | `ld.lld` used to link HSACO; defaults to `$ROCM_PATH/llvm/bin/ld.lld`. |
| `FLANG`, `FIROPT`, `TCO`, `CLANG` | Optional tool overrides. |
| `FLANG_INTRINSIC_MODULES_PATH` | Override Flang's intrinsic-module directory. |
| `CUDA_LIB_DIR` | CUDA Driver API library directory. |
| `TILEOFF_CUDA_LIBDEVICE` | CUDA `libdevice.10.bc` override. The driver auto-detects it when generated IR references `__nv_*`. |
| `ROCM_PATH` | ROCm installation prefix; default `/opt/rocm`. |
| `TILEOFF_ROCM_DEVICE_LIB_DIR` | ROCm bitcode directory containing OCML/OCKL and control bitcode. |
| `TILEOFF_HIP_LIB_DIR` | Directory containing `libamdhip64`; defaults to `$ROCM_PATH/lib` or `lib64`. |
| `TILEOFF_LAUNCH_ABI` | Default host launch ABI; `3`. An explicit `--tileoff-launch-abi` overrides it. |
| `TILEOFF_TARGET` | Default accelerator target: `cuda` or `hip`. |
| `TILEOFF_GPU_ARCH` | Default target architecture such as `sm_90a` or `gfx942`. |
| `TILEOFF_SM` | CUDA architecture compatibility variable. |
| `TILEOFF_AMD_GPU_ARCH` | AMD architecture compatibility variable; default `gfx90a` when HIP is selected and no architecture is supplied. |
| `TILEOFF_BACKEND` | Preferred backend; default `auto`. |
| `TILEOFF_FALLBACK_BACKEND` | Fallback backend; default `triton`. |
| `TILEOFF_ALLOW_BACKEND_FALLBACK` | Boolean backend-fallback control. |
| `TILEOFF_NUM_WARPS` | Requested warp count; current default `1`. |
| `TILEOFF_THREADS_PER_WARP` | Subgroup width override; CUDA defaults to 32, HIP architecture defaults are described above. |
| `TILEOFF_NUM_STAGES` | Pipeline stages; current default `3`. |
| `TILEOFF_F64_MATMUL_STRATEGY` | Default f64 matmul strategy. |
| `TILEOFF_MATMUL_PIPELINE` | Build-time CUDA matmul policy: `0`, `1`, or current default `auto`. |
| `TILEOFF_MATMUL_F64_PIPELINE_STAGES` | FP64 dot auto-policy stage count; default `3`. |
| `TILEOFF_MATMUL_TF32_PIPELINE_STAGES` | TF32 auto-policy stage count; default `3`. |
| `TILEOFF_MATMUL_ALIGNED_LOADS` | Opt-in guarded TF32 alignment specialisation at build time. |
| `TILEOFF_MATMUL_PACK_A` | Experimental patch: build-time `auto`/`1` emits automatic A-packing candidates; unset/`0` does not. |
| `TILEOFF_WORKDIR` | Intermediate-directory parent. |
| `TILEOFF_TTIR_TO_TTGIR_PASSES` | Advanced TTIR-to-TTGIR pass-pipeline override. |
| `TILEOFF_TTGIR_TO_LLVM_PASSES` | Advanced TTGIR-to-LLVM-MLIR pass-pipeline override. |
| `TILEOFF_HIP_TTIR_TO_TTGIR_PASSES` | HIP-only TTIR-to-TritonGPU pass-pipeline override. |
| `TILEOFF_HIP_TTGIR_TO_LLVM_PASSES` | HIP-only TritonGPU-to-LLVM-MLIR pass-pipeline override. |
| `TILEOFF_ALLOW_EMPTY_KERNELS` | Permit an empty kernel list despite a TileOffload `parallel` launch; default false. Data-only sources do not need it. |

The work directory must be writable and contain no whitespace because some
toolchain components and generated command lines require whitespace-free
intermediate paths.

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `TILEOFF_DEVICE` | Device ordinal for either runtime. Takes precedence over the vendor-specific variable; default `0`. |
| `TILEOFF_CUDA_DEVICE` | CUDA device ordinal when `TILEOFF_DEVICE` is unset. |
| `TILEOFF_HIP_DEVICE` | HIP device ordinal when `TILEOFF_DEVICE` is unset. |
| `TILEOFF_USE_CURRENT_CONTEXT` | Use the caller's current CUDA or HIP context instead of retaining a primary context. |
| `TILEOFF_ASYNC_RESIDENT` | Set to `1` to enqueue eligible cached-array launches without waiting after each launch; unset/default retains synchronous completion. |
| `TILEOFF_DEBUG` | Print initialization, bundle, cache, data-region, launch, grid, tile, ABI, and reduction diagnostics. |
| `TILEOFF_REDUCTION_STATS` | Print aggregate reduction-workspace counters at exit. |
| `TILEOFF_MATMUL_PACK_A` | Experimental patch runtime policy: `0` original, `auto`/unset threshold-based, `1` force eligible shapes; requires candidate-enabled binary. |
| `TILEOFF_MATMUL_SHARED_BYTES` | Advanced f32 matmul dynamic-shared-memory override; cannot be below the computed safe minimum. |
| `TILEOFF_MATMUL_F64_SHARED_BYTES` | Advanced f64 matmul dynamic-shared-memory override; cannot be below the computed safe minimum. |
| `TILEOFF_PTX_DIR`, `TILEOFF_PTX`, `TILEOFF_KERNELS_JSON` | Legacy CUDA external-bundle debugging fallbacks. Driver-built objects normally use embedded images and JSON. |

The automatic-packing patch uses compiled shared-memory metadata for its new
helper/packed kernels rather than the legacy matmul shared-memory overrides.

### Asynchronous resident execution

```sh
TILEOFF_ASYNC_RESIDENT=1 TILEOFF_DEVICE=0 ./program
```

This is independent of the host launch ABI: selecting v3 does not enable async
execution. Only eligible array launches whose allocations remain cached may
return before device completion. Temporary/host-output paths and reductions
returning host scalars retain synchronization. Host transfers and allocation
lifetime operations order against queued work on the TileOffload stream.

Use explicit waits around a timed sequence:

```fortran
! Warm up before measuring.
call compute()
!$tileoff wait

t0 = wall_time()
do r = 1, reps
  call compute()
end do
!$tileoff wait
t1 = wall_time()

! Fetch results outside the timed interval if measuring resident compute.
!$tileoff update host(c)
```

Without the second wait, the timer may measure submission rather than completed
GPU work. Without the first, warm-up work may leak into the timed interval.
An eventual successful validation does not prove that the timed interval
included device execution. State whether transfers are included in any reported
performance measurement.

### Accelerator context behavior

The CUDA build uses the CUDA Driver API; the HIP build uses an internal adapter
over the HIP runtime/module APIs. By default each initializes its platform,
selects `TILEOFF_DEVICE` or the vendor-specific device variable, retains the
device's primary context, and restores the caller's previous current context on
return.

With `TILEOFF_USE_CURRENT_CONTEXT=1`, the caller must make a context for the
selected platform current before entering TileOffload. Runtime state is created for
that exact context and the runtime does not retain or release it. Using the
option with no current context is a fatal error.

Modules, function handles, device allocations, data-region frames, streams,
events, reduction workspaces and experimental matmul packing scratch are stored per context and are never reused in
another context or accelerator runtime.

### Thread safety

The revised runtime uses thread-local active-context/operation state, a short
registry lock, per-context mutexes, and a shared lifetime guard. Operations on
one context remain serialized; operations on separate initialized contexts can
proceed independently. Cold initialization and cleanup require broader locking.
Callers must still coordinate shared data-region lifetimes and their own host
accesses. This does not introduce multiple user queues within a context.

Debug, async-resident, and reduction-stat flags are sampled once per outer
runtime operation. Immutable kernel metadata and resolved descriptors are reused
to reduce repeated parsing and lookup; live argument values, layouts, and
ownership remain validated. Device/context selection is not permanently cached
across calls.

## Runtime device selection and MPI

The public `TileOffload` Fortran module provides zero-based device selection:

```fortran
use TileOffload, only: tileoff_get_num_devices, tileoff_set_device_num, &
                       tileoff_get_device_num
use, intrinsic :: iso_c_binding, only: c_int32_t
integer(c_int32_t) :: ngpus, device

ngpus = tileoff_get_num_devices()
if (ngpus <= 0) error stop "No visible accelerator devices"
device = 0_c_int32_t
call tileoff_set_device_num(device)
device = tileoff_get_device_num()
```

These are the current renamed interfaces; they are not `tileoff_*` symbols.
They bind to the matching C runtime entry points. The module source is
`flang/lib/Runtime/TileOffload/TileOffload.f90`. If your installation does not
provide `tileoffload.mod`, compile it once with the same host Flang and add its
module directory to your application build:

```sh
mkdir -p tileoff-modules
"$LLVM_BUILD/bin/flang" -c \
  /path/to/llvm-project/flang/lib/Runtime/TileOffload/TileOffload.f90 \
  -Jtileoff-modules -o tileoff-modules/TileOffload.o
```

Use `-Itileoff-modules` and link the module object with the appropriate runtime.
For a program containing only these API calls, `--tileoff-runtime` ensures the
wrapper links the runtime even if no directive triggers automatic detection.

For MPI, select by **node-local rank**, before entering data or launching work:

```fortran
! Declarations, in the application's specification part:
use mpi_f08
use TileOffload
use, intrinsic :: iso_c_binding, only: c_int32_t
integer :: ierr, local_rank
integer(c_int32_t) :: ngpus
type(MPI_Comm) :: local_comm

! After MPI_Init:
call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                         MPI_INFO_NULL, local_comm, ierr)
call MPI_Comm_rank(local_comm, local_rank, ierr)
ngpus = tileoff_get_num_devices()
if (ngpus <= 0) call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
call tileoff_set_device_num(int(mod(local_rank, int(ngpus)), c_int32_t))
call MPI_Comm_free(local_comm, ierr)
```

The modulo policy intentionally shares devices if local ranks exceed visible
GPUs; require a one-rank-per-GPU allocation if sharing is unwanted. Device
ordinals refer to the process's visible device set. If a scheduler exposes one
GPU per rank, each rank normally selects ordinal zero.

Selection is host-thread-local and takes precedence over environment defaults
for that thread. Changing it does not migrate allocations, data-region frames
or pending work, and the setter is not a wait. Establish the device before
`enter data`. Before switching away from an active phase, wait and complete
that phase's ownership/host-transfer operations on its original device. A
thread with explicit device selection uses the selected device's context,
rather than relying on `TILEOFF_USE_CURRENT_CONTEXT`.

This API supplies process-local selection; it does not implement MPI halo
exchange, GPU-aware MPI buffers or inter-device copies.

## Compiler and runtime architecture

### Principal implementation areas

| Area | Principal files |
| --- | --- |
| Parse tree and syntax | `flang/include/flang/Parser/parse-tree-tileoffload.h`, `flang/lib/Parser/tileoffload-parsers.cpp` |
| Unparsing and semantics | `flang/lib/Parser/unparse.cpp`, `flang/lib/Semantics/resolve-names.cpp` |
| PFT and FIR generation | `flang/include/flang/Lower/PFTBuilder.h`, `flang/lib/Lower/PFTBuilder.cpp`, `flang/lib/Lower/Bridge.cpp` |
| TileOffload dialect | `flang/include/flang/Optimizer/Dialect/TileOffload/TileOffloadOps.td`, `TileOffloadDialect.td`, `flang/lib/Optimizer/Dialect/TileOffload/TileOffloadDialect.cpp` |
| Recognition and planning | `TileOffloadKernelAnalysis.h/.cpp`, `TileOffloadKernelPlan.h` |
| Triton backend | `flang/lib/Optimizer/Dialect/TileOffload/TileOffloadLowerToTriton.cpp` |
| Direct CUDA Tile emitter | `flang/lib/Optimizer/Dialect/TileOffload/TileOffloadCudaTileEmitter.inc` |
| Automatic packing patch | `TileOffloadMatmulPackEmitter.inc`, `tileoffload_matmul_pack.inc`, and driver shared-memory metadata |
| Device-selection module/API | `flang/lib/Runtime/TileOffload/TileOffload.f90`, `tileoffload_device.h` |
| Host runtime lowering | `flang/lib/Optimizer/Dialect/TileOffload/TileOffloadLowerToRuntime.cpp` |
| External ABI aliases | `flang/lib/Optimizer/Dialect/TileOffload/TileOffloadEmitFortranAliases.cpp` |
| Pass pipeline | `TileOffloadPasses.td`, `TileOffloadPipelines.cpp` |
| CUDA/HIP runtimes | `flang/lib/Runtime/TileOffload/tileoffload_runtime.cpp`, built as `FortranTileOffloadRuntime` or `FortranTileOffloadRuntimeHIP` |
| Compiler wrapper | `TileOffload/bin/tileoffload-flang` |

The FIR dialect includes:

- `TileOffload.launch` with tile sizes, pack targets, reduction metadata, and
  `no_copyback` behavior;
- `TileOffload.terminator` for its single-block region;
- `TileOffload.data_region_enter` and `TileOffload.data_region_exit`;
- `TileOffload.copyin`, `TileOffload.create`, `TileOffload.copyout`, and `TileOffload.delete`;
- `TileOffload.present`, `TileOffload.update_host`, and `TileOffload.update_device`;
- `TileOffload.release`, `TileOffload.release_all`, and `TileOffload.wait`.

Hand-written MLIR tests must terminate `TileOffload.launch` with
`TileOffload.terminator`; `fir.end` is not the tileoffload region terminator.

### Pass pipeline

The registered `tileoff-pipeline` performs:

1. `TileOffload-assign-kernel-ids` for stable per-bundle IDs and symbols;
2. recognition, planning, backend selection, device IR, and JSON emission;
3. `TileOffload-lower-to-runtime` for host launch and data calls; and
4. optional `TileOffload-emit-fortran-aliases` for external Fortran ABI
   compatibility.

`TileOffload-outline-kernels` also exists for development experiments but is not a
normal stage of the driver route.

### Recognition and consumed operations

Recognition builds an `ElementwiseKernel` containing loop and extent sources,
array arguments and accesses, scalar/index captures, output expressions,
reduction metadata, and a list of consumed FIR operations.

Validation requires every operation with observable effects in the launch
region to be represented by the kernel or accepted as structural
loop/terminator machinery. When adding a pattern, update consumed-operation
accounting with the recogniser. Otherwise a valid pattern may be rejected—or
an effect could be erased without being represented in device code.

### Host launch ABI v3 (default)

The compiler constructs a host request containing launch dimensions, array
records, scalar/index captures, and reduction-result records, then emits:

```text
__tileoff_launch_v3(request_pointer)
```

This replaces the generated sequence of begin/bind/commit calls. Scalar values
and reduction seeds are captured for the invocation. The runtime consumes the
request during the call; async mode does not make stack request storage into a
device-owned object. Cached device allocations follow the normal lifetime rules.
The v3 entry reuses the checked launch machinery under one outer operation and
context guard; it does not change the numerical device kernel.

```sh
# Default v3; an environment override can change this.
tileoffload-flang -O3 -c kernel.f90

# Explicit selection overrides TILEOFF_LAUNCH_ABI.
tileoffload-flang --tileoff-launch-abi 3 -O3 -c kernel.f90
tileoffload-flang --tileoff-launch-abi 2 -O3 -c kernel.f90
```

Both `tileoff-pipeline` and standalone `TileOffload-lower-to-runtime` default to v3:

```sh
fir-opt --tileoff-pipeline="launch-abi=2 ttir-output=k.ttir json-output=k.json" input.fir
fir-opt --TileOffload-lower-to-runtime="launch-abi=2" input.fir
```

V3 lowering requires an explicit supported 64-bit `x86_64` or `aarch64` host
`llvm.target_triple`. Missing triples, x32/ILP32 forms, and other host targets
are rejected by the current implementation. Use v2 for unsupported host targets;
hand-written MLIR intended to test v3 must declare its actual supported target.

### V2 compatibility and device metadata

V2 remains available through `__tileoff_begin_launch_v2`, typed
`__tileoff_bind_*_v2` calls, and `__tileoff_commit_launch_v2`. Keep these runtime
symbols and their compatibility tests.

**The host launch selection and JSON device launch ABI are different contracts.**
The v3 integration retains device JSON `launch_abi_version = 2`, device signatures,
and backend-private argument accounting. Do not rename JSON fields, change their
version to 3, or change runtime device-ABI checks just because v3 is the host
default. No new device precision mode is implied by v3.

Maintain the default consistently in the driver, `TileOffloadPipelines.cpp`, and
`TileOffloadPasses.td`. Rebuild generated pass declarations through the normal build;
do not edit generated files. The driver logs the selected host ABI and includes
it in its toolchain fingerprint.

## Backend artifact contract

Kernel recognition and scheduling are backend-neutral. `TileOffloadKernelPlan`
contains stable identity, recognised expressions/accesses, tile and subgroup
schedule, public ABI, pack bindings, copyback policy, and optional synthetic
reduction-stage plans.

`TileOffloadCodegenBackend` provides:

- a backend name;
- emitted device-IR and runtime image kinds;
- per-plan support queries and rejection reasons;
- module framing and kernel emission; and
- the count of backend-private pointer arguments.

`triton` provides the broadest coverage and can emit for CUDA or HIP.
`cuda-tile` provides a narrower CUDA-only path. Backend selection supports
`auto`, a named preferred backend, optional per-kernel fallback, and detailed
rejection diagnostics. A compilation may emit both backends for CUDA, but
mixing CUDA and HIP targets in one executable remains unsupported.

### JSON metadata

Schema version 1 and backend-contract version 1 include top-level backend,
accelerator-target, and image fields plus a list of kernel descriptors. Each
descriptor records:

- stable bundle-qualified `id` and `name`;
- backend, accelerator target, device-IR kind, and device-image kind;
- image index and file;
- kernel kind and rank;
- logical tile, warp/stage schedule, subgroup width, and threads per CTA;
- device launch ABI version (still `2` with the v3 host launcher);
- array, scalar, output, and reduction-result counts;
- parameter roles, slots, names, types, array dimensions, and layout fields;
- loop lower-bound and extent roles, constant steps and runtime-step scalar indices;
- pack bindings and `copy_back_writes`;
- backend-private pointer count; and
- reduction operator and synthetic-stage identity where applicable; and
- in the automatic-packing patch, pack/packed-kernel IDs, transposed-A
  orientation and compiled shared-memory requirements for the new kernels.

Legacy PTX and Triton-private field aliases remain while older tools are being
retired.

### Runtime images

The typed registration ABI accepts PTX, cubin, and HSACO images. The normal
Triton CUDA path emits PTX; the Triton HIP path emits an AMDGPU object and links
it into HSACO. The CUDA runtime rejects HIP/HSACO metadata, and the HIP runtime
rejects CUDA/PTX or cubin metadata, preventing an image from being loaded by
the wrong platform runtime.

The implemented CUDA Tile backend reuses planning, public ABI, metadata,
embedding and runtime dispatch, while emitting CUDA Tile IR and cubin. Other
backends can reuse these layers too, provided they produce an image supported
by the selected runtime.

Adding a compiler backend enum alone is insufficient. Driver dispatch,
manifest/embed logic, runtime loading, contract validation, and tests must be
updated together.

## Performance tuning and profiling

### Preserve residency across application phases

For an iterative application, establish one intended data lifetime across setup
and the timestep loop. Fetch only the fields needed by host consumers and release
storage after its final use. Closing a region at the end of initialization and
opening another at the start of hydro can introduce a large copyout/re-upload.
One MPI process removes inter-process communication, but physical-boundary halo
updates and any application-level tile exchanges may still run.

### Tune individual kernels

Tile shape and warp count are separate parameters. The first Fortran dimension
is contiguous, so wider tiles in that dimension are useful candidates. The
emitted TTGIR layout, array strides, masks, and final device code determine the
actual memory behavior. More warps are not automatically better.

### Inspect generated kernels

Keep intermediates for the exact source/configuration being measured:

```sh
mkdir -p tileoffload-inspect
tileoffload-flang --tileoff-target cuda --tileoff-gpu-arch sm_90a \
  --tileoff-num-warps 8 --tileoff-threads-per-warp 32 \
  --tileoff-keep --tileoff-workdir "$PWD/tileoffload-inspect" \
  -O3 -c reset_field_kernel.f90 -o reset_field_kernel.o

rg --files tileoffload-inspect | rg '\.ptx$'
rg -n --glob '*.ptx' '\.entry' tileoffload-inspect
rg -n --glob '*.json' '"tile"|"num_warps"|"threads_per_cta"' tileoffload-inspect
```

Retain normal application include/module flags and relink before measuring.
The normal driver log does not list every kernel name. Generated `.kernels.json`
and per-kernel filenames map source bundles to names; `.entry` identifies PTX
entry points. Do not assume an old numeric kernel name identifies a new build.

Inspect `.ttgir.mlir` for `sizePerThread`, `threadsPerWarp`, and `warpsPerCTA`;
inspect PTX for address calculation, predicated memory operations, and arithmetic.
Scalar 64-bit loads/stores are not inherently uncoalesced. PTX register names
are virtual registers and cannot establish final physical register usage,
spilling, or achieved occupancy.

### Nsight Systems summaries

For CUDA tracing, use the application's normal single-process invocation and
working directory, with a unique report name:

```sh
TILEOFF_ASYNC_RESIDENT=1 nsys profile \
  --trace=cuda,nvtx --sample=none \
  -o cloverleaf-tileoffload ./clover_leaf

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  cloverleaf-tileoffload.nsys-rep > cloverleaf-tileoffload-summary.txt
```

NVTX ranges appear only if instrumentation is enabled; CUDA tracing does not
require application NVTX instrumentation. Nsight Systems can also generate these
summaries from its SQLite export. Retain the full trace when investigating
ordering or gaps: aggregate summaries do not locate copies within application
phases or establish overlap.

Compare matching mesh dimensions, step counts, summary frequency, and validation
results. Compare kernel families as well as individual kernels because one
implementation may split work into more launches. Time inside
`cuEventSynchronize`, stream waits, or synchronous copies includes waiting for
GPU work; do not add it to GPU kernel time as independent overhead.

### Nsight Compute and restricted profiling environments

When GPU counter access is available, collect a few invocations of one kernel:

```sh
# Replace ENTRY_NAME with the exact entry from the current generated PTX.
TILEOFF_ASYNC_RESIDENT=1 ncu \
  --kernel-name-base function --kernel-name ENTRY_NAME \
  --launch-skip 10 --launch-count 3 --kill yes \
  --section LaunchStats --section Occupancy \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --export kernel-profile ./clover_leaf

ncu --import kernel-profile.ncu-rep --page details > kernel-profile.txt
ncu --import kernel-profile.ncu-rep --page raw --csv > kernel-profile.csv
```

Skip/count apply to matching kernel launches, not all application launches.
Counter collection may replay each invocation. `--kill yes` stops the diagnostic
run after collection, so run numerical validation separately to completion.

If `ERR_NVGPUCTRPERM` appears, collection is blocked by platform permissions;
root inside a container is not sufficient by itself. Ask the platform
administrator about supported profiling access. If access cannot be granted,
continue with Nsight Systems durations and generated TTGIR/PTX. Do not treat a
failed counter capture as an occupancy or bandwidth measurement, and stop the
application manually if it continues after the profiling error.

## Testing

### Flang regression suite

Run all Flang tests with:

```sh
cmake --build "$LLVM_BUILD" --target check-flang
```

Run a focused test directory with:

```sh
"$LLVM_BUILD/bin/llvm-lit" -sv \
  /path/to/llvm-project/flang/test/Lower/TileOffload
```

The suite covers parser/unparser behavior, semantics, FIR lowering,
runtime-call lowering, TTIR and JSON, arbitrary loop bounds, affine indices,
mixed-rank stencils, conditional indices, fixed stencil expansions,
multi-output expressions, integer and floating-point operations, assumed-shape
and allocatable descriptors, matmul variants, one- and two-dimensional
reductions, multi-warp lowering, nested data regions, derived-component data
designators, `present`, `no_copyback`, data-only sources, external ABI aliases,
CUDA/HIP accelerator-target JSON and image metadata, and negative diagnostics.

Prefer `CHECK-LABEL`, `CHECK-NEXT`, bounded `CHECK`, and `CHECK-DAG` patterns to
fragile `CHECK-SAME` assertions when operations are intentionally printed on
separate lines. Tests should verify semantics and types rather than local SSA
names.

### Host-ABI regression tests

Keep positive v2 begin/bind/commit checks paired with an explicit
`launch-abi=2` in the command that produces their host IR. Include continued
`RUN` lines when editing test commands. V3 emits one launch call per source
launch, not one replacement for each old begin/bind/commit call.

The dedicated v3 test should cover:

- omitted `launch-abi` selecting v3;
- explicit v3 through the pipeline and standalone lowering;
- explicit v2 compatibility;
- absence of generated v2 begin/bind/commit calls in v3 output;
- request fields, scalar kinds/seeds, and multi-result ordering;
- identical device TTIR/JSON across host ABI selections where expected; and
- unsupported host triples and invalid ABI selections.

Keep checks for `TileOffload.launch` IR, data transfers, release, and synchronization
operations unchanged unless their actual semantics change. V2 matmul argument
checks cannot be migrated by merely renaming the callee: v3 passes a request
pointer, and the corresponding request stores must be checked instead.

Run both TileOffload test directories:

```sh
"$LLVM_BUILD/bin/llvm-lit" -sv \
  /path/to/llvm-project/flang/test/TileOffload \
  /path/to/llvm-project/flang/test/Lower/TileOffload
```

Files ending in `.before-v3-default`, `.before-v2-pin`, or `.bak` are editor
backups, not normal `.f90`/`.mlir` lit tests. Review changes and remove backups
before committing. A switch in defaults does not establish that every legacy
test has been migrated; run the suite and inspect remaining failures.

### Bounds, steps and automatic-packing regressions

Validate positive and negative steps, non-unit constants, runtime step changes,
non-one/negative bounds, rectangular shapes, partial tiles, empty iteration
spaces and untouched output elements. A zero runtime step must fail before a
GPU launch. Repeat through host ABIs 2 and 3 and async modes 0 and 1.

For the automatic-packing patch, run its CPU checks and GPU script before
benchmarking. GPU cases should include changing A between calls, scratch growth,
alignment fallbacks, and observing the pack kernel in debug output. CPU mocks
and successful patch application are not substitutes for Triton parsing,
device compilation or GPU numerical tests.

### Reduction validation executable

```sh
cmake -S tests/reduction -B build/reduction \
  -DTILEOFF_REDUCTION_TEST_DRIVER="$PWD/bin/tileoffload-flang" \
  -DLLVM_BUILD="$LLVM_BUILD" \
  -DTRITON_OPT="$TRITON_OPT" \
  -DMLIR_TRANSLATE="$MLIR_TRANSLATE" \
  -DLLC="$LLC"

cmake --build build/reduction
ctest --test-dir build/reduction -V
```

Configure with `-DTILEOFF_NUM_WARPS=4` to exercise multi-warp reductions.

### CUDA and HIP smoke tests

After building the selected runtime, compile and run a small numerical kernel
on each available target before testing a full application:

```sh
# NVIDIA
tileoffload-flang --tileoff-target cuda --tileoff-gpu-arch sm_90a \
  vector_add.f90 -O3 -o vector_add.cuda
TILEOFF_DEBUG=1 TILEOFF_DEVICE=0 ./vector_add.cuda

# AMD
tileoffload-flang --tileoff-target hip --tileoff-gpu-arch gfx942 \
  vector_add.f90 -O3 -o vector_add.hip
TILEOFF_DEBUG=1 TILEOFF_DEVICE=0 ./vector_add.hip
```

Use an architecture that exactly matches the installed GPU. The HIP path is
particularly sensitive to compatible Triton, LLVM, ROCm device-library, and
`ld.lld` revisions.


### Runtime debugging

```sh
tileoffload-flang --tileoff-keep --tileoff-verbose \
  --tileoff-target TARGET --tileoff-gpu-arch ARCH -c kernel.f90
TILEOFF_DEBUG=1 TILEOFF_DEVICE=0 ./program
```

For CUDA memory checking:

```sh
CUDA_LAUNCH_BLOCKING=1 compute-sanitizer \
  --tool memcheck --error-exitcode 99 ./program
```

Validate results separately from timed launches, and keep required host
updates outside the timed region.

## Extending TileOffload

### Add or change directive syntax

1. Add parse-tree nodes in `parse-tree-tileoffload.h`.
2. Add parsers in `tileoffload-parsers.cpp` and executable-construct routing when
   needed.
3. Resolve contained variables and names in `resolve-names.cpp`.
4. Add unparse support in `unparse.cpp`.
5. Add PFT/Bridge lowering to an existing or new TileOffload FIR operation.
6. Define and verify the operation in `TileOffloadOps.td` and `TileOffloadDialect.cpp`.
7. Lower it to the runtime or consume it in kernel planning.
8. Add parser, unparser, semantics, FIR, runtime-lowering, and negative tests.

Use `Fortran::lower::SomeExpr` for semantic expressions obtained from
`Fortran::semantics::GetExpr`, and call Bridge expression-address helpers using
their current signature rather than an obsolete location-first overload.

### Add an expression operation

1. Extend `ElementwiseExprKind`.
2. Recognise the exact FIR operation or Flang lowering idiom in
   `TileOffloadKernelAnalysis.cpp`.
3. Preserve result-kind and element-type semantics, especially conversions.
4. Mark all represented operations as consumed.
5. Emit the expression in every applicable backend.
6. Add rank-one/rank-two, type, edge-case, and negative tests.

Flang may lower one intrinsic differently for different kinds or optimization
levels. Integer `abs`, min/max, `merge`, and real conversion are examples where
matching one apparent FIR spelling is too fragile.

### Add a kernel pattern

1. Define a kernel kind and recognition result.
2. Prove bounds, extents, access ranks/layouts, types, side-effect rules, and
   scalar classification.
3. Construct the backend-neutral schedule and public ABI.
4. Add runtime binding when the host request/binding interfaces cannot represent
   the pattern.
5. Implement backend emission and `querySupport` checks.
6. Emit and validate JSON metadata.
7. Add end-to-end numerical tests and FIR/TTIR/JSON regression tests.

Fail closed. A precise compile-time diagnostic is preferable to a kernel whose
indexing, mutation, or ABI has not been proven safe.

### Add a device backend

1. Implement `TileOffloadCodegenBackend` against `TileOffloadKernelPlan`.
2. Give unsupported plans precise `querySupport` diagnostics.
3. Register preferred, automatic, and fallback selection.
4. Emit schema-v1/backend-contract-v1 metadata.
5. Add driver dispatch for the emitted device-IR and image kinds.
6. Extend typed bundle generation if a new runtime image type is required.
7. Extend runtime validation and module loading.
8. Test preferred, automatic, fallback, disabled-fallback, unsupported, and
   mixed-backend cases.

Do not place backend-private parameters in the stable public ABI. Record them
through the private-argument contract.

### Add an accelerator target

An accelerator target is distinct from a code-generation backend. Adding one
requires coordinated changes to:

1. the compiler target option and JSON `accelerator_target` contract;
2. the backend's target-specific image kind and lowering configuration;
3. driver architecture parsing, device-library linking, image generation, and
   final runtime selection;
4. typed bundle image-kind registration;
5. module, allocation, copy, stream/event, and launch operations in the target
   runtime; and
6. target/image mismatch diagnostics and end-to-end device tests.

Do not allow a runtime to accept metadata or images for a different target.

### Change a runtime ABI

Update these together:

- runtime-call creation in `TileOffloadLowerToRuntime.cpp`;
- exported runtime function signatures;
- JSON parameter roles and types when the device contract also changes;
- driver image/ABI validation;
- compatibility aliases or a schema/ABI version; and
- MLIR lowering and executable integration tests.

Accelerator-owned objects must remain in state keyed by the active CUDA or HIP
context; never infer ownership from a process-global device-pointer cache.

## Diagnostics and troubleshooting

### `TileOffload cannot plan launch`

The launch is outside the recognised subset. Read the final recogniser reason
first. Common causes include:

- a zero step, an unsupported runtime-step type, or a step that is not launch-invariant;
- bounds or accesses that cannot be recovered safely;
- an unsupported call, conversion, or intrinsic;
- an indirect or unprovable array subscript;
- a loop-carried dependence;
- an unsupported mutable scalar;
- a descriptor or rank the ABI cannot represent; or
- an operation with observable effects that was not consumed by the plan.

### `TileOffload backend selection failed`

The preferred backend is unregistered or its `querySupport` rejected the plan.
Use `--tileoff-backend triton`, permit a Triton fallback, or inspect the detailed
rejection. `--tileoff-no-backend-fallback` is useful in tests that must prove a
specific backend handled every kernel.

### Kernel IDs collide after the rename

The current driver passes `TILEOFF_KERNEL_BUNDLE_KEY` to `fir-opt`, and
`TileOffloadAssignKernelIds.cpp` reads that exact name. A mismatch such as
`TILEOFFLOAD_KERNEL_BUNDLE_KEY` can leave separately compiled sources using the
same sequential IDs. Fix the driver/compiler agreement and rebuild **all**
contributing objects and archives. Do not disable the runtime collision check.

### CUDA Tile translator or architecture errors

Confirm both `cuda-tile-translate` and `tileiras` resolve, or set the documented
absolute-path overrides. The assembler's local `--help` output is authoritative
for its accepted `--gpu-name` spellings. An unsupported architecture requires a
compatible toolchain or Triton backend, not a blind suffix removal.

Syntax errors in `.cuda-tile.mlir` generally indicate emitter/translator version
mismatch. Keep the generated IR, exact tool versions and failing invocation.
A fallback warning followed by `selected backend: mixed` is expected when a
source contains kernels outside the direct backend's subset.

### Driver fails at `tileoff-pipeline`

Re-run with `--tileoff-verbose --tileoff-keep`, execute the printed `fir-opt`
command directly, and inspect the retained `.fir`, `.kernels.ttir`,
`.kernels.json`, and `.host.fir` files.

### PTX reports an unresolved `__nv_*` function

CUDA kernels that use operations such as double-precision square root may
reference CUDA libdevice. The driver detects these references, links
`libdevice.10.bc` with `llvm-link`, runs `opt -O3` so required definitions are
materialized/inlined, and only then invokes NVPTX `llc`.

If `ptxas` still reports an unresolved symbol such as `__nv_sqrt`, ensure
`LLVM_LINK`, `OPT`, and `LLC` come from compatible LLVM builds and that the
selected libdevice is compatible with them. Override discovery with
`--tileoff-cuda-libdevice FILE` or `TILEOFF_CUDA_LIBDEVICE`, retain intermediates,
and run `ptxas` on the generated PTX directly.

### HIP/ROCm toolchain errors

For `--tileoff-target hip`, confirm that:

- `--tileoff-gpu-arch` names the installed GPU, for example `gfx90a` or `gfx942`;
- `ROCM_PATH` points to the intended ROCm installation;
- `LLC`, `MLIR_TRANSLATE`, and `LD_LLD` are compatible with the Triton build;
- the device-library directory contains `ocml.bc`, `ockl.bc`, the required
  `oclc_*` control modules, and an ISA module for the selected `gfx...`; and
- the final link also uses `--tileoff-target hip` and can find `libamdhip64`.

Override device-library discovery with `--tileoff-rocm-device-lib-dir` and HIP
runtime discovery with `--tileoff-hip-lib-dir`. If Triton's AMD pass spellings
differ from the defaults, set `TILEOFF_HIP_TTIR_TO_TTGIR_PASSES` and/or
`TILEOFF_HIP_TTGIR_TO_LLVM_PASSES`.

An error that the AMDGPU object or HSACO was not generated usually indicates a
target-architecture or LLVM/ROCm version mismatch. Retain intermediates and run
the printed `llc` and `ld.lld` commands directly.

### Runtime rejects the accelerator target or image kind

CUDA objects carry `accelerator_target=cuda` with PTX/cubin images; HIP objects
carry `accelerator_target=hip` with HSACO images. A target/image mismatch means
the object was linked against the wrong TileOffload runtime or bundles for different
targets were mixed. Recompile consistently and repeat the same
`--tileoff-target` on the final link.

### `no kernels were emitted`

Data-only TileOffload sources are accepted automatically. If the message says that
a TileOffload `parallel` launch was present, recognition or pipeline output is
inconsistent. Inspect retained FIR/JSON rather than setting
`TILEOFF_ALLOW_EMPTY_KERNELS=1` in normal builds; the variable suppresses a
safety check and does not make a missing kernel execute.

### Undefined `_QP...` references when linking plain objects

Rebuild the TileOffload compiler and recompile the affected TileOffload object with the
current external-alias pass enabled. TileOffload declarations should call ordinary
top-level Flang procedures through `name_`, while TileOffload definitions provide a
compatible external entry point. Use `nm` to confirm that callers and
definitions agree.

### `present` or `update host` reports no allocation

The object was never entered into persistent storage, was released by its
owning data region, belongs to another accelerator context, or was used only
through a host-temporary launch. Establish storage with `enter data`, `create`,
`update device`, or `pack(...:device)` before asserting presence or updating
the host.

### `data_delete requires an active ENTER DATA region`

A cached allocation is not proof of an active ownership frame: `update device`
can create a cache entry even if the intended `enter data` never executed.
Do not use `exit data copyout(...)` solely to fetch results if a later procedure
will perform `exit data delete(...)`; the first exit already ends the frame.
Use `update host(...)` for the intermediate fetch and one final region exit.

An earlier frontend bug omitted standalone TileOffload directives from the PFT lexical
successor chain. In particular, `enter data create(...)` after an allocation
error check containing `STOP` could be skipped. The fix classifies
`TileOffloadStandaloneConstruct` as an executable directive in `PFTBuilder.h`.
Rebuild the frontend and affected Fortran objects; moving the directive to a
different procedure is a workaround, not the intended requirement. Verify the
runtime enter/create calls in lowered FIR when diagnosing an old build.

This specific fix does not establish that every unstructured control-flow corner
case is supported; keep the separate termination-shape diagnostics below.

### Data-region ownership errors

`copyout` and `delete` on `exit data` refer to the innermost active frame. Make
sure that frame acquired every listed object. Do not use `release` or
`release all` to bypass a live frame; exit nested regions in last-in,
first-out order.

### Descriptor sizing or contiguity errors

Prefer explicit-shape arrays or supported contiguous descriptors. A `create`
operation must determine the full byte size. Noncontiguous sections and
unsupported ranks cannot be mapped safely by the current runtime ABI.

### FIR verification after control-flow termination

Some PFT/control-flow shapes can still expose a standalone TileOffload directive
after a block already terminated by an infinite `do`/conditional `exit`
sequence, producing an error such as:

```text
operation with block successors must terminate its parent block
```

Until that frontend control-flow case is fixed, put the corresponding
`exit data` cleanup on the actual termination path immediately before the
`exit`, or restructure the loop so the directive is reached through an
ordinary fall-through block.

### Wrong results with persistent placement

Check the lifetime in this order:

1. Was each input initialized with `copyin` or `update device`?
2. Did host code modify it after the last host-to-device transfer?
3. Did `no_copyback` intentionally keep the result on the device?
4. Are in-place read/write arguments bound to the same cached object?
5. Was a host update or final-owner `copyout` performed before validation?
6. Did an inner data region defer copyout because an outer owner remained?
7. Was the allocation released only after its final consumer?

Enable `TILEOFF_DEBUG=1` and inspect context, bundle, cache, region depth,
ownership count, pack targets, byte counts, extents, lower bounds, strides,
grid, tile, subgroup/block size, accelerator target, and image metadata.

## Known limitations

- TileOffload is experimental and deliberately recogniser-based rather than a
  general-purpose Fortran device compiler.
- Supported loop bounds need not start at one. Constant/runtime steps must be
  nonzero and representable in the supported signed 32-bit step ABI; runtime
  steps must be launch-invariant. Backend-specific restrictions still apply.
- Kernel computation supports logical ranks one and two. Data-descriptor
  operations support ranks one through three.
- Device extents, lower bounds, strides, and index captures use signed 32-bit
  runtime values.
- General type-changing arithmetic conversions are unsupported; the affine
  integer-index-to-real case is handled explicitly.
- All output arrays in one kernel must currently have the same element type.
- Fused multi-result reductions require one common operator and result type.
- Matrix multiplication supports only f32 and f64.
- Triton has the broadest implemented coverage. Direct CUDA Tile currently
  handles a restricted pointwise/rank-1 reduction subset and emits cubin;
  matmul, general stencils and fused multi-result reductions require Triton.
- Automatic matmul packing is currently an opt-in TF32/CUDA/Triton patch,
  with full compiler and GPU validation outstanding. It does not cache packed
  contents across launches or support automatic B/C packing.
- CUDA and HIP use separate runtime libraries. Mixing CUDA and HIP bundles in
  one executable or loading an image through the wrong runtime is unsupported.
- Mixed CUDA Tile/Triton kernels are supported for CUDA. Mixed accelerator
  targets in one executable are unsupported.
- The runtime owns one stream per CUDA or HIP context and serializes operations
  within each context; initialization/cleanup also use shared registry/lifetime
  coordination.
- Host ABI v3 currently requires an explicit supported 64-bit x86_64 or aarch64
  target triple. V2 remains the explicit compatibility path.
- The HIP path depends on revision-compatible Triton, LLVM, ROCm device
  libraries, and `ld.lld`; AMD lowering pass names may require the documented
  environment overrides for a particular Triton revision.
- There is no source-level `private` clause; only proven iteration-private
  scalar temporaries are promoted automatically.
- There are no asynchronous queue IDs, user stream clauses, exposed events,
  or cross-stream dependency clauses.
- Noncontiguous sections, arbitrary indirect gathers, complex values,
  character values, derived-type device elements, unsigned arithmetic,
  arbitrary function calls, recursion, and general unstructured control flow
  are outside the current kernel subset.
- Derived-component data objects are supported, but arbitrary array-section
  mapping is not.
- Floating-point min/max, comparisons, transcendental functions, contraction,
  and reduction order inherit backend behavior. Validate NaNs, infinities,
  signed zero, and reproducibility when an application depends on them.
- A frontend control-flow corner case remains for standalone data directives
  reached after certain terminated `do`/`exit` block shapes.

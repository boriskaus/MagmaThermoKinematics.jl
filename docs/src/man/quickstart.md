# Quick Start

This package requires selecting a backend and dimensionality before loading backend-specific diffusion/fields modules.

Important distinction:

- `environment!(...)` initializes backend modules inside MagmaThermoKinematics.
- If your own script/module uses ParallelStencil macros directly (for example `@zeros`, `@parallel`, `@parallel_indices`), initialize ParallelStencil in your script scope as well with `@init_parallel_stencil(...)`.

## 2D CPU Workflow

:::code-group

```julia [CPUs]
using ParallelStencil
using MagmaThermoKinematics

environment!(:cpu, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2)   # needed when this script uses @zeros/@parallel

using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D
```

```julia [Nvidia GPUs]
using CUDA
using ParallelStencil
using MagmaThermoKinematics

environment!(:gpu, Float64, 2)
CUDA.device!(0)
@init_parallel_stencil(CUDA, Float64, 2)      # needed when this script uses @zeros/@parallel

using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D
```

```julia [Apple GPUs (Metal)]
using Metal
using ParallelStencil
using MagmaThermoKinematics

environment!(:metal, Float32, 2)
@init_parallel_stencil(Metal, Float32, 2)     # needed when this script uses @zeros/@parallel

using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D
```
:::

## 3D GPU Workflow

:::code-group

```julia [CPUs]
using ParallelStencil
using MagmaThermoKinematics

environment!(:cpu, Float64, 3)
@init_parallel_stencil(Threads, Float64, 3)   # needed when this script uses @zeros/@parallel

using MagmaThermoKinematics.Diffusion3D
using MagmaThermoKinematics.Fields3D
```

```julia [Nvidia GPUs]
using CUDA
using ParallelStencil
using MagmaThermoKinematics

environment!(:gpu, Float64, 3)
CUDA.device!(0)
@init_parallel_stencil(CUDA, Float64, 3)      # needed when this script uses @zeros/@parallel

using MagmaThermoKinematics.Diffusion3D
using MagmaThermoKinematics.Fields3D
```

```julia [Apple GPUs (Metal)]
using Metal
using ParallelStencil
using MagmaThermoKinematics

environment!(:metal, Float32, 3)
@init_parallel_stencil(Metal, Float32, 3)     # needed when this script uses @zeros/@parallel

using MagmaThermoKinematics.Diffusion3D
using MagmaThermoKinematics.Fields3D
```
:::

## Minimal Model Setup Pattern

```julia
Grid = CreateGrid(size=(500, 500), extent=(30e3, 30e3))
Num  = Numeric_params(verbose=false)
```

Then:

1. Build arrays and phases.
2. Initialize tracers and initial temperature fields.
3. Inject dikes/sills when required.
4. Advance the diffusion/advection steps.
5. Save visualization output and diagnostics.

See [Examples](examples.md) for complete scripts.

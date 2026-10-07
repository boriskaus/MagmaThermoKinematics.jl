# Quick Start

MagmaThermoKinematics runs on a [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) backend. The same code is used for 2D and 3D models: the dimensionality follows from the size of the grid and arrays you create.

## Backend Selection

:::code-group

```julia [CPUs]
using MagmaThermoKinematics

backend = CPU()
```

```julia [Nvidia GPUs]
using CUDA
using MagmaThermoKinematics

backend = CUDABackend()
```
:::

Pass the backend when allocating arrays, either through `CreateArrays(...; backend)` or through `NumParam(backend=backend)` when using the `MTK_GMG` workflow.

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

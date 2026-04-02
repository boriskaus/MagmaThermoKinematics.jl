# Quick Start

This package requires selecting a backend and dimensionality before loading backend-specific diffusion/fields modules.

## 2D CPU Workflow

:::code-group

```julia [CPUs]
using ParallelStencil
using MagmaThermoKinematics

environment!(:cpu, Float64, 2)

using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D
```

```julia [Nvidia GPUs]
using CUDA
using ParallelStencil
using MagmaThermoKinematics

environment!(:gpu, Float64, 2)

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

using MagmaThermoKinematics.Diffusion3D
using MagmaThermoKinematics.Fields3D
```

```julia [Nvidia GPUs]
using CUDA
using ParallelStencil
using MagmaThermoKinematics

environment!(:gpu, Float64, 3)

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

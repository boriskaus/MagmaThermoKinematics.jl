# Dependencies

MagmaThermoKinematics.jl relies on a set of Julia packages for numerics, physics, and data handling.

## Core Dependencies

The package uses for example:

- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for the energy solver kernels, which run on CPUs and GPUs.
- [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) for material properties (for example nonlinear conductivity and melting behavior).
- [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl) for tracer storage.
- [Parameters.jl](https://github.com/mauro3/Parameters.jl) for parameter handling.
- [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) for grid-tracer interpolation.
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for efficient small-array operations.

These dependencies are installed automatically when adding MagmaThermoKinematics.jl.

## Visualization and Output

Examples commonly use:

- Makie.jl for plotting, through CairoMakie or GLMakie (not installed with MagmaThermoKinematics; add it to your own environment).
- WriteVTK.jl to generate VTK/PVD outputs for ParaView.

## Real-World Model Setup

For geometry and setup generation from geophysical models, workflows can use [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl).jl. The repository includes examples showing integration with MTK.

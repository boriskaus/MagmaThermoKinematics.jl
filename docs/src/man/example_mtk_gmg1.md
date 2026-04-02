# MTK_GMG Example 1

This page documents the coding style used in [examples/MTK_GMG_2D_example1.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/MTK_GMG_2D_example1.jl).

## Imports and Backend

```julia
const USE_GPU=false;
if USE_GPU
    using CUDA
end
using ParallelStencil, ParallelStencil.FiniteDifferences2D

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 2)
    CUDA.device!(0)
    @init_parallel_stencil(CUDA, Float64, 2)
else
    environment!(:cpu, Float64, 2)
    @init_parallel_stencil(Threads, Float64, 2)
end
using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D
using MagmaThermoKinematics.MTK_GMG_2D
using MagmaThermoKinematics.GeophysicalModelGenerator
using GeoParams, Random
using Plots
using MagmaThermoKinematics.MTK_GMG
```

## Overriding MTK_GMG Hooks

```julia
function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
    return nothing
end

function MTK_GMG.MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)
    Arrays.T_init   .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm
    @views  Arrays.Phases[Arrays.Z .> -5000] .= 0
    Arrays.Phases_init .= Arrays.Phases
    return nothing
end
```

## Parameter Setup and Run

```julia
Num = NumParam(Nx=135*2, Nz=135*2, SimName="Test1", maxTime_Myrs=0.005, USE_GPU=USE_GPU)
Dike_params = DikeParam(Type="ElasticDike", InjectionInterval_year=1000, W_in=5e3, H_in=250, DikePhase=2)

MatParam = (
    SetMaterialParams(Name="Host rock", Phase=1,
        Density=ConstantDensity(ρ=2700kg/m^3),
        LatentHeat=ConstantLatentHeat(Q_L=2.55e5J/kg),
        Conductivity=T_Conductivity_Whittington(),
        HeatCapacity=T_HeatCapacity_Whittington(),
        Melting=MeltingParam_Assimilation()),
)

Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Dike_params)
```

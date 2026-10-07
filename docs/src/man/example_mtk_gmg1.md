# MTK_GMG Example 1

 with different magma temperatures to study the thermal effects on crustal rocks.

## Imports and Backend
All simulations need to import the appropriate libraries:
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
function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
    return nothing
end

function MTK_GMG.MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters)
    Arrays.T_init   .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm
    @views  Arrays.Phases[Arrays.Z .> -5000] .= 0
    Arrays.Phases_init .= Arrays.Phases
    return nothing
end
```

## Parameter Setup and Run
`NumParam` sets the numerical parameters; defaults are set, such that you only need to specify the non-default part.
```julia
Num = NumParam(Nx=135*2, Nz=135*2, SimName="Test1", maxTime_Myrs=0.005, USE_GPU=USE_GPU)
sill = PennyShapedSill(Center=Point2(0.0, -7.0e3)m, W=2.5e3m, H=250m, E=1.5e10Pa, ν=0.3NoUnits)
Sill_params = SillParams(
    sill                   = sill,
    InjectionInterval_year = 1000,
    nTr_dike               = 300*4,
    SillPhase              = 2,
    T_in_Celsius           = 1000,
)
```

Next, you need to specify the material parameters used in the simulation for each of the phases in the model. The material parameters themselves are taken from `GeoParams`, and can thus be nonlinear.
```julia
MatParam = (
    SetMaterialParams(Name="Host rock", Phase=1,
        Density=ConstantDensity(ρ=2700kg/m^3),
        LatentHeat=ConstantLatentHeat(Q_L=2.55e5J/kg),
        Conductivity=T_Conductivity_Whittington(),
        HeatCapacity=T_HeatCapacity_Whittington(),
        Melting=MeltingParam_Assimilation()),
)
```

Once that is done, you can run a simulation with:
```julia
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Sill_params)
```

## Visualize results
The default simulation create PVD/VTK files: 
```julia
NumParam
  SimName: String "Test1"
  Output_VTK: Bool true
```
This can be visualized with [ParaView](https://www.paraview.org), here you can also create a movie of the results.
For that, simply open `Test1/Test1.pvd` in paraview.


## Zircon ages
Every time a dike is injected, we also inject tracers within the dike that are tracked throughout the rest of the simulation. At the end of the simulation they are returned as a vector with  `Tracers`. 
On every tracer, we store time and temperature information and history. That is useful as it allows us to reconstruct how zircons may have crystallized from the magma and thus track the zircon age data history.

The optional [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl) extension allows computing this, by growing an individual zircon crystal for every tracer of the simulation.  

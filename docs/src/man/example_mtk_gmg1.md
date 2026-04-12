# MTK_GMG Example 1

This page documents the coding style used in [examples/MTK_GMG_2D_example1.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/MTK_GMG_2D_example1.jl). It is a simple 2D example where we 

## Imports and Backend
All simulations need to important the appropriate libraries:
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
`NumParam` sets the numerical parameters; defaults are set, such that you only need to specify the non-default part.
```julia
Num = NumParam(Nx=135*2, Nz=135*2, SimName="Test1", maxTime_Myrs=0.005, USE_GPU=USE_GPU)
Dike_params = DikeParam(Type="ElasticDike", InjectionInterval_year=1000, W_in=5e3, H_in=250, DikePhase=2)
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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Dike_params)
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
For example:
```julia
julia> Tracers[3]
Tracer
  num: Int64 3
  coord: Array{Float64}((2,)) [9346.5998252003, -11344.343444273873]
  T: Float64 693.0370340531517
  Phase: Int64 2
  Phi: Float64 0.0008870128834362513
  time_vec: Array{Float64}((69,)) [0.005066297710863042, 0.00513795680154003, 0.005209615892217019, 0.0052812749828940065, 0.005352934073570995, 0.0054245931642479825, 0.00549625225492497, 0.005567911345601959, 0.005639570436278947, 0.0057112295269559354  …  0.00929418406080534, 0.009365843151482329, 0.009437502242159316, 0.009509161332836306, 0.009580820423513294, 0.009652479514190281, 0.00972413860486727, 0.009795797695544256, 0.009867456786221246, 0.009939115876898233]
  T_vec: Array{Float64}((69,)) [845.1862075756897, 826.1510317482357, 814.9719385554686, 806.948832469245, 800.8739566366835, 796.0342995499269, 792.0595253039968, 788.6801220029093, 785.7476346629572, 783.0895605653898  …  697.3007455990053, 696.873391809416, 696.4427602709188, 696.0087376186582, 695.5714132151782, 695.1310868034119, 694.6882373635913, 694.2434410010095, 693.7972776998067, 693.3502531528633]
```

On every tracer, we store time and temperature information and history. That is useful as it allows us to reconstruct how zircons may have crystallized from the magma and thus track the zircon age data history.

The optional [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl) extension allows computing this, by growing an individual zircon crystal for every tracer of the simulation.  

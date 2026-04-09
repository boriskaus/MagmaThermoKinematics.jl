# MTK_GMG 3D Examples

This page follows the same script-oriented style as the 2D MTK_GMG examples, but for 3D setups.

## Why MTK_GMG Is Flexible for Real-World Systems

These examples show that the same MTK_GMG solver pipeline can be reused across different volcanic systems by changing setup data and runtime choices instead of rewriting core numerics.

- Unzen3D demonstrates a workflow where the model domain, phase structure, and thermal state are assembled from imported topography and project-specific initialization logic.
- Lanin3D demonstrates a workflow where a custom topography is downloaded using GMT if the file doesn't exist, and a custom output hook is defined to print time and iteration diagnostics.
- Both examples use the same core concepts: CartData construction, material tuples, dike parameterization, and hook-based customization for output and diagnostics.

In practice, this means MagmaThermoKinematics.jl can move from synthetic benchmark setups to real-world applications by swapping geological input data and user hooks while keeping the computational framework stable.

## Unzen3D

This section documents the script in [examples/MTK_GMG_3D_example.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/MTK_GMG_3D_example.jl).

![](../assets/movies/Unzen3D.gif)

### Imports and Backend

```julia
const USE_GPU = false
if USE_GPU
	using CUDA
end

using MagmaThermoKinematics
@static if USE_GPU
	environment!(:gpu, Float64, 3)
	CUDA.device!(1)
else
	environment!(:cpu, Float64, 3)
end
using MagmaThermoKinematics.Diffusion3D
using MagmaThermoKinematics.Fields3D
using MagmaThermoKinematics.MTK_GMG
using MagmaThermoKinematics.MTK_GMG_3D
using Random, GeoParams, GeophysicalModelGenerator
```

### GeophysicalModelGenerator Setup

```julia
Topo_cart = load_GMG(joinpath(@__DIR__, "Topo_cart"))
Nx, Ny, Nz = 100, 100, 100
X, Y, Z = xyz_grid(range(-23, 23, length = Nx),
				   range(-19, 19, length = Ny),
				   range(-20, 5, length = Nz))
Data_3D = CartData(X, Y, Z, (Phases = zeros(Int64, size(X)), Temp = zeros(size(X))))

Below = below_surface(Data_3D, Topo_cart)
Data_3D.fields.Phases[Below] .= 1
```

### Parameter Setup and Run

```julia
Num = NumParam(SimName = "Unzen3D", maxTime_Myrs = 0.025, USE_GPU = USE_GPU,
			   AddRandomSills = true, RandomSills_timestep = 5)

# Dike_params and MatParam are defined in the script.
Grid, Arrays, Tracers, Dikes, time_props =
	MTK_GMG_3D.MTK_GeoParams_3D(MatParam, Num, Dike_params, CartData_input = Data_3D)
```

## Lanin3D

This section documents the scripts:

- [examples/MTK_GMG_3D_Lanin.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/MTK_GMG_3D_Lanin.jl)

![](../assets/movies/Lanin3D.gif)

### Custom Setup Builder

```julia
# Create 3D grid of the region
if !isfile(joinpath(@__DIR__,"Topo_cart_Lanin3D.jld2"))
    using GMT, Statistics
    println("Creating topography grid from GMG for Lanin 3D example...")
    Topo       =   import_topo(lon = [-71.9, -71.1], lat=[-39.95, -39.35], file="@earth_relief_01s.grd")
    proj       =   ProjectionPoint(; Lat=mean(Topo.lat.val), Lon=mean(Topo.lon.val))
    Topo_cart  =   convert2CartData(Topo, proj)

    Xt,Yt,Zt   =   xyz_grid(-20:.025:20,-20:.025:20,0)
    Topo_cart  =   project_CartData(CartData(Xt,Yt,Zt,(Zt=Zt,)), Topo, proj)
    write_paraview(Topo_cart,joinpath(@__DIR__,"Topo_cart_Lanin3D"));

    save_GMG(joinpath(@__DIR__,"Topo_cart_Lanin3D"), Topo_cart)
end

```

### Hooks and Run

```julia
import MagmaThermoKinematics.MTK_GMG
import MagmaThermoKinematics.MTK_GMG_3D

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters,
								  Arrays::NamedTuple, Mat_tup::Tuple,
								  Dikes::DikeParameters)
	println("$(Num.it), Time=$(round(Num.time / Num.SecYear / 1e3, digits = 3)) kyrs")
	return nothing
end

Grid, Arrays, Tracers, Dikes, time_props =
	MTK_GMG_3D.MTK_GeoParams_3D(MatParam, Num, Dike_params, CartData_input = Data_3D)
```

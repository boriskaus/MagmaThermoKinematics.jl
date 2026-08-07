<h1> <img src="docs/src/assets/logo.png" alt="MagmaThermoKinematics.jl" width="50"> MagmaThermoKinematics.jl </h1>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://boriskaus.github.io/MagmaThermoKinematics.jl/dev/)
[![Build Status](https://github.com/boriskaus/MagmaThermoKinematics.jl/workflows/CI/badge.svg)](https://github.com/boriskaus/MagmaThermoKinematics.jl/actions)
[![version](https://juliahub.com/docs/General/MagmaThermoKinematics/stable/version.svg)](https://juliahub.com/ui/Packages/General/MagmaThermoKinematics)
[![DOI](https://zenodo.org/badge/337510164.svg)](https://zenodo.org/badge/latestdoi/337510164)
[![codecov](https://codecov.io/gh/boriskaus/MagmaThermoKinematics.jl/graph/badge.svg?token=RHOL3SU2YE)](https://codecov.io/gh/boriskaus/MagmaThermoKinematics.jl)

Understanding how magmatic systems work is of interest to a wide range of Earth Scientists.

This easy to use and versatile package simulates the thermal evolution of magmatic systems following the intrusion of dikes and sills. It can take 2D, 2D axisymmetric and 3D geometries into account, and works on both parallel CPU's and GPU's. A finite difference discretization is employed for the energy equation, combined with semi-Lagrangian advection and tracers to track the thermal evolution of emplaced magma. Dikes are emplaced kinematically and the host rock is shifted to accommodate space for the intruding dikes/sills, in a number of ways including by using analytical models for penny-shaped cracks in elastic media. Cooling, crystallizing and latent heat effects are taken into account, and the thermal evolution of tracers can be used to simulate zircon age distributions.

Magma can also be *removed* from the system: eruptions fire either on a kinematic eruptible-volume criterion or on a physical chamber-overpressure balance following [Degruyter & Huber (2014)](https://doi.org/10.1016/j.epsl.2014.06.047), and optionally deflate the chamber with a volume-conserving displacement field. A kinematic free surface, initialized flat or from real topography, is advected by the same host-rock displacements, so both intrusion and eruption deform the ground surface.

Below we give a number of example scripts that show how it can be used to simulate a number of scenarios. The [documentation](https://boriskaus.github.io/MagmaThermoKinematics.jl/dev/) describes the governing equations and the numerical scheme, the eruption and free-surface models, and the full API.

## Contents
- [MagmaThermoKinematics.jl](#magmathermokinematicsjl)
  - [Contents](#contents)
  - [100-lines 2D example](#100-lines-2d-example)
  - [100-lines 3D example](#100-lines-3d-example)
  - [Eruptions and free surface](#eruptions-and-free-surface)
  - [Zircon ages](#zircon-ages)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Ongoing development](#ongoing-development)
  - [Related work](#related-work)
  - [Benchmarks](#benchmarking)
  - [Citing](#citing-magmathermokinematics.jl)

## 100-lines 2D example
A simple example that simulates the emplacement of dikes within the crust over a period of 10'000 years is shown below.

![2-D dike intrusion](examples/movies/Example2D.gif)

The code to simulate this, including visualization, is <100 lines (if we remove empty ones) and the key parts of it are shown below
```julia
const USE_GPU=false;
if USE_GPU; using CUDA; end      # needs to be loaded before loading Parallkel=
using ParallelStencil, ParallelStencil.FiniteDifferences2D
using MagmaThermoKinematics
using InjectSills
@static if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
    @init_parallel_stencil(CUDA, Float64, 2)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
    @init_parallel_stencil(Threads, Float64, 2)
end
using MagmaThermoKinematics.Diffusion2D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields2D
using Plots

#------------------------------------------------------------------------------------------
@views function MainCode_2D();
Nx,Nz                   =   500,500
Grid                    =   CreateGrid(size=(Nx,Nz), extent=(30e3, 30e3)) # grid points & domain size
Num                     =   Numeric_params(verbose=false)                   # Nonlinear solver options

# Set material parameters
MatParam                =   (
        SetMaterialParams(Name="Rock", Phase=1,
             Density    = ConstantDensity(ρ=2800kg/m^3),
           HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
           Conductivity = ConstantConductivity(k=1.5Watt/K/m),
             LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
                Melting = MeltingParam_Caricchi()),
                            )

GeoT                    =   20.0/1e3;                   # Geothermal gradient [K/km]
W_in, H_in              =   5e3,    0.2e3;              # Width and thickness of dike
T_in                    =   900;                        # Intrusion temperature
InjectionInterval       =   0.1kyr;                     # Inject a new dike every X kyrs
maxTime                 =   25kyr;                      # Maximum simulation time in kyrs
H_ran, W_ran            =   Grid.L.*[0.3; 0.4];         # Size of domain in which we randomly place dikes and range of angles
κ                       =   1.2/(2800*1050);            # thermal diffusivity
dt                      =   minimum(Grid.Δ.^2)/κ/10;    # stable timestep (required for explicit FD)
nt                      =   floor(Int64,maxTime/dt);    # number of required timesteps
nTr_dike                =   300;                        # number of tracers inserted per dike

# Array initializations
Arrays = CreateArrays(Dict( (Nx,  Nz)=>(T=0,T_K=0, T_it_old=0, Tupdate=0, Tbuffer=0, K=1.5, Rho=2800, Cp=1050, Tnew=0,  Hr=0, Hl=0, Kc=1, P=0, X=0, Z=0, ϕₒ=0, ϕ=0, dϕdT=0),
                                (Nx-1,Nz)=>(qx=0,Kx=0), (Nx, Nz-1)=>(qz=0,Kz=0 ) ))
# CPU buffers
Tnew_cpu                =   Matrix{Float64}(undef, Grid.N...)
Phi_melt_cpu            =   similar(Tnew_cpu)
if USE_GPU;
    Phases      =   CUDA.ones(Int64,Grid.N...)
else
    Phases      =   ones(Int64,Grid.N...)
end

@parallel (1:Nx, 1:Nz) GridArray!(Arrays.X,  Arrays.Z, Grid.coord1D[1], Grid.coord1D[2])
Tracers                 =   StructArray{Tracer{Float32}}(undef, 1)                   # Initialize tracers
Arrays.T               .=   -Arrays.Z.*GeoT;                                        # Initial (linear) temperature profile

# Preparation of visualisation
ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])

time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
for it = 1:nt   # Time loop

    if floor(time/InjectionInterval)> dike_inj       # Add new dike every X years
        dike_inj  =     floor(time/InjectionInterval)                                               # Keeps track on what was injected already
        cen       =     (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];         # Randomly vary center of dike
        if cen[end]<-12e3;
            Angle_rand = rand( 80.0:0.1:100.0)                                      # Orientation: near-vertical @ depth
        else
            Angle_rand = rand(-10.0:0.1:10.0);
        end                                  # Orientation: near-vertical @ shallower depth
        sill      =     EllipticalIntrusion(Center=Point2(cen[1],cen[2])*m, Angle=Vec1(Angle_rand)*NoUnits, W=W_in*m, H=H_in*m)
        Tnew_cpu .=     Array(Arrays.T)
        Tracers, Tnew_cpu, Vol, _, _   =   inject_sills(Tracers, Tnew_cpu, Grid.coord1D, sill, T_in, 2, nTr_dike);   # Add dike, move hostrocks
        Arrays.T .=     Data.Array(Tnew_cpu)
        InjectVol +=    Vol                                                                 # Keep track of injected volume
        println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(time),digits=2)) m³/s")
    end

    Nonlinear_Diffusion_step_2D!(Arrays, MatParam, Phases, Grid, dt, Num)   # Perform a nonlinear diffusion step

    copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, Arrays.Tnew, Arrays.ϕ)     # Copy arrays to CPU to update properties
    UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers

    @parallel assign!(Arrays.T, Arrays.Tnew)
    @parallel assign!(Arrays.Tnew, Arrays.T)                                # Update temperature
    time                =   time + dt;                                      # Keep track of evolved time
    Melt_Time[it]       =   sum(Arrays.ϕ)/prod(Grid.N)                      # Melt fraction in crust
    Time_vec[it]        =   time;                                           # Vector with time
    println(" Timestep $it = $(round(time/kyr*100)/100) kyrs")

    if mod(it,20)==0  # Visualisation
        x,z         =   Grid.coord1D[1], Grid.coord1D[2]
        p1          =   heatmap(x/1e3, z/1e3, Array(Arrays.T)',  aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:lajolla, clims=(0.,900.), xlabel="Width [km]",ylabel="Depth [km]", title="$(round(time/kyr, digits=2)) kyrs", dpi=200, fontsize=6, colorbar_title="Temperature")
        p2          =   heatmap(x/1e3,z/1e3, Array(Arrays.ϕ)',  aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:nuuk,    clims=(0., 1. ), xlabel="Width [km]",             dpi=200, fontsize=6, colorbar_title="Melt Fraction")
        plot(p1, p2, layout=(1,2)); frame(anim)
    end
end
gif(anim, "Example2D.gif", fps = 15)   # create gif animation
return Time_vec, Melt_Time, Tracers, Grid, Arrays;
end # end of main function

Time_vec, Melt_Time, Tracers, Grid, Arrays = MainCode_2D(); # start the main code
plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") # Create plot

```
The main routines are thus ``inject_sills(..)``, which inserts a new dike or sill (of given dimensions and orientation) into the domain using the [InjectSills.jl](https://github.com/JuliaGeodynamics/InjectSills.jl) package, and ``Nonlinear_Diffusion_step_2D!(...)``, which computes thermal diffusion. Variable thermal conductivity, and latent heat are all taken into account.

If you have a multicore processor (chances are very high that you do), the code can also take advantage of that. The only thing that you have to do is start julia with multiple threads, which on linux or macOS is done with:
```
$julia --threads=8
```
You can check how many threads you are using by:
```julia
julia> Threads.nthreads()
8
```
You start the simulation with
```julia
julia> include("Example2D.jl")
```
provided that you are in the same directory as the file (check that with `pwd()`).

If you happen to have a machine with an NVIDIA graphics card build in, the code will run (substantially) faster by changing this flag:
```julia
const USE_GPU=true;
```

The full code example can be downloaded [here](./examples/Example2D.jl)

## 100-lines 3D example
To go from 2D to 3D, only a few minor changes to the code above are required. A movie of our example, which was computed on a laptop, is:
![3-D dike intrusion](examples/movies/Example3D.gif)

Here the full 3D code:
```julia
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end
using ParallelStencil, ParallelStencil.FiniteDifferences2D

using MagmaThermoKinematics
using InjectSills
@static if USE_GPU
    environment!(:gpu, Float64, 3)      # initialize parallel stencil in 3D
    @init_parallel_stencil(CUDA, Float64, 3)
else
    environment!(:cpu, Float64, 3)      # initialize parallel stencil in 3D
    @init_parallel_stencil(Threads, Float64, 3)
end
using MagmaThermoKinematics.Diffusion3D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields3D
using Plots
using WriteVTK

#------------------------------------------------------------------------------------------
@views function MainCode_3D();
    Nx,Ny,Nz                =   250,250,250
    Grid                    =   CreateGrid(size=(Nx,Ny,Nz), extent=(30e3, 30e3, 30e3)) # grid points & domain size
    Num                     =   Numeric_params(verbose=false)                   # Nonlinear solver options

    # Set material parameters
    MatParam                =   (
            SetMaterialParams(Name="Rock", Phase=1,
                 Density    = ConstantDensity(ρ=2800kg/m^3),
               HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
                    Melting = MeltingParam_Caricchi()),
                                )

    GeoT                    =   20.0/1e3;                   # Geothermal gradient [K/km]
    W_in, H_in              =   5e3,    0.5e3;              # Width and thickness of dike
    T_in                    =   900;                        # Intrusion temperature
    InjectionInterval       =   0.1kyr;                     # Inject a new dike every X kyrs
    maxTime                 =   15kyr;                      # Maximum simulation time in kyrs
    H_ran, W_ran            =   Grid.L[1]*0.3,Grid.L[3]*0.4;# Size of domain in which we randomly place dikes and range of angles
    κ                       =   1.2/(2800*1050);            # thermal diffusivity
    dt                      =   minimum(Grid.Δ.^2)/κ/10;    # stable timestep (required for explicit FD)
    nt                      =   floor(Int64,maxTime/dt);    # number of required timesteps
    nTr_dike                =   300;                        # number of tracers inserted per dike

    # Array initializations
    Arrays = CreateArrays(Dict( (Nx,  Ny, Nz)=>(T=0,T_K=0, T_it_old=0, Tupdate=0, Tbuffer=0, K=1.5, Rho=2800, Cp=1050, Tnew=0,  Hr=0, Hl=0, Kc=1, P=0, X=0, Y=0, Z=0, ϕₒ=0, ϕ=0, dϕdT=0),
                                (Nx-1,Ny,Nz)=>(qx=0,Kx=0), (Nx, Ny-1, Nz)=>(qy=0,Ky=0 ) , (Nx, Ny, Nz-1)=>(qz=0,Kz=0 ) ))
    # CPU buffers
    Tnew_cpu                =   zeros(Float64, Grid.N...)
    Phi_melt_cpu            =   similar(Tnew_cpu)
    if USE_GPU; Phases      =   CUDA.ones(Int64,Grid.N...)
    else        Phases      =   ones(Int64,Grid.N...)   end

    @parallel (1:Nx,1:Ny,1:Nz) GridArray!(Arrays.X,Arrays.Y,Arrays.Z, Grid.coord1D[1], Grid.coord1D[2], Grid.coord1D[3])
    Tracers                 =   StructArray{Tracer{Float32}}(undef, 1)                   # Initialize tracers
    Arrays.T               .=   -Arrays.Z.*GeoT;                                        # Initial (linear) temperature profile

    # Preparation of VTK/Paraview output
    if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; pvd = paraview_collection("Example3D");

    time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
    for it = 1:nt   # Time loop

        if floor(time/InjectionInterval)> dike_inj       # Add new dike every X years
            dike_inj  =     floor(time/InjectionInterval)                                               # Keeps track on what was injected already
            cen       =     (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, 3).*[W_ran;W_ran;H_ran];   # Randomly vary center of dike
            if cen[end]<-12e3;  Angle_rand = [rand(80.0:0.1:100.0); rand(0:360)]                        # Dikes at depth
            else                Angle_rand = [rand(-10.0:0.1:10.0); rand(0:360)] end                    # Sills at shallower depth
            sill      =     EllipticalIntrusion(Center=Point3(cen[1],cen[2],cen[3])*m, Angle=Vec2(Angle_rand[1],Angle_rand[2])*NoUnits, W=W_in*m, H=H_in*m)
            Tnew_cpu .=     Array(Arrays.T)
            Tracers, Tnew_cpu, Vol, _, _   =   inject_sills(Tracers, Tnew_cpu, Grid.coord1D, sill, T_in, 2, nTr_dike);   # Add dike, move hostrocks
            Arrays.T .=     Data.Array(Tnew_cpu)
            InjectVol +=    Vol                                                                 # Keep track of injected volume
            println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(time),digits=2)) m³/s")
        end

        Nonlinear_Diffusion_step_3D!(Arrays, MatParam, Phases, Grid, dt, Num)   # Perform a nonlinear diffusion step

        copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, Arrays.Tnew, Arrays.ϕ)     # Copy arrays to CPU to update properties
        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers

        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)                                # Update temperature
        time                =   time + dt;                                      # Keep track of evolved time
        Melt_Time[it]       =   sum(Arrays.ϕ)/prod(Grid.N)                      # Melt fraction in crust
        Time_vec[it]        =   time;                                           # Vector with time
        println(" Timestep $it = $(round(time/kyr*100)/100) kyrs")

        if mod(it,20)==0  # Visualisation
            x,y,z         =   Grid.coord1D[1], Grid.coord1D[2], Grid.coord1D[3]
            vtkfile = vtk_grid("./viz3D_out/ex3D_$(Int32(it+1e4))", Vector(x/1e3), Vector(y/1e3), Vector(z/1e3)) # 3-D VTK file
            vtkfile["Temperature"] = Array(Arrays.T); vtkfile["MeltFraction"] = Array(Arrays.ϕ);                 # Store fields in file
            outfiles = vtk_save(vtkfile); pvd[time/kyr] = vtkfile                                   # Save file & update pvd file
        end
    end
    vtk_save(pvd)
    return Time_vec, Melt_Time, Tracers, Grid, Arrays;
end # end of main function

Time_vec, Melt_Time, Tracers, Grid, Arrays = MainCode_3D(); # start the main code
```
The result of the script are a range of VTK files, which can be visualized with the 3D software [Paraview](https://www.paraview.org). The full code example can be downloaded [here](./examples/Example3D.jl), and the paraview statefile (to reproduce the movie) is available [here](./examples/movies/Example3D_Paraview.pvsm).

## Eruptions and free surface
Once a chamber becomes eruptible — melt fraction above a threshold `ϕ_erupt` — melt can be withdrawn from the domain. Two trigger models are available through `EruptionParams`:

- **Kinematic**: an eruption fires once the eruptible volume reaches `V_crit`, removing a fixed fraction `erupt_efficiency` of the eruptible melt.
- **Physical**: the chamber overpressure `ΔP` is integrated every timestep from a recharge / crystallization / wall-relaxation balance following Degruyter & Huber (2014), and the chamber drains when `ΔP` reaches `ΔP_crit`. The erupted volume is then set by the elastic and thermodynamic storage the crossed overpressure represents, rather than by a tuned efficiency.

Either trigger optionally deflates the chamber with a volume-conserving displacement field (column subsidence, or a superposition of local Mogi kernels), and freezes a corresponding fraction of tracers so their T-t paths record the eruption instant.

The top of the domain can be tracked as a kinematic free surface, initialized flat or from a topography array or function, and advected by the same host-rock displacements that emplace sills and deflate the chamber. Injection therefore uplifts and eruption subsides the ground surface. `mass_budget` and `enthalpy` are provided as conservation diagnostics.

Four self-contained example scripts cover both triggers in 2D and 3D, including a run on real downloaded topography:

```
examples/MTK_GMG_2D_Eruption_Kinematic.jl
examples/MTK_GMG_2D_Eruption_DegruyterHuber.jl
examples/MTK_GMG_3D_Eruption_DegruyterHuber_FlatTopo.jl
examples/MTK_GMG_3D_Eruption_DegruyterHuber_Lanin.jl
```

See the [Eruptions](https://boriskaus.github.io/MagmaThermoKinematics.jl/dev/man/eruptions) and [Free Surface](https://boriskaus.github.io/MagmaThermoKinematics.jl/dev/man/free_surface) documentation for the governing equations.

## Zircon ages
The temperature-time path recorded by every tracer can be converted into zircon age distributions in three ways: the "UCLA" method of Oscar Lovera, the "Geneva" method of Gregor Weber and coworkers (both benchmarked in Schmitt et al., 2023, and implemented in `examples/Plot_ZirconAgeStatistics.jl`), and an explicit 1D radially-symmetric crystal-growth model provided by [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl). The last is loaded automatically as a package extension when you do `using ZirconGrowth`; see the [Zircon ages](https://boriskaus.github.io/MagmaThermoKinematics.jl/dev/man/zircon_growth) documentation.

## Dependencies
We rely on [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) for the energy solver, [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) to define material properties (such as nonlinear conductivity, melting, etc.), [InjectSills.jl](https://github.com/JuliaGeodynamics/InjectSills.jl) to kinematically emplace dikes and sills (penny-shaped cracks, elliptical intrusions, cylindrical top-accreting bodies), [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl) to generate an array of tracer structures, [Random.jl](https://docs.julialang.org/en/v1/stdlib/Random/) for random number generation, [Parameters.jl](https://github.com/mauro3/Parameters.jl) to simplify setting parameters, [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) to interpolate properties such as temperature from a fixed grid to tracers, and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for speed. All these dependencies should be installed automatically if you install `MagmaThermoKinematics.jl`.

[Plots.jl](http://docs.juliaplots.org/latest/) is employed for plotting, and [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) is used in the 3D example to generate `*.vtr/*.pvd` files that can be visualized with [Paraview](https://www.paraview.org). You have to add both packages yourself; they are however anyways useful to have.

If you want to apply it to a real-world system, you can use the [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) package to create a setup. We have a few examples to demonstrate his package integrates with `MTK`.

## Installation
After installing julia in the usual manner, you can add (and test) the package with
```
julia>]
  pkg> add MagmaThermoKinematics
  pkg> test MagmaThermoKinematics
```
Dependencies such as `ParallelStencil.jl` are installed automatically.
The testing suite run above performs a large number of tests and, among others, compares the results with analytical solutions for advection/diffusion. Let us know if you encounter problems.

If you want to run the examples and create plots, you may also want to install these packages:
```
julia>]
  pkg> add Plots
  pkg> add Makie
  pkg> add WriteVTK
```
Next, you can download one of the codes above, put it in your current directory, and start it with
```
julia> include("Example2D.jl")
```
And finally, if you have installed this package previously on your system, but want to update it to the latest version:
```
julia>]
  pkg> update MagmaThermoKinematics
```

If you are interested in running benchmark scenarios, comparing `MagmaThermoKinematics.jl`, versus codes previously used by the `Geneva` (Gregor Weber, Luca Caricchi) and `UCLA` (Oscar Lovera) research group, run this:
```
julia> include("examples/Example2D_ZASSy.jl")
```

## Ongoing development

We are working on a more general magmatic systems software as part of the [MAGMA](https://magma.uni-mainz.de) project funded by the European Research Council. That will not only include thermal diffusion solvers and kinematically emplaced dikes (as done here), but also mechanical multiphysics solvers (to compute stress and deformation rate in the system, for example). For that we follow a modular and reusable software approach, where various software componentys are are defined in external package and re-usable packages, will which ultimately make it easier to write new software and apply that to natural cases. An example is the [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) package where material properties (e.g., density, heat capacity, thermal conductivity) are defined, that can be used by other packages (such as MagmaThermoKinematics.jl). The advantage of this approach is that such material properties only have to be defined once, and can subsequently be used in a whole range of software packages.
If you are interested in this, have a look at [https://github.com/JuliaGeodynamics/](https://github.com/JuliaGeodynamics/).

## Benchmarking
MagmaThermoKinematics was benchmarked versus the code of the UCLA group (Oscar Lovera) and the Geneva Group (Gregor Weber). Very good agreement was found between `MTK` and the other codes as described in

- Schmitt, A.K., Sliwinski, J., Caricchi, L., Bachmann, O., Riel, N., Kaus, B.J.P., de Léon, A.C., Cornet, J., Friedrichs, B., Lovera, O., Sheldrake, T., Weber, G., 2023. Zircon age spectra to quantify magma evolution. Geosphere 19. https://doi.org/10.1130/GES02563.1

Note that quite large differences in the thermal structure can occur, even for the same magma flux, depending on the *assumption* you make on where and how magma is intruded (underplating, central injection, injection through dikes). Often, but not always, this difference is larger than the effect of using different material parameters.

## Related work
Thermal-kinematic codes such as the ones presented here have been around for some time with various degrees of sophistication (e.g., [1],[2],[3],[4]). A recent effort in Julia, similar to what we do here, is described in [5].

Yet, as far as we are aware, the source code of these other packages is currently not openly available (at least not in a non-binary format), which makes it often non-straightforward to understand what is actually done under the hood. No existing code works in 3D and can take advantage of GPU's.

[1] Dufek, J., & Bergantz, G. W. (2005). Lower crustal magma genesis and preservation: A stochastic framework for the evaluation of basalt–crust interaction. *Journal of Petrology*, 46(11), 2167–2195. [https://doi.org/10.1093/petrology/egi049](https://doi.org/10.1093/petrology/egi049)

[2] Annen, C., Blundy, J. D., & Sparks, R. S. J. (2006). The genesis of intermediate and silicic magmas in deep crustal hot zones. *Journal of Petrology*. 47(3), 505–539. [https://doi.org/10.1093/petrology/egi084](https://doi.org/10.1093/petrology/egi084)

[3] Caricchi, L., Annen, C., Blundy, J., Simpson, G., & Pinel, V. (2014). Frequency and magnitude of volcanic eruptions controlled by magma injection and buoyancy. *Nature Geoscience*, 7, 126–130. [https://doi.org/10.1038/NGEO2041](https://doi.org/10.1038/NGEO2041)

[4] Tierney, C. R., Schmitt, A. K., Lovera, O. M., & de Silva, S. L. (2016). Voluminous plutonism during a period of volcanic quiescence revealed by thermochemical modeling of zircon. *Geology*, 44, 683–686. [https://doi.org/10.1130/G37968.1](https://doi.org/10.1130/G37968.1)

[5] Melnik, O.E., Utkin, I.S., Bindeman, I.N., 2021. Magma Chamber Formation by Dike Accretion and Crustal Melting: 2D Thermo‐Compositional Model With Emphasis on Eruptions and Implication for Zircon Records. *J Geophys Res Solid Earth* 126. [https://doi.org/10.1029/2021JB023008](https://doi.org/10.1029/2021JB023008). A preprint of their work is available [here](https://www.essoar.org/doi/10.1002/essoar.10505594.1).


## Citing MagmaThermoKinematics.jl

If you use MagmaThermoKinematics, please cite:

1) The exact version of the software that you used to produce results. We automatically create a permanent [Zenodo](https://zenodo.org/badge/latestdoi/337510164) of every release version of the code (see the right bottom part of the Zenodo page for the correct citation). You can list this citation in the publication list

2) Schmitt, A.K., Sliwinski, J., Caricchi, L., Bachmann, O., Riel, N., Kaus, B.J.P., de Léon, A.C., Cornet, J., Friedrichs, B., Lovera, O., Sheldrake, T., Weber, G., 2023. Zircon age spectra to quantify magma evolution. Geosphere 19. https://doi.org/10.1130/GES02563.1

The Schmitt et al. (2023) paper shows benchmarks of how `MagmaThermoKinematics` compares to other software packages. The appendix of this paper also clarifies the equations we solve and shows the impact of various parameters (such as temperature-dependent conductivity) on model results.

> **Warning**
>
> Code users are fully responsible for the results they publish. There are many things that can go wrong with numerical modelling, even if you use a well-benchmarked code (too low resolution, don't check timestep convergence, poor nonlinear convergence, used beyond what the software was developed for, etc. etc. - we have seen many epic failures over the years). If you want us to double-check, you can send us your manuscript (and input file!) before you submit.

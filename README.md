# MagmaThermoKinematics.jl

Understanding how magmatic systems evolve and how the rock record can be interpreted is of interest to a wide range of Earth Scientists.

This easy to use and versatile package simulates the thermal evolution of magmatic systems, consisting of (kinematically) emplaced dikes. It can take 2D, 2D axisymmetric and 3D geometries into account, and works in parallel on both CPU (and GPU's). A finite difference discretization is employed for the energy equation, combined with semi-Lagrangian advection and tracers to track the thermal evolution of emplaced magma. Dikes are emplaced kinematically and the host rock is shifted to accommodate space for the intruding dikes/sills, using analytical models for penny-shaped cracks in elastic media. Cooling, crystallizing and latent heat effects are taken into account, and the thermal evolution of tracers can be used to simulate zircon age distributions.

Below we give a number of example scripts that show how it can be used to simulate a number of scenarios.


## Contents
  - [100-lines 2D example](#100-lines-2d-example)
  - [100-lines 3D example](#100-lines-3d-example)
  - [Dependencies](#dependencies)
  - [Installation](#installation)

## 100-lines 2D example
A simple example that simulates the emplacement of dikes within the crust over a period of 10'000 years is shown below. 

![2-D dike intrusion](examples/movies/Example2D.gif)

The code to simulate this, including visualization, is only 100 lines and the key parts of it are shown below
```julia
# Load required packages
#(...) 

#------------------------------------------------------------------------------------------
@views function MainCode_2D();

# Model parameters
W,H                     =   30km, 30km;             # Width, Height
ρ                       =   2800;                   # Density 
cp                      =   1050;                   # Heat capacity
k_rock, k_magma         =   1.5, 1.2;               # Thermal conductivity of host rock & magma
La                      =   350e3;                  # Latent heat J/kg/K
GeoT                    =   20.0/km;                # Geothermal gradient [K/km]
x_in,z_in               =   20km,   -15km;          # Center of dike [x,z]
W_in, H_in              =   5km,    0.2km;          # Width and thickness of dike
T_in                    =   900;                    # Intrusion temperature
InjectionInterval       =   0.1kyr;                 # Inject a new dike every X kyrs
maxTime                 =   25kyr;                  # Maximum simulation time in kyrs
H_ran, W_ran            =   H*0.4, W*0.3;           # Size of domain in which we randomly place dikes and range of angles   
DikeType                =   "ElasticDike"           # Type to be injected ("ElasticDike","SquareDike")

Nx, Nz                  =   500, 500;               # resolution (grid cells in x,z direction)
dx,dz                   =   W/(Nx-1), H/(Nz-1);     # grid size
κ                       =   k_rock./(ρ*cp);         # thermal diffusivity   
dt                      =   min(dx^2, dz^2)./κ/10;  # stable timestep (required for explicit FD)
nt::Int64               =   floor(maxTime/dt);      # number of required timesteps
nTr_dike                =   300;                    # number of tracers inserted per dike


#(...) 
# Initialize numerical parameters and arrays
#(...) 

# Set up model geometry & initial T structure
# (...)
dike                    =   Dike(W=W_in,H=H_in,Type=DikeType,T=T_in); # "Reference" dike with given thickness,radius and T
T                       .=   -Z.*GeoT;                                # Initial (linear) temperature profile
Phi, dPhi_dt            =   SolidFraction(T, Phi_o, dt);              # Compute solid fractio
# Preparation of visualisation
#(...)

time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
for it = 1:nt   # Time loop

    if floor(time/InjectionInterval)> dike_inj                                      # Add new dike every X years
        dike_inj  =   floor(time/InjectionInterval)                                 # Keeps track on what was injected already
        cen       =   [W/2.; -H/2.] + rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];        # Randomly vary center of dike 
        if cen[end]<-12km;  Angle_rand = rand( 80.0:0.1:100.0)                      # Orientation: near-vertical @ depth             
        else                Angle_rand = rand(-10.0:0.1:10.0); end                  # Orientation: near-vertical @ shallower depth     
        dike      =   Dike(dike, Center=cen[:],Angle=[Angle_rand]);                 # Specify dike with random location/angle but fixed size/T 
        Tracers, T, Vol     =   InjectDike(Tracers, T, Grid, dike, nTr_dike);       # Add dike, move hostrocks
        InjectVol           +=  Vol                                                 # Keep track of injected volume
        println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(time),digits=2)) m³/s")
    end
    
    Phi, dPhi_dt     =  SolidFraction(T, Phi_o, dt);                                # Compute solid fraction
    K               .=  Phi.*k_rock .+ (1 .- Phi).*k_magma;                         # Thermal conductivity

    # Perform a diffusion step
    @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, dt, dx, dz, La, dPhi_dt);  
    @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                      # set lateral boundary conditions (flux-free)
    Tnew[:,1] .= GeoT*H; Tnew[:,end] .= 0.0;                                                    # bottom & top temperature (constant)
    
    Tracers             =   UpdateTracers(Tracers, Grid, Tnew, Phi);                            # Update info on tracers 
    T, Tnew             =   Tnew, T;                                                            # Update temperature
    time                =   time + dt;                                                          # Keep track of evolved time
    Melt_Time[it]       =   sum( 1.0 .- Phi)/(Nx*Nz)                                            # Melt fraction in crust    
    Time_vec[it]        =   time;                                                               # Vector with time
    println(" Timestep $it = $(round(time/kyr*100)/100) kyrs")

    if mod(it,20)==0  # Visualisation
        Phi_melt    =   1.0 .- Phi;     
        p1          =   heatmap(x/km, z/km, T',         aspect_ratio=1, xlims=(x[1]/km,x[end]/km), ylims=(z[1]/km,z[end]/km),   c=:lajolla, clims=(0.,900.), xlabel="Width [km]",ylabel="Depth [km]", title="$(round(time/kyr, digits=2)) kyrs", dpi=200, fontsize=6, colorbar_title="Temperature")
        p2          =   heatmap(x/km, z/km, Phi_melt',  aspect_ratio=1, xlims=(x[1]/km,x[end]/km), ylims=(z[1]/km,z[end]/km),   c=:nuuk,    clims=(0., 1. ), xlabel="Width [km]",             dpi=200, fontsize=6, colorbar_title="Melt Fraction")
        plot(p1, p2, layout=(1,2)); frame(anim)
    end
end
gif(anim, "Example2D.gif", fps = 15)   # create gif animation
return Time_vec, Melt_Time;
end # end of main function

Time_vec,Melt_Time = MainCode_2D(); # start the main code
plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot
```
The main routines are thus ``InjectDike(..)``, which inserts a new dike (of given dimensions and orientation) into the domain, and ``diffusion2D_step!(...)``, which computes thermal diffusion. Variable thermal conductivity, and latent heat are all taken into account. 

The full code example can be downloaded [here](./examples/Example2D.jl)

## 100-lines 3D example
To go from 2D to 3D, only a few minor changes to the code above are required. A movie of our example, which was computed on a laptop, is:
![3-D dike intrusion](examples/movies/Example3D.gif)

The main changes, compared to the 2D example, are:
```julia
#(...)
using MagmaThermoKinematics.Diffusion3D
#(...)
using ParallelStencil.FiniteDifferences3D
using WriteVTK                                   

# Initialize 
@init_parallel_stencil(Threads, Float64, 3);    # initialize parallel stencil in 2D

#------------------------------------------------------------------------------------------
@views function MainCode_3D();

# Model parameters
W,L,H                   =   30km,30km,30km;         # Width, Length, Height
#(...)
x_in,y_in,z_in          =   20km,20km,-15km;        # Center of dike [x,y,z coordinates]
W_in, H_in              =   5km,  0.5km;            # Width and thickness of dike [m]
#(..)
Nx, Ny, Nz              =   250, 250, 250;                    # Resolution
dx,dy,dz                =   W/(Nx-1), L/(Nx-1), H/(Nz-1);     # Grid size [m]
#(...)

# Array initializations
#(...)

# Set up model geometry & initial T structure
x,y,z                   =   (0:Nx-1)*dx, (0:Ny-1)*dy, (-(Nz-1):0)*dz;       # 1D coordinate arrays
crd                     =   collect(Iterators.product(x,y,z))               # Generate coordinates from 1D coordinate vectors   
X,Y,Z                   =   (x->x[1]).(crd),(x->x[2]).(crd),(x->x[3]).(crd);# Transfer coords to 3D arrays
Grid                    =   (x,y,z);                                        # Grid 
#(...)

# Preparation of VTK/Paraview output 
if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; pvd = paraview_collection("Example3D");

#(...)
for it = 1:nt   # Time loop

    if floor(time/InjectionInterval)> dike_inj        # Add new dike every X years
        #(...)       
        cen             =   [W/2.;L/2.;-H/2.] + rand(-0.5:1e-3:0.5, 3).*[W_ran;W_ran;H_ran];    # Randomly vary center of dike 
        if cen[end]<-12km;  Angle_rand = [rand(80.0:0.1:100.0); rand(0:360)]                    # Dikes at depth             
        else                Angle_rand = [rand(-10.0:0.1:10.0); rand(0:360)] end                # Sills at shallower depth
        dike            =   Dike(dike,Center=cen[:],Angle=Angle_rand);                          # Specify dike with random location/angle but fixed size 
        #(...)       
    end

    #(...)       
  
    # Perform a diffusion step
    @parallel diffusion3D_step_varK!(Tnew, T, qx, qy, qz, K, Kx, Ky, Kz, Rho, Cp, dt, dx, dy, dz,  La, dPhi_dt);  
    @parallel (1:size(T,2), 1:size(T,3)) bc3D_x!(Tnew);                                         # set lateral boundary conditions (flux-free)
    @parallel (1:size(T,1), 1:size(T,3)) bc3D_y!(Tnew);                                         # set lateral boundary conditions (flux-free)
    Tnew[:,:,1] .= GeoT*H; Tnew[:,:,end] .= 0.0;                                                # bottom & top temperature (constant)
    
    #(...)

    if mod(it,10)==0  # Visualisation
        Phi_melt        =   1.0 .- Phi;             
        vtkfile = vtk_grid("./viz3D_out/ex3D_$(Int32(it+1e4))", Vector(x/km), Vector(y/km), Vector(z/km)) # 3-D
        vtkfile["Temperature"] = T; vtkfile["MeltFraction"] = Phi_melt;
        outfiles = vtk_save(vtkfile); pvd[time_kyrs] = vtkfile 
    end

end
vtk_save(pvd)


return nothing; #Tracers, T, Grid;
end # end of main function

MainCode_3D(); # start the main code
```
The result of the script are a range of VTK files, which can be visualized with the 3D software [Paraview](https://www.paraview.org). The full code example can be downloaded [here](./examples/Example3D.jl), and the paraview statefile (to reproduce the movie) is available [here](./examples/movies/Example3D_Paraview.pvsm).

## Dependencies
We rely on [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) to for the energy solver, [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl) to generate an aray of tracer structures, [Random.jl](https://docs.julialang.org/en/v1/stdlib/Random/) for random number generation, [Parameters.jl](https://github.com/mauro3/Parameters.jl) to simplify setting parameters (such as specifying dike properties), [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) to interpolate properties such as temperature from a fixed grid to tracers, and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for speed. All these dependencies should be installed automatically if you install `MagmaThermoKinematics.jl`.

[Plots.jl](http://docs.juliaplots.org/latest/) is employed for plotting, and [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) is used in the 3D example to generate `*.vtr/*.pvd` files that can be visualized with [Paraview](https://www.paraview.org). You have to add both packages yourself; they are however anyways useful to have.

## Installation
After installing julia in the usual manner, you can add (and test) the package with 
```
julia>]
  pkg> add https://github.com/omlins/ParallelStencil.jl
  pkg> add https://github.com/boriskaus/MagmaThermoKinematics.jl
  pkg> test MagmaThermoKinematics
```
We use ParallelStencil.jl, which is not (yet) a registed julia package, which is why you have to install that first.
The testing suite run above performs a large number of tests and, among others, compares the results with analytical solutions for advection/diffusion. Let us know if you encounter problems. 

If you want to run the examples and create plots, you may also want to install these two packages:
```
julia>]
  pkg> add Plots
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

## Dependencies


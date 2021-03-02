# MagmaThermoKinematics.jl

Understanding how magmatic systems evolve and how the rock record can be interpreted is of interest to a wide range of Earth Scientists.

This easy to use and versatile package simulates the thermal evolution of magmatic systems, consisting of (kinematically) emplaced dikes. It can take 2D, 2D axisymmetric and 3D geometries into account, and works in parallel on both CPU (and GPU's). A finite difference discretization is employed for the energy equation, combined with semi-lagrangian advection and tracers to track the thermal evolution of emplaced magma. Dikes are emplaced kinematically and the host rock is shifted to accommodate space for the intruding dikes/sills. Cooling, crystallizing and latent heat effects are taken into account, and the thermal evolution of tracers can be used to simulate zircon age distributions.

Below we give a number of example scripts that show how it can be used to simulate a number of scenarios.


## Contents
  - [100-line 2D example](#100-lines-2d-example)
  - [100-line 3D example](#3d-example)
  - [Dependencies](#dependencies)
  - [Installation](#installation)

## 100-line 2D example
A simple example that simulates the emplacement of dikes within the crust over a period of 10'000 years is shown below. 

![2-D dike intrusion](examples/movies/Example2D.gif)

The code to simulate this, including visualization, is only 100 lines and the key parts of it are shown below
```julia
# Load required packages
#(...) 

#------------------------------------------------------------------------------------------
@views function MainCode_2D();

# Model parameters
W,H                     =   30, 30;                 # Width, Height in km
ρ                       =   2800;                   # Density 
cp                      =   1050;                   # Heat capacity
k_rock, k_magma         =   1.5, 1.2;               # Thermal conductivity of host rock & magma
La                      =   350e3;                  # Latent heat J/kg/K
GeoT                    =   20.0;                   # Geothermal gradient [K/km]
x_in,z_in               =   20e3,   -15e3;          # Center of dike [x,z coordinates in m]
W_in, H_in              =   5e3,    2e2;            # Width and thickness of dike [m]
T_in                    =   900;                    # intrusion temperature
InjectionInterval_kyrs  =   0.1;                    # inject a new dike every X kyrs
maxTime_kyrs            =   25;                     # Maximum simulation time in kyrs
H_ran, W_ran            =   H*0.4, W*0.3;           # size of domain in which we randomly place dikes and range of angles   
DikeType                =   "ElasticDike"           # Type to be injected ("SquareDike","ElasticDike")

Nx, Nz                  =   500, 500;                           # resolution
dx                      =   W/(Nx-1)*1e3; dz = H*1e3/(Nz-1);    # grid size [m]
κ                       =   k_rock./(ρ*cp);                     # thermal diffusivity   
dt                      =   min(dx^2, dz^2)./κ/10;              # stable timestep (required for explicit FD)
nt                      =   floor(maxTime_kyrs*1e3*SecYear/dt); # number of required timesteps
nTr_dike                =   300;                                # number of tracers inserted per dike


#(...) 
# Initialize numerical parameters and arrays
#(...) 

# Set up model geometry & initial T structure
# (...)
T                       .=   -Z./1e3.*GeoT;                     # Initial (linear) temperature profile
SolidFraction!(T, Phi_o, Phi, dPhi_dt, dt);                     # Compute solid fraction

# Preparation of visualisation
#(...)

time,time_kyrs, dike_inj, InjectVol = 0.0, 0.0, 0.0, 0.0;
Time_vec,Melt_Time          = zeros(nt,1),zeros(nt,1);
for it = 1:nt   # Time loop

    if floor(time_kyrs/InjectionInterval_kyrs)> dike_inj        # Add new dike every X years
        dike_inj            =   floor(time_kyrs/InjectionInterval_kyrs)                             # Keeps track on what was injected already
        cen                 =   [W/2.; -H/2.]; center = (rand(2,1) .- 0.5).*[W_ran;H_ran] + cen;    # Randomly vary center of dike 
        if center[end]<-12; Angle_rand = rand( 80.0:0.1:100.0)                                      # Dikes at depth             
        else                Angle_rand = rand(-10.0:0.1:10.0); end                                  # Sills at shallower depth
        dike                =   Dike(Width=W_in, Thickness=H_in, Center=center[:]*1e3,Angle=[Angle_rand],Type=DikeType,T=T_in); # Specify dike with random location/angle but fixed size 
        Tracers, T, Vol     =   InjectDike(Tracers, T, Grid, dike, nTr_dike);                       # Add dike, move hostrocks
        InjectVol           +=  Vol                                                                 # Keep track of injected volume
        println("Added new dike; total injected magma volume = $(InjectVol/1e9) km³; rate Q=$(InjectVol/(time_kyrs*1e3*SecYear)) m³/s")
    end

    SolidFraction!(T, Phi_o, Phi, dPhi_dt, dt);                                                 # Compute solid fraction
    K               .=  Phi.*k_rock .+ (1 .- Phi).*k_magma;                                     # Thermal conductivity

    # Perform a diffusion step
    @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, dt, dx, dz, La, dPhi_dt);  
    @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                      # set lateral boundary conditions (flux-free)
    Tnew[:,1] .= GeoT*H; Tnew[:,end] .= 0.0;                                                    # bottom & top temperature (constant)
    
    Tracers             =   UpdateTracers(Tracers, Grid, Tnew, Phi);                            # Update info on tracers 
    T, Tnew             =   Tnew, T;                                                            # Update temperature
    time,Time_vec[it]   =   time + dt, time/SecYear/1e3;                                        # Keep track of evolved time
    Melt_Time[it]       =   sum( 1.0 .- Phi)/(Nx*Nz)                                            # Melt fraction in crust    
    println(" Timestep $it = $(round(time_kyrs*100)/100) kyrs")

    if mod(it,20)==0  # Visualisation
        Phi_melt    =   1.0 .- Phi;            x_km, z_km  =   x./1e3, z./1e3;
        p1          =   heatmap(x_km, z_km, T',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="$(round(time_kyrs, digits=2)) kyrs", clims=(0.,900.), xlabel="Width [km]",ylabel="Depth [km]", dpi=200, fontsize=6, colorbar_title="Temperature")
        p2          =   heatmap(x_km, z_km, Phi_melt',  aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:vik,     xlabel="Width [km]", clims=(0.,1.), dpi=200, fontsize=6, colorbar_title="Melt Fraction")
        plot(p1, p2, layout=(1,2)); frame(anim)
    end

end
gif(anim, "Example2D.gif", fps = 15)   # create gif animation
return Time_vec, Melt_Time;
end # end of main function

Time_vec,Melt_Time = MainCode_2D(); # start the main code
plot(Time_vec, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt")
```
The main routines are thus ``InjectDike(..)``, which inserts a new dike (of given dimensions and orientation) into the domain, and ``diffusion2D_step!(...)``, which computes thermal diffusion. Variable thermal conductivity, and latent heat are all taken into account. 

The full code example can be downloaded [here](./examples/Example2D.jl)

## 3D example
To go from 2D to 3D, only a few minor changes to the code above are required. Below, we highlight the changes (apart from array initializations, which now have to be done in 3D).

```julia
#(...)
using MagmaThermoKinematics.Diffusion3D
#(...)
using WriteVTK                                   

# Initialize 
@init_parallel_stencil(Threads, Float64, 3);    # initialize parallel stencil in 2D


@views function MainCode_3D();

# Model parameters
W,L,H                   =   30,30,30;               # Width, Length, Height in km
#(...)
x_in,y_in,z_in          =   20e3,20e3,-15e3;        # Center of dike [x,z coordinates in m]
#(..)
Nx, Ny, Nz              =   250, 250, 250;                      # resolution
dx,dy,dz                =   W/(Nx-1)*1e3, L/(Nx-1)*1e3, H*1e3/(Nz-1);    # grid size [m]
#(...)

# Array initializations
#(...)

# Set up model geometry & initial T structure
x,y,z                   =   0:dx:(Nx-1)*dx, 0:dy:(Ny-1)*dy,-H*1e3:dz:(-H*1e3+(Nz-1)*dz);
coords                  =   collect(Iterators.product(x,y,z))               # generate coordinates from 1D coordinate vectors   
X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);         # transfer coords to 3D arrays
Grid, Spacing           =   (x,y,z), (dx,dy,dz);                            # Grid & spacing

#(...)

# Preparation of VTK/Paraview output 
if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; pvd = paraview_collection("Example3D");

#(...)
for it = 1:nt   # Time loop

    if floor(time_kyrs/InjectionInterval_kyrs)> dike_inj        # Add new dike every X years
        #(...)       
        cen                 =   [W/2.;L/2.;-H/2.]; center = (rand(3,1) .- 0.5).*[W_ran;W_ran;H_ran] + cen; # Random variation of location (over a distance 
        dike                =   Dike(Width=W_in, Thickness=H_in,Center=center[:]*1e3,Angle=(rand(2).-0.5).*Angle_ran,Type=DikeType,T=T_in); # Specify dike with random location/angle but fixed size 
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
        x_km,y_km,z_km  =   x./1e3, y./1e3, z./1e3;
        vtkfile = vtk_grid("./viz3D_out/ex3D_$(Int32(it+1e4))", Vector(x_km), Vector(y_km), Vector(z_km)) # 3-D
        vtkfile["Temperature"] = T; vtkfile["MeltFraction"] = Phi_melt;
        outfiles = vtk_save(vtkfile); pvd[time_kyrs] = vtkfile 
    end

end
vtk_save(pvd)


return nothing; #Tracers, T, Grid;
end # end of main function

MainCode_3D(); # start the main code
```

The result are a range of VTK files, which can be visualized with the 3D software [Paraview](https://www.paraview.org). A movie generated from that is:




The full code example can be downloaded [here](./examples/Example3D.jl)




## Dependencies
We rely on [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) to for the energy solver, `StructArrays.jl` to generate an aray of tracer structures, and `Random.jl` for random number generation, `Parameters.jl` to simplify setting parameters (such as specifying dike properties),  `Interpolations.jl` to interpolate properties such as temperature from a fixed grid to tracers,and StaticArrays.jl for speed. All these dependencies should be installed automatically if you download `MagmaThermoKinematics.jl`.

`Plots.jl` is employed for plotting, and `WriteVTK.jl` is used in the 3D example to generate *.VTR/*.PVD files that can be visualized with [Paraview](https://www.paraview.org). You have to add both packages yourself; they are however anyways useful to have.

## Installation
This is a julia package, so after installing julia in the usual manner, you can add the package with 
```
julia>]
  pkg> add https://github.com/boriskaus/MagmaThermoKinematics.jl
```
Next, you can download one of the code above, put it in the directory you are and start it with
```
julia> include("Example2D.jl")
```
If you want to do a full testing of the package on your system, you can run the testing framework from within the package manager:
```
julia>]
  pkg> test MagmaThermoKinematics
```
And finally, if you have installed this package previously on your system, but want to update it to the latest version:
```
julia>]
  pkg> update MagmaThermoKinematics
```




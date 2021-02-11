# ZirconThermoKinematics.jl

Understanding how magmatic systems evolve and how the rock record can be interpreted is of interest to a wide range of Earth Scientists.

This easy to use and versatile package simulates the thermal evolution of magmatic systems, consisting of (kinematically) emplaced dikes. It can take 2D, 2D axisymmetric and 3D geometries into account, and works in parallel on both CPU (and GPU's). A finite difference discretization is employed for the energy equation, combined with semi-lagrangian advection and tracers to track the thermal evolution of emplaced magma. Dikes are emplaced kinematically and the host rock is shifted to accommodate space for the intruding dikes/sills. Cooling, crystallizing and latent heat effects are taken into account, and the thermal evolution of tracers can be used to simulate zircon age distributions.

Below we give a number of example scripts that show how it can be used to simulate a number of scenarios.


## Contents
  - [100-lines 2D example](#100-lines-2d-example)
  - [Dependencies](#dependencies)
  - [Installation](#installation)

## 100-lines 2D example
A simple example that simulates the emplacement of dikes within the crust over a period of 10'000 years is shown below. 

![2-D dike intrusion](examples/movies/Example2D.gif)

The code to simulate this, including visualization, is only 100 lines and the key parts of it are shown below
```
# Load required packages
#(...) 

#------------------------------------------------------------------------------------------
@views function MainCode_2D();

# Model parameters
W,H                     =   50, 30;                             # Width, Height in km
ρ                       =   2800;                               # Density 
cp                      =   1050;                               # Heat capacity
k_rock, k_magma         =   1.5, 1.2;                           # Thermal conductivity of host rock & magma
L                       =   350e3;                              # Latent heat J/kg/K
GeoT                    =   20.0;                               # Geothermal gradient [K/km]
x_in,z_in               =   20e3,   -15e3;                      # Center of dike [x,z coordinates in m]
W_in, H_in              =   5e3,    5e2;                        # Width and thickness of dike [m]
T_in                    =   900;                                # intrusion temperature
InjectionInterval_kyrs  =   0.2;                                # inject a new dike every X years
maxTime_kyrs            =   10;                                 # Maximum simulation time in kyrs
H_ran, W_ran, Angle_ran =   H/4.0, W/4.0, 40.0;                 # size of domain amdin which we randomly place dikes and range of angles   
DikeType                =   "SquareDike"                        # Type to be injected

#(...) 
# Initialize numerical parameters and arrays
#(...) 

# Set up initial temperature structure
T                       .=   -Z./1e3.*GeoT;                     # initial (linear) temperature profile

# Add initial dike
dike                    =   Dike([W_in;H_in],[x_in;z_in],[0], DikeType,T_in); # Specify dike 
T, Tracers, InjectVol   =   InjectDike([], T, Grid, Spacing, dike, nTr_dike); # Inject first dike
Phi,dPhi_dt             =   SolidFraction(T, Phi_o, dt);                      # Compute solid fraction
Tnew                    .=  T;                                         

# Preparation of visualisation
#(...)

time,time_kyrs, dike_inj = 0.0, 0.0, 0.0;
for it = 1:nt   # Time loop

    # Add new dike every X years
    if floor(time_kyrs/InjectionInterval_kyrs)> dike_inj                                                 
        dike_inj            =   floor(time_kyrs/InjectionInterval_kyrs)     # book-keeping
        
        # Vary center of injected dike randomly & generate new dike
        cen                 =   [W/2.; -H/2.]; center = (rand(2,1) .- 0.5).*[W_ran;H_ran] + cen;            
        dike                =   Dike([W_in;H_in], center[:]*1e3 ,(rand(1).-0.5).*Angle_ran, DikeType,T_in); 

        # Inject new dike to domain 
        T, Tracers, Vol     =   InjectDike(Tracers, T, Grid, Spacing, dike, nTr_dike);                     
        InjectVol           +=  Vol                                         # Total injected volume
        println("Added new dike; total injected magma volume = $(InjectVol/1e9) km^3; rate Q=$(InjectVol/(time_kyrs*1e3*SecYear)) m^3/s")
    end

    Phi,dPhi_dt     =   SolidFraction(T, Phi_o, dt);                        # Compute solid fraction
    K               .=  Phi.*k_rock .+ (1 .- Phi).*k_magma;                 # Thermal conductivity
    
    # Perform a diffusion step
    @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, dt, dx, dz,  L, dPhi_dt);  
    @parallel (1:size(T,2)) bc2D_x!(Tnew);                                  # lateral boundaries (flux-free)
    T[:,1] .= GeoT*H; T[:,end] .= 0.0;                                      # bottom & top temperature (constant)
  
    Tracers         =   UpdateTracers(Tracers, Grid, Spacing, Tnew, Phi);   # Update info on tracers 
    T, Tnew         =   Tnew, T;                                            # Update temperature
    time, time_kyrs =   time + dt, time/SecYear/1e3;                        # Keep track of evolved time
    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")

    if mod(it,20)==0  # Visualisation
        Phi_melt    =   1.0 .- Phi;                                         # Melt fraction
        x_km, z_km  =   x./1e3, z./1e3;
 
        p1          =   heatmap(x_km, z_km, T',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="Temperature, $(round(time_kyrs)) kyrs", dpi=300)
        p2          =   heatmap(x_km, z_km, Phi_melt',  aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:vik,     title="Melt fraction",          xlabel="Width [km]", clims=(0.,1.), dpi=150)
        plot(p1, p2, layout=(2,1)); frame(anim)
    end

end
gif(anim, "Temp2D.gif", fps = 15)   # create gif animation
end # end of main function

MainCode_2D() # start the main code
```
The main routines are thus ``InjectDike(..)``, which inserts a new dike (of given dimensions and orientation) into the domain, and ``diffusion2D_step!(...)``, which computes thermal diffusion. Variable thermal conductivity, and latent heat are all taken into account. 

The full code example can be downloaded [here](./examples/Example2D.jl)

## Dependencies
We rely on `ParallelStencil.jl` to for the energy solver, `StructArrays.jl` to generate an aray of tracer structures, and `Random.jl` for random number generation. `Plots.jl` is employed for plotting.  

## Installation
This is a julia package, so after installing julia in the usual manner, you can add the package with 
```
julia>]
  pkg> add https://github.com/boriskaus/ZirconThermoKinematics.jl
```

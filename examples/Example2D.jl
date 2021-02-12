using ZirconThermoKinematics
using ZirconThermoKinematics.Diffusion2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots                                     

# Initialize 
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D

#------------------------------------------------------------------------------------------
@views function MainCode_2D();

# Model parameters
W,H                     =   30, 30;                 # Width, Height in km
ρ                       =   2800;                   # Density 
cp                      =   1050;                   # Heat capacity
k_rock, k_magma         =   1.5, 1.2;               # Thermal conductivity of host rock & magma
L                       =   350e3;                  # Latent heat J/kg/K
GeoT                    =   20.0;                   # Geothermal gradient [K/km]
x_in,z_in               =   20e3,   -15e3;          # Center of dike [x,z coordinates in m]
W_in, H_in              =   5e3,    2e2;            # Width and thickness of dike [m]
T_in                    =   900;                    # intrusion temperature
InjectionInterval_kyrs  =   0.1;                    # inject a new dike every X kyrs
maxTime_kyrs            =   15;                     # Maximum simulation time in kyrs
H_ran, W_ran, Angle_ran =   H/4.0, W/4.0, 90.0;     # size of domain amdin which we randomly place dikes and range of angles   
DikeType                =   "ElasticDike"           # Type to be injected ("SquareDike","ElasticDike")

Nx, Nz                  =   500, 500;                           # resolution
dx                      =   W/(Nx-1)*1e3; dz = H*1e3/(Nz-1);    # grid size [m]
κ                       =   k_rock./(ρ*cp);                     # thermal diffusivity   
dt                      =   min(dx^2, dz^2)./κ/20;              # stable timestep (required for explicit FD)
nt                      =   floor(maxTime_kyrs*1e3*SecYear/dt); # number of required timesteps
nTr_dike                =   1000;                               # number of tracers inserted per dike

# Array initializations (1 - main arrays on which we can initialize properties)
T                       =   @zeros(Nx,   Nz);                    
K                       =   @ones(Nx,    Nz)*k_rock;
Rho                     =   @ones(Nx,    Nz)*ρ;       
Cp                      =   @ones(Nx,    Nz)*cp;

# Work array initialization
Tnew, qx,qz, Kx, Kz     =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
X,Xc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints
Phi_o, Phi, dPhi_dt     =   @zeros(Nx,   Nz), @zeros(Nx,   Nz  ),   @zeros(Nx,   Nz)    # solid fraction

# Set up model geometry & initial T structure
x,z                     =   0:dx:W*1e3, -H*1e3:dz:(-H*1e3+(Nz-1)*dz);
X,Z                     =   ones(Nz)' .* x, z' .* ones(Nx);                             # 2D coordinate grids
Xc                      =   (X[2:Nx,:] + X[1:Nx-1,:])/2.0;
Grid, Spacing           =   (X,Z), (dx,dz);
T                       .=   -Z./1e3.*GeoT;                                             # initial (linear) temperature profile

# Add initial dike
dike                    =   Dike(Width=W_in, Thickness=H_in,Center=[x_in;z_in],Angle=[45],Type=DikeType,T=T_in);  # Specify dike 
T, Tracers, InjectVol   =   InjectDike([], T, Grid, Spacing, dike, nTr_dike);           # Inject first dike
Phi,dPhi_dt             =   SolidFraction(T, Phi_o, dt);                                # Compute solid fraction
Tnew                    .=  T;                                                          # To get correct boundary conditions.

# Preparation of visualisation
ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
println("Animation directory: $(anim.dir)")

time,time_kyrs, dike_inj = 0.0, 0.0, 0.0;
for it = 1:nt   # Time loop

    if floor(time_kyrs/InjectionInterval_kyrs)> dike_inj        # Add new dike every X years
        dike_inj            =   floor(time_kyrs/InjectionInterval_kyrs)                                     # Keeps track on whether we injected already
        cen                 =   [W/2.; -H/2.]; center = (rand(2,1) .- 0.5).*[W_ran;H_ran] + cen;            # Random variation of location (over a distance )
        dike                =   Dike(Width=W_in, Thickness=H_in,Center=center[:]*1e3,Angle=(rand(1).-0.5).*Angle_ran,Type=DikeType,T=T_in); # Specify dike with random location/angle but fixed size 
        T, Tracers, Vol     =   InjectDike(Tracers, T, Grid, Spacing, dike, nTr_dike);                      # Add dike, move hostrocks
        InjectVol           +=  Vol                                                                         # Keep track of injected volume
        println("Added new dike; total injected magma volume = $(InjectVol/1e9) km^3; rate Q=$(InjectVol/(time_kyrs*1e3*SecYear)) m^3/s")
    end

    Phi,dPhi_dt     =   SolidFraction(T, Phi_o, dt);                                            # Compute solid fraction
    K               .=  Phi.*k_rock .+ (1 .- Phi).*k_magma;                                     # Thermal conductivity

    # Perform a diffusion step
    @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, dt, dx, dz,  L, dPhi_dt);  
    @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                      # set lateral boundary conditions (flux-free)
    Tnew[:,1] .= GeoT*H; Tnew[:,end] .= 0.0;                                                    # bottom & top temperature (constant)
    
    Tracers         =   UpdateTracers(Tracers, Grid, Spacing, Tnew, Phi);                       # Update info on tracers 
    T, Tnew         =   Tnew, T;                                                                # Update temperature
    time, time_kyrs =   time + dt, time/SecYear/1e3;                                            # Keep track of evolved time
    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")

    if mod(it,20)==0  # Visualisation
        Phi_melt    =   1.0 .- Phi;             
        x_km, z_km  =   x./1e3, z./1e3;
        p1          =   heatmap(x_km, z_km, T',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="Temperature, $(round(time_kyrs, digits=2)) kyrs",  dpi=150)
        p2          =   heatmap(x_km, z_km, Phi_melt',  aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:vik,     title="Melt fraction",xlabel="Width [km]", clims=(0.,1.), dpi=150)
        plot(p1, p2, layout=(1,2)); frame(anim)
    end

end
gif(anim, "Example2D.gif", fps = 15)   # create gif animation
return Tracers, T, Grid
end # end of main function

Tracers,T,Grid = MainCode_2D(); # start the main code
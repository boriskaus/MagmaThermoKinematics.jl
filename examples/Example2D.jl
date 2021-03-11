using MagmaThermoKinematics
using MagmaThermoKinematics.Units
using MagmaThermoKinematics.Diffusion2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots                                     

# Initialize 
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D

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

# Array initializations
T                       =   @zeros(Nx,   Nz);                    
K                       =   @ones(Nx,    Nz)*k_rock;
Rho                     =   @ones(Nx,    Nz)*ρ;       
Cp                      =   @ones(Nx,    Nz)*cp;

# Work array initialization
Tnew, qx,qz, Kx, Kz     =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
X,Xc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints
Phi_o, Phi, dPhi_dt     =   @zeros(Nx,   Nz), @zeros(Nx,   Nz  ),   @zeros(Nx,   Nz)    # solid fraction

# Set up model geometry & initial T structure
x,z                     =   (0:Nx-1)*dx, (-(Nz-1):0)*dz;                            # 1-D coordinate arrays
crd                     =   collect(Iterators.product(x,z))                         # Generate coordinates from 1D coordinate vectors   
X,Z                     =   (x->x[1]).(crd), (x->x[2]).(crd);                       # Transfer coords to 3D arrays
Grid                    =   (x,z);                                                  # Grid 
Tracers                 =   StructArray{Tracer}(undef, 1)                           # Initialize tracers   
dike                    =   Dike(W=W_in,H=H_in,Type=DikeType,T=T_in);               # "Reference" dike with given thickness,radius and T
T                       .=   -Z.*GeoT;                                              # Initial (linear) temperature profile
Phi, dPhi_dt            =   SolidFraction(T, Phi_o, dt);                            # Compute solid fraction

# Preparation of visualisation
ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])

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
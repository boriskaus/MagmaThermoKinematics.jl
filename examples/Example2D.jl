using MagmaThermoKinematics
using MagmaThermoKinematics.Diffusion2D
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
La                      =   350e3;                  # Latent heat J/kg/K
GeoT                    =   20.0;                   # Geothermal gradient [K/km]
x_in,z_in               =   20e3,   -15e3;          # Center of dike [x,z coordinates in m]
W_in, H_in              =   5e3,    200;            # Width and thickness of dike [m]
T_in                    =   900;                    # intrusion temperature
InjectionInterval_kyrs  =   0.1;                    # inject a new dike every X kyrs
maxTime_kyrs            =   25;                     # Maximum simulation time in kyrs
H_ran, W_ran            =   H*0.4, W*0.3;           # size of domain in which we randomly place dikes and range of angles   
DikeType                =   "ElasticDike"           # Type to be injected ("SquareDike","ElasticDike")

Nx, Nz                  =   500, 500;                           # resolution
dx                      =   W/(Nx-1)*1e3; dz = H*1e3/(Nz-1);    # grid size [m]
κ                       =   k_rock./(ρ*cp);                     # thermal diffusivity   
dt                      =   min(dx^2, dz^2)./κ/10;              # stable timestep (required for explicit FD)
nt::Int64               =   floor(maxTime_kyrs*1e3*SecYear/dt); # number of required timesteps
nTr_dike                =   300;                                # number of tracers inserted per dike

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
x,z                     =   0:dx:(Nx-1)*dx, -H*1e3:dz:(-H*1e3+(Nz-1)*dz);
coords                  =   collect(Iterators.product(x,z))                 # generate coordinates from 1D coordinate vectors   
X,Z                     =   (x->x[1]).(coords), (x->x[2]).(coords);         # transfer coords to 3D arrays
Grid, Spacing           =   (x,z), (dx,dz);                                 # Grid & spacing
Tracers                 =   StructArray{Tracer}(undef, 1)                   # Initialize tracers   
T                       .=   -Z./1e3.*GeoT;                                 # Initial (linear) temperature profile
SolidFraction!(T, Phi_o, Phi, dPhi_dt, dt);                                 # Compute solid fraction

# Preparation of visualisation
ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])

time,time_kyrs, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
for it = 1:nt   # Time loop

    if floor(time_kyrs/InjectionInterval_kyrs)> dike_inj                            # Add new dike every X years
        dike_inj  =   floor(time_kyrs/InjectionInterval_kyrs)                       # Keeps track on what was injected already
        cen       =   [W/2.; -H/2.] + rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];        # Randomly vary center of dike 
        if cen[end]<-12;    Angle_rand = rand( 80.0:0.1:100.0)                      # Orientation: near-vertical @ depth             
        else                Angle_rand = rand(-10.0:0.1:10.0); end                  # Orientation: near-vertical @ shallower depth     
        dike      =   Dike(Width=W_in, Thickness=H_in, Center=cen[:]*1e3,Angle=[Angle_rand],Type=DikeType,T=T_in); # Specify dike with random location/angle but fixed size 
        Tracers, T, Vol     =   InjectDike(Tracers, T, Grid, dike, nTr_dike);       # Add dike, move hostrocks
        InjectVol           +=  Vol                                                 # Keep track of injected volume
        println("Added new dike; total injected magma volume = $(InjectVol/1e9) km³; rate Q=$(InjectVol/(time_kyrs*1e3*SecYear)) m³/s")
    end

    SolidFraction!(T, Phi_o, Phi, dPhi_dt, dt);                                     # Compute solid fraction
    K               .=  Phi.*k_rock .+ (1 .- Phi).*k_magma;                         # Thermal conductivity

    # Perform a diffusion step
    @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, dt, dx, dz, La, dPhi_dt);  
    @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                      # set lateral boundary conditions (flux-free)
    Tnew[:,1] .= GeoT*H; Tnew[:,end] .= 0.0;                                                    # bottom & top temperature (constant)
    
    Tracers             =   UpdateTracers(Tracers, Grid, Tnew, Phi);                            # Update info on tracers 
    T, Tnew             =   Tnew, T;                                                            # Update temperature
    time,time_kyrs      =   time + dt, time/SecYear/1e3;                                        # Keep track of evolved time
    Melt_Time[it]       =   sum( 1.0 .- Phi)/(Nx*Nz)                                            # Melt fraction in crust    
    Time_vec[it]        =   time_kyrs;                                                          # Vector with time
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
plot(Time_vec, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt") #Create plot
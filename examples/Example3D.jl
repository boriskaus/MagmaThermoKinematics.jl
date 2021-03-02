using MagmaThermoKinematics
using MagmaThermoKinematics.Diffusion3D
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Plots  
using WriteVTK                                   

# Initialize 
@init_parallel_stencil(Threads, Float64, 3);    # initialize parallel stencil in 2D

#------------------------------------------------------------------------------------------
@views function MainCode_3D();

# Model parameters
W,L,H                   =   30,30,30;               # Width, Length, Height in km
ρ                       =   2800;                   # Density 
cp                      =   1050;                   # Heat capacity
k_rock, k_magma         =   1.5, 1.2;               # Thermal conductivity of host rock & magma
La                      =   350e3;                  # Latent heat J/kg/K
GeoT                    =   20.0;                   # Geothermal gradient [K/km]
x_in,y_in,z_in          =   20e3,20e3,-15e3;        # Center of dike [x,z coordinates in m]
#W_in, H_in              =   5e3,  2e2;              # Width and thickness of dike [m]
W_in, H_in              =   5e3,  5e2;              # Width and thickness of dike [m]

T_in                    =   900;                    # Intrusion temperature
InjectionInterval_kyrs  =   0.1;                    # Inject a new dike every X kyrs
maxTime_kyrs            =   15;                     # Maximum simulation time in kyrs
H_ran, W_ran            =   H*0.4, W*0.3;           # Size of domain in which we randomly place dikes and range of angles   
DikeType                =   "ElasticDike"           # Type to be injected ("SquareDike","ElasticDike")

#Nx, Ny, Nz              =   500, 500, 500;                      # resolution
Nx, Ny, Nz              =   250, 250, 250;                      # resolution

dx,dy,dz                =   W/(Nx-1)*1e3, L/(Nx-1)*1e3, H*1e3/(Nz-1);    # grid size [m]
κ                       =   k_rock./(ρ*cp);                     # thermal diffusivity   
dt                      =   min(dx^2,dy^2, dz^2)./κ/10;         # stable timestep (required for explicit FD)
nt::Int64               =   floor(maxTime_kyrs*1e3*SecYear/dt); # number of required timesteps
nTr_dike                =   1000;                               # number of tracers inserted per dike

# Array initializations
T                       =   @zeros(Nx,Ny,Nz);                    
K                       =   @ones(Nx,Ny, Nz)*k_rock;
Rho                     =   @ones(Nx,Ny, Nz)*ρ;       
Cp                      =   @ones(Nx,Ny, Nz)*cp;

# Work array initialization
Tnew, qx,qy,qz          =   @zeros(Nx,Ny,Nz),   @zeros(Nx-1,Ny,Nz), @zeros(Nx,Ny-1,Nz), @zeros(Nx,Ny,Nz-1)  # thermal solver
Kx, Ky, Kz              =   @zeros(Nx-1,Ny,Nz), @zeros(Nx,Ny-1,Nz), @zeros(Nx,Ny,Nz-1)                      # thermal conductivities
Phi_o, Phi, dPhi_dt     =   @zeros(Nx,Ny,Nz),   @zeros(Nx,  Ny,Nz), @zeros(Nx,Ny,  Nz)                      # solid fraction

# Set up model geometry & initial T structure
x,y,z                   =   0:dx:(Nx-1)*dx, 0:dy:(Ny-1)*dy,-H*1e3:dz:(-H*1e3+(Nz-1)*dz);
coords                  =   collect(Iterators.product(x,y,z))               # generate coordinates from 1D coordinate vectors   
X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);         # transfer coords to 3D arrays
Grid, Spacing           =   (x,y,z), (dx,dy,dz);                            # Grid & spacing
Tracers                 =   StructArray{Tracer}(undef, 1)                   # Initialize tracers   
T                       .=   -Z./1e3.*GeoT;                                 # Initial (linear) temperature profile
SolidFraction!(T, Phi_o, Phi, dPhi_dt, dt);                                 # Compute solid fraction

# Preparation of VTK/Paraview output 
if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; pvd = paraview_collection("Example3D");

time,time_kyrs, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
for it = 1:nt   # Time loop

    if floor(time_kyrs/InjectionInterval_kyrs)> dike_inj                                        # Add new dike every X years
        dike_inj        =   floor(time_kyrs/InjectionInterval_kyrs)                             # Keeps track on whether we injected already
        cen             =   [W/2.;L/2.;-H/2.] + rand(-0.5:1e-3:0.5, 3).*[W_ran;W_ran;H_ran];    # Randomly vary center of dike 
        if cen[end]<-12;    Angle_rand = [rand(80.0:0.1:100.0); rand(0:360)]                    # Dikes at depth             
        else                Angle_rand = [rand(-10.0:0.1:10.0); rand(0:360)] end                # Sills at shallower depth
        dike            =   Dike(Width=W_in, Thickness=H_in,Center=cen[:]*1e3,Angle=Angle_rand,Type=DikeType,T=T_in); # Specify dike with random location/angle but fixed size 
        Tracers, T, Vol =   InjectDike(Tracers, T, Grid, dike, nTr_dike);                       # Add dike, move hostrocks
        InjectVol       +=  Vol                                                                 # Keep track of injected volume
        println("Added new dike; total injected magma volume = $(InjectVol/1e9) km³; rate Q=$(InjectVol/(time_kyrs*1e3*SecYear)) m³/s")
    end
    SolidFraction!(T, Phi_o, Phi, dPhi_dt, dt);                                                 # Compute solid fraction
    K               .=  Phi.*k_rock .+ (1 .- Phi).*k_magma;                                     # Thermal conductivity

    # Perform a diffusion step
    @parallel diffusion3D_step_varK!(Tnew, T, qx, qy, qz, K, Kx, Ky, Kz, Rho, Cp, dt, dx, dy, dz,  La, dPhi_dt);  
    @parallel (1:size(T,2), 1:size(T,3)) bc3D_x!(Tnew);                                         # set lateral boundary conditions (flux-free)
    @parallel (1:size(T,1), 1:size(T,3)) bc3D_y!(Tnew);                                         # set lateral boundary conditions (flux-free)
    Tnew[:,:,1] .= GeoT*H; Tnew[:,:,end] .= 0.0;                                                # bottom & top temperature (constant)
    
    Tracers         =   UpdateTracers(Tracers, Grid, Tnew, Phi);                                # Update info on tracers 
    T, Tnew         =   Tnew, T;                                                                # Update temperature
    time, time_kyrs =   time + dt, time/SecYear/1e3;                                            # Keep track of evolved time
    Melt_Time[it]   =   sum( 1.0 .- Phi)/(Nx*Ny*Nz)                                             # Melt fraction in crust    
    Time_vec[it]    =   time_kyrs;                                                              # Vector with time
    println(" Timestep $it = $(round(time_kyrs*100)/100) kyrs")

    if mod(it,5)==0  # Visualisation
        Phi_melt        =   1.0 .- Phi;   x_km,y_km,z_km  =   x./1e3, y./1e3, z./1e3;
        vtkfile = vtk_grid("./viz3D_out/ex3D_$(Int32(it+1e4))", Vector(x_km), Vector(y_km), Vector(z_km)) # 3-D vtk file
        vtkfile["Temperature"] = T; vtkfile["MeltFraction"] = Phi_melt;                         # Store fields in file
        outfiles = vtk_save(vtkfile); pvd[time_kyrs] = vtkfile                                  # Save file & update pvd file
    end
end
vtk_save(pvd)   # save Example3D.pvd file, which you can open with, for example, paraview
return Time_vec, Melt_Time      
end             # end of main function

MainCode_3D(); # start the main code
plot(Time_vec, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example3D") #Create plot
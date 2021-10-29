using MagmaThermoKinematics
using MagmaThermoKinematics.Diffusion2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots                     
using GeoParams
using Printf


# Initialize 
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D

#------------------------------------------------------------------------------------------
@views function MainCode_2D();


K = u"K"    # unsure why this is not available from GeoParams

@show typeof(kg) typeof(kJ) typeof(km) typeof(C) typeof(μW)
CharDim      = GEO_units(length=1000km, temperature=1000C, stress=10MPa, viscosity=1e20Pas);

# Define material parameters for the simulation. By using GeoParams 
MatParam     = Array{MaterialParams, 1}(undef, 2);
MatParam[1]  = SetMaterialParams(Name="Host Rock", Phase=1, 
                                 Density    = ConstantDensity(ρ=2700/m^3),
                          EnergySourceTerms = ConstantLatentHeat(Q_L=313/kg),
                               #Conductivity = ConstantConductivity(k=1.5Watt/K/m),     # in case we use constant k
                               Conductivity = T_Conductivity_Whittacker(),
                               HeatCapacity = ConstantHeatCapacity(cp=1000/kg/K),
                                    Melting = MeltingParam_Caricchi())

MatParam[2]  = SetMaterialParams(Name="Melt", Phase=2, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                             EnergySourceTerms = ConstantLatentHeat(Q_L=313kJ/kg),
                                  Conductivity = ConstantConductivity(k=1.2Watt/K/m),
                                  HeatCapacity = ConstantHeatCapacity(cp=1000/kg/K),
                                       Melting = MeltingParam_Caricchi())

# These                                        
SecYear = 3600*24*365.25
kyr = 1000*SecYear
km³ = 1000^3

# Model parameters
W,H                     =   25e3, 20e3;                 # Width, Height
ρ                       =   2700;                       # Density 
cp                      =   1000;                       # Heat capacity
k_rock, k_magma         =   1.5, 1.2;                   # Thermal conductivity of host rock & magma
La                      =   350e3;                      # Latent heat J/kg/K
GeoT                    =   50.0/1000;                  # Geothermal gradient [K/km]
W_in, H_in              =   20e3,    100;               # Width and thickness of each dike that is intruded

T_in                    =   1115;                       # Intrusion temperature
maxTime                 =   1250kyr;                    # Maximum simulation time in kyrs

TotalVolume             =   1580e3^3           
maxThickness            =   5000                        # max thickness of dike
nInjections             =   maxThickness/H_in
InjectionInterval       =   maxTime/nInjections;        # Inject a new dike every X kyrs


DikeType                =   "SquareDike"              # Type to be injected ("ElasticDike","SquareDike")
#DikeType                =   "SquareDike_TopAccretion"   # Type to be injected ("ElasticDike","SquareDike", "SquareDike_TopAccretion")
#DikeType                =   "ElasticDike"   # Type to be injected ("ElasticDike","SquareDike", "SquareDike_TopAccretion")

cen                     =   [0.; -6.5e3 - 100/2];       # Center of dike 

Nx, Nz                  =   200, 250*2;               # resolution (grid cells in x,z direction)
dx,dz                   =   W/(Nx-1), H/(Nz-1);     # grid size
κ                       =   k_rock./(ρ*cp);         # thermal diffusivity   
dt                      =   min(dx^2, dz^2)./κ/20;  # stable timestep (required for explicit FD)
nt::Int64               =   floor(maxTime/dt);      # number of required timesteps
nTr_dike                =   300;                    # number of tracers inserted per dike

@show nt maxTime dt

# Array initializations
T                       =   @zeros(Nx,   Nz);                    
Kc                      =   @ones(Nx,    Nz)*k_rock;
Rho                     =   @ones(Nx,    Nz)*ρ;       
Cp                      =   @ones(Nx,    Nz)*cp;

# Work array initialization
Tnew, qr,qz, Kr, Kz     =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
R,Rc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints
Phi_o, Phi, dPhi_dt     =   @zeros(Nx,   Nz), @zeros(Nx,   Nz  ),   @zeros(Nx,   Nz)    # solid fraction

# Set up model geometry & initial T structure
x,z                     =   (0:Nx-1)*dx, (-(Nz-1):0)*dz;                            # 1-D coordinate arrays
crd                     =   collect(Iterators.product(x,z))                         # Generate coordinates from 1D coordinate vectors   
R,Z                     =   (x->x[1]).(crd), (x->x[2]).(crd);                       # Transfer coords to 3D arrays
Rc                      =   (R[2:end,:] + R[1:end-1,:])/2 
Grid                    =   (x,z);                                                  # Grid 
Tracers                 =   StructArray{Tracer}(undef, 1)                           # Initialize tracers   
dike                    =   Dike(W=W_in,H=H_in,Type=DikeType,T=T_in);               # "Reference" dike with given thickness,radius and T
T                       .=   -Z.*GeoT;                                              # Initial (linear) temperature profile
Phi, dPhi_dt            =   SolidFraction(T, Phi_o, dt);                            # Compute solid fraction

# Preparation of visualisation
ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])

time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
for it = 1:nt   # Time loop

    if floor(time/InjectionInterval)> dike_inj              # Add new dike every X years
        dike_inj   =   floor(time/InjectionInterval)        # Keeps track on what was injected already
        Angle_rand =   0;
        dike       =   Dike(dike, Center=cen[:],Angle=[Angle_rand]);                # Specify dike with random location/angle but fixed size/T 
        Tracers, T, Vol     =   InjectDike(Tracers, T, Grid, dike, nTr_dike);       # Add dike, move hostrocks
        InjectVol           +=  Vol                                                 # Keep track of injected volume
        Qrate           = InjectVol/time
        Qrate_km3_yr    = Qrate*SecYear/(1e3^3)
        Qrate_km3_yr_km2    = Qrate_km3_yr/(pi*(W_in/2/1e3)^2)
        
        @printf "Added new dike; total injected magma volume = %.2f km³; rate Q=%.2f m³s⁻¹ = %.2e km³yr⁻¹ = %.2e km³yr⁻¹km⁻² \n" InjectVol/km³ Qrate Qrate_km3_yr Qrate_km3_yr_km2

     #   println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(Qrate,digits=2)) m³/s, Q=$(Qrate_km3_yr) km³/yr = $(Qrate_km3_yr_km2)km³/yr/km^2")
    end
    
    T_K         =  (T .+ 273.15).*K 
    Phi_melt    =   ComputeMeltingParam(0,T_K,MatParam[1].Melting[1])
    Phi         =   1.0 .- Phi_melt;
    dPhi_dt     =   (Phi-Phi_o)./dt
  #  Phi, dPhi_dt     =  SolidFraction(T, Phi_o, dt);                                # Compute solid fraction

    Kc          =   ComputeConductivity(0,T_K,MatParam[1].Conductivity[1])           # Thermal conductivity
    Kc          =   ustrip.(Kc)
    
    
    # Perform a diffusion step
    #@parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kr, Kz, Rho, Cp, dt, dr, dz, La, dPhi_dt);  
    @parallel diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, Kc, Kr, Kz, Rho, Cp, dt, dx, dz, La, dPhi_dt) 

    # set lateral boundary conditions (flux-free)
    @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                     
    
    #Tnew[:,1] .= GeoT*H;                               # fixed bottom temperature (constant)
    @parallel (1:size(T,1)) bc2D_z_bottom!(Tnew);       # flux-free bottom BC                                                
    
    # Fixed top BC
    Tnew[:,end] .= 0.0;                                 # fixed top temperature (constant)

    
    Tracers             =   UpdateTracers(Tracers, Grid, Tnew, Phi);                            # Update info on tracers 
    T, Tnew             =   Tnew, T;                                                            # Update temperature
    time                =   time + dt;                                                          # Keep track of evolved time
    Melt_Time[it]       =   sum( 1.0 .- Phi)/(Nx*Nz)                                            # Melt fraction in crust    
    Time_vec[it]        =   time;                                                               # Vector with time

    if mod(it,1000)==0  # Visualisation
        println(" Timestep $it = $(round(time/kyr*100)/100) kyrs, maxT = $(maximum(T)) C")
       # Phi_melt    =   1.0 .- Phi;     
        
        p1          =   heatmap(x/1000, z/1000, T',     levels=0:125:1000,    fill=true,    aspect_ratio=1, xlims=(x[1]/1000,x[end]/1000), ylims=(z[1]/1000,z[end]/1000),   c=:nipy_spectral, clims=(0.,1000.), xlabel="Width [km]",ylabel="Depth [km]", title="$(round(time/kyr, digits=2)) kyrs", dpi=200, fontsize=6, colorbar_title="Temperature")
        p1          =   contour!(x/1000, z/1000, T',     levels=0:125:1000,    fill=false,    aspect_ratio=1, xlims=(x[1]/1000,x[end]/1000), ylims=(z[1]/1000,z[end]/1000), c=:black,  xlabel="Width [km]",ylabel="Depth [km]", title="$(round(time/kyr, digits=2)) kyrs", dpi=200, fontsize=6, colorbar_title="Temperature")
     #   p1          =   contour!(x/1000, z/1000, T',     levels=700,    fill=false,    aspect_ratio=1, xlims=(x[1]/1000,x[end]/1000), ylims=(z[1]/1000,z[end]/1000), c=:white,  xlabel="Width [km]",ylabel="Depth [km]", title="$(round(time/kyr, digits=2)) kyrs", dpi=200, fontsize=6, colorbar_title="Temperature")
        
        p2          =   heatmap(x/1000, z/1000, Phi_melt', aspect_ratio=1, xlims=(x[1]/1000,x[end]/1000), ylims=(z[1]/1000,z[end]/1000),   c=:nuuk,     xlabel="Width [km]",             dpi=200, fontsize=6, colorbar_title="Melt Fraction")
       # p3          =   heatmap(x/1000, z/1000, Phi_melt1', aspect_ratio=1, xlims=(x[1]/1000,x[end]/1000), ylims=(z[1]/1000,z[end]/1000),   c=:nuuk,     xlabel="Width [km]",             dpi=200, fontsize=6, colorbar_title="Melt Fraction")
        
        plot(p1, p2, layout=(1,2)); frame(anim)
    end
end
gif(anim, "Example2D.gif", fps = 15)   # create gif animation
return Time_vec, Melt_Time;
end # end of main function

Time_vec,Melt_Time = MainCode_2D(); # start the main code
#plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot
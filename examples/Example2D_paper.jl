using MagmaThermoKinematics
using MagmaThermoKinematics.Diffusion2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
#using Plots 
using CairoMakie
using GeoParams
using Printf
using MAT           # saves files in matlab format

using Statistics
using LinearAlgebra: norm

# Initialize 
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D

#------------------------------------------------------------------------------------------
@views function MainCode_2D(Mat_tup);

# These are useful parameters                                       
SecYear     = 3600*24*365.25
kyr         = 1000*SecYear
Myr         = 1e6*SecYear  
km³         = 1000^3

# Model parameters
W,H                     =   20e3, 20e3;                 # Width, Height
Tsurface_C              =   0.0;
GeoT                    =   40.0/1000;                  # Geothermal gradient [K/km]
maxTime                 =   1.5Myr;                     # Maximum simulation time in kyrs

SimName                 =   "Zassy_Geneva_zeroFlux"   # name of simulation
SaveOutput_steps        =   1e4;                        # saves output every x steps 
CreateFig_steps         =   2000;                       # Create a figure every X steps

if ~isempty(Mat_tup[1].EnergySourceTerms)
    La =  NumValue(Mat_tup[1].EnergySourceTerms[1].Q_L);             # latent heat 
else
    La = 0.
end

# Specify sill parameters ----------------
InjectionInterval       =   10kyr;                      # Inject a new dike every X kyrs
W_in, H_in              =   20e3,   74.6269;            # Width and thickness of each dike that is intruded
T_in                    =   1000;                       # Intrusion temperature
SillThickness           =   H_in                        # Sill thickness [m]. Warning: This must be equal to an integer number of dz

#DikeType                =   "SquareDike"              # Type to be injected ("ElasticDike","SquareDike")
DikeType                =   "CylindricalDike_TopAccretion"   # Type to be injected ("ElasticDike","SquareDike", "SquareDike_TopAccretion")
#DikeType = "CylindricalDike_TopAccretion_FullModelAdvection"
#DikeType                =   "ElasticDike"              # Type to be injected ("ElasticDike","SquareDike", "SquareDike_TopAccretion")
cen                     =   [0.; -7.0e3 - 0/2];         # Center of dike 

# Compute derived parameters from what is specified above
SillRadius              =   W_in/2                      # Sill radius    
SillArea                =   pi*SillRadius^2             # horizontal area of sill
SillVolume              =   SillArea*SillThickness      # in m^3 

nInjections             =   maxTime/InjectionInterval   
TotalVolume             =   SillVolume*nInjections
TotalVolume_km3         =   TotalVolume/1e9  # in km^3     
ThicknessInjectedCrust  =   nInjections*SillThickness               # in m
VertAccretionRate_m_yr  =   ThicknessInjectedCrust/maxTime*SecYear  # vertical accretion rate in m/yr
MagmaFlux_km3_yr        =   TotalVolume_km3/maxTime*SecYear                     

Flux                    =   TotalVolume/SillArea/maxTime*SecYear    # Magma flux in m3/yr/m2 
MagmaFlux_km3_yr_km2    =   Flux/1e3                                # vertical magma flux in km3/yr/km2
maxThickness            =   maxTime/SecYear*Flux                    # max thickness of dike (from flux)

flux_free_bottom_BC     =   true
Nx, Nz                  =   269, 269;                    # resolution (grid cells in x,z direction)
dx,dz                   =   W/(Nx-1), H/(Nz-1);         # grid size
κ                       =   3.3/(1000*2700)
dt                      =   0.4*min(dx^2, dz^2)./κ/4;   # stable timestep (required for explicit FD)
nt::Int64               =   floor(maxTime/dt);          # number of required timesteps
nTr_dike                =   300;                        # number of tracers inserted per dike
@show dt/SecYear

# Array initializations
T                       =   @zeros(Nx,   Nz);    
Kc                      =   @ones(Nx,    Nz);
Rho                     =   @ones(Nx,    Nz);       
Cp                      =   @ones(Nx,    Nz);
Phi_melt, dϕdT          =   @zeros(Nx,   Nz), @zeros(Nx,   Nz  )                        # Melt fraction and derivative of mekt fraction vs T

# Work array initialization
Tnew,T_init,qr,qz,Kr,Kz =   @zeros(Nx,   Nz), @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
R,Rc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints

# Set up model geometry & initial T structure
x,z                     =   (0:Nx-1)*dx, (-(Nz-1):0)*dz;                            # 1-D coordinate arrays
crd                     =   collect(Iterators.product(x,z))                         # Generate coordinates from 1D coordinate vectors   
R,Z                     =   (x->x[1]).(crd), (x->x[2]).(crd);                       # Transfer coords to 3D arrays
Rc                      =   (R[2:end,:] + R[1:end-1,:])/2 
Grid                    =   (x,z);                                                  # Grid 
Tracers                 =   StructArray{Tracer}(undef, 1)                           # Initialize tracers   
dike                    =   Dike(W=W_in,H=H_in,Type=DikeType,T=T_in);               # "Reference" dike with given thickness,radius and T
T_init                 .=   Tsurface_C .- Z.*GeoT;                                  # Initial (linear) temperature profile
@parallel assign!(Tnew, T_init)
@parallel assign!(T, T_init)

P                       =   @zeros(Nx,Nz);
Phases                  =   ones(Int64,Nx,Nz)
time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);

if isdir(SimName)==false mkdir(SimName) end;    # create simulation directory if needed

for it = 1:nt   # Time loop

    # Add new dike every X years:
    if floor(time/InjectionInterval)> dike_inj        
        dike_inj            =   floor(time/InjectionInterval)                   # Keeps track on what was injected already
        Angle_rand          =   0;
        dike                =   Dike(dike, Center=cen[:],Angle=[Angle_rand]);   # Specify dike with random location/angle but fixed size/T 
        Tracers, T,Vol      =   InjectDike(Tracers, T, Grid, dike, nTr_dike,AdvectionMethod="Euler",InterpolationMethod="Quadratic");   # Add dike, move hostrocks
        InjectVol           +=  Vol                                             # Keep track of injected volume
        Qrate               =   InjectVol/time
        Qrate_km3_yr        =   Qrate*SecYear/km³
        Qrate_km3_yr_km2    =   Qrate_km3_yr/(pi*(W_in/2/1e3)^2)
        @printf "  Added new dike; total injected magma volume = %.2f km³; rate Q= %.2e km³yr⁻¹ = %.2e km³yr⁻¹km⁻² \n" InjectVol/km³ Qrate_km3_yr Qrate_km3_yr_km2
    end
    
    # Update material properties: rho, cp, k and dϕdT 
    T_K         =  T .+ 273.15     # all GeoParams routines expect T in K
    compute_meltfraction!(Phi_melt, Mat_tup, Phases, P, T_K) 
    compute_density!(Rho, Mat_tup, Phases, P,     T_K)
    compute_heatcapacity!(Cp, Mat_tup, Phases, P, T_K)
    compute_conductivity!(Kc, Mat_tup, Phases, P, T_K)
    compute_dϕdT!(dϕdT, Mat_tup, Phases, P,       T_K)
    
    # Perform a diffusion step
    # Perform nonlinear iterations (for latent heat which depends on T)
    err, iter = 1., 1
    while err>1e-6 && iter<20
        @parallel diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, Kc, Kr, Kz, Rho, Cp, dt, dx, dz, La, dϕdT) # diffusion step
        @parallel (1:size(T,2)) bc2D_x!(Tnew);                      # flux-free lateral boundary conditions
        if flux_free_bottom_BC==true
            @parallel (1:size(T,1)) bc2D_z_bottom!(Tnew);           # flux-free bottom BC  (if false=isothermal)
        end
        
        dϕdT_o  = dϕdT
        compute_dϕdT!(dϕdT, Mat_tup, Phases, P,  Tnew .+ 273.15)    # update dϕdT using latest estimate for T
        err     = norm(dϕdT-dϕdT_o)                                 # compute error
        iter   += 1
    end


    # Update variables
    Tracers             =   UpdateTracers(Tracers, Grid, Tnew, Phi_melt);      # Update info on tracers 
    @parallel assign!(T, Tnew)
    @parallel assign!(Tnew, T)
    time                =   time + dt;                                         # Keep track of evolved time
    Melt_Time[it]       =   sum( Phi_melt)/(Nx*Nz)                             # Melt fraction in crust    
    Time_vec[it]        =   time;                                              # Vector with time
    
    # Visualize results
    if mod(it,CreateFig_steps)==0  || it==nt 
        time_Myrs = time/Myr;
        # ---------------------------------
        # Create plot (using Makie)
        fig = Figure(resolution = (2000,1000))

        # 1D figure with cross-sections
        time_Myrs_rnd = round(time_Myrs,digits=3)
        ax1 = Axis(fig[1,1], xlabel = "Temperature [ᵒC]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs")
        lines!(fig[1,1],T[1,:],z/1e3,label="Center")
        lines!(fig[1,1],T[end,:],z/1e3,label="Side")
        lines!(fig[1,1],T_init[end,:],z/1e3,label="Initial")
        axislegend(ax1)
        limits!(ax1, 0, 1000, -20, 0)
      
        # 2D temperature plot 
        ax2=Axis(fig[1, 2],xlabel = "Width [km]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs, Geneva model; Temperature [ᵒC]")
        co = contourf!(fig[1, 2], x/1e3, z/1e3, T, levels = 0:50:1000,colormap = :jet)
        co1 = contour!(fig[1, 2], x/1e3, z/1e3, T, levels = 690:690)       # solidus
        limits!(ax2, 0, 20, -20, 0)
        Colorbar(fig[1, 3], co)
        
        # 2D melt fraction plots:
        ax3=Axis(fig[1, 4],xlabel = "Width [km]", ylabel = "Depth [km]", title = " ϕ (melt fraction)")
        co = heatmap!(fig[1, 4], x/1e3, z/1e3, Phi_melt, colormap = :vik, colorrange=(0, 1))
        limits!(ax3, 0, 20, -20, 0)
        Colorbar(fig[1, 5], co)
        
        
        # Save figure
        save("$(SimName)/$(SimName)_$it.png", fig)
        #
        # ---------------------------------
        println("Timestep $it = $(round(time/kyr*100)/100) kyrs, maxT = $(round(maximum(T),digits=3)) ᵒC")
    end

    # Save output to disk once in a while
    if mod(it,SaveOutput_steps)==0  || it==nt 
        filename = "$(SimName)/$(SimName)_$it.mat"
        matwrite(filename, 
                        Dict("Tnew"=> Tnew, 
                             "time"=> time, 
                             "x"   => Vector(x), 
                             "z"   => Vector(z))
            )
        println("  Saved matlab output to $filename")    
    end

end

return x,z,T, Time_vec, Melt_Time, Tracers;
end # end of main function


# Define material parameters for the simulation. 

# Note: by using GeoParams we can easily change parameters without having to modify the main code
# Note that we currently assume that the host rock & melt have the same properties
# If you want that to be different, you'll have to declare properties for the other phases as well 
#  (& updated the Phases array in the main code)

MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                 Density    = ConstantDensity(ρ=2700/m^3),
                          EnergySourceTerms = ConstantLatentHeat(Q_L=3.13e5J/kg),
                          #     Conductivity = ConstantConductivity(k=1.0Watt/K/m),     # in case we use constant k
                               Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                               #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                               HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                    Melting = MeltingParam_4thOrder()),                     # Marxer & Ulmer data
                # add more parameters here, in case you have >1 phase in the model                                    
                )

# Call the main code with the specified material parameters
x,z,T, Time_vec,Melt_Time, Tracers = MainCode_2D(MatParam); # start the main code






#plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot
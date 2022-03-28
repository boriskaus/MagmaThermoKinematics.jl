using MagmaThermoKinematics
using MagmaThermoKinematics.Diffusion2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
#using Plots 
#using CairoMakie
using GLMakie
using GeoParams
using Printf

using Statistics

# Initialize 
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D

#------------------------------------------------------------------------------------------
@views function MainCode_2D(Mat_tup);


K = u"K"    # unsure why this is not available from GeoParams


# These are usefull parameters                                       
SecYear     = 3600*24*365.25
kyr         = 1000*SecYear
Myr         = 1e6*SecYear  
km³         = 1000^3

# Model parameters
W,H                     =   20e3, 20e3;                 # Width, Height
GeoT                    =   40.0/1000;                  # Geothermal gradient [K/km]
W_in, H_in              =   20e3,    100;               # Width and thickness of each dike that is intruded

T_in                    =   1000;                       # Intrusion temperature
maxTime                 =   1.5Myr;                     # Maximum simulation time in kyrs

TotalVolume             =   1580e3^3           
Flux                    =   7e-6*1000                   # in m^3/yr/m^2
maxThickness            =   maxTime/SecYear*Flux        # max thickness of dike (from flux)
@show maxThickness
nInjections             =   maxThickness/H_in
InjectionInterval       =   maxTime/nInjections;        # Inject a new dike every X kyrs


#DikeType                =   "SquareDike"              # Type to be injected ("ElasticDike","SquareDike")
#DikeType                =   "CylindricalDike_TopAccretion"   # Type to be injected ("ElasticDike","SquareDike", "SquareDike_TopAccretion")
DikeType = "CylindricalDike_TopAccretion_FullModelAdvection"
#DikeType                =   "ElasticDike"              # Type to be injected ("ElasticDike","SquareDike", "SquareDike_TopAccretion")

cen                     =   [0.; -6.5e3 - 100/2];       # Center of dike 

flux_free_bottom_BC     =   true
Nx, Nz                  =   101, 101;                # resolution (grid cells in x,z direction)
dx,dz                   =   W/(Nx-1), H/(Nz-1);      # grid size
κ = 2.8/(1000*2700)
#κ                       =   1e-6                    # approximate thermal diffusivity   (real one may be T-dependent)
dt                      =   min(dx^2, dz^2)./κ/15;  # stable timestep (required for explicit FD)
nt::Int64               =   floor(maxTime/dt);      # number of required timesteps
nTr_dike                =   300;                    # number of tracers inserted per dike

@show nt maxTime dt

# Array initializations
T                       =   @zeros(Nx,   Nz);                    
Kc                      =   @ones(Nx,    Nz);
Rho                     =   @ones(Nx,    Nz);       
Cp                      =   @ones(Nx,    Nz);

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
P                       =   @zeros(Nx,Nz);

Phases                  =   ones(Int64,Nx,Nz)

# Compute initial melt fraction
Phi_melt                =   Phi;
compute_meltfraction!(Phi_melt, Mat_tup, Phases, P,T)                               # Compute melt fraction
Phi                     =   1.0 .- Phi_melt;
dPhi_dt                 =   (Phi-Phi_o)./dt


# Preparation of visualisation
#ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])

#=
# Create main figure, which will be updated every time the obsevable changes
fig = Figure(resolution = (960,450))
Phi_melt    =   1.0 .- Phi;     


    time=0.0
    ax1 = Axis(fig, aspect = 1, xlabel="width [km]", ylabel="Depth [km]", title=" $(round(time/kyr*100)/100) kyrs ")
    ax2 = Axis(fig, aspect = 1, xlabel="width [km]")
    cmap1 = :rainbow1       
    cmap1 = :vik   # better because perceptionally uniform
    cmap2 = :acton

    pltobj1 = heatmap!(ax1, x/1e3, z/1e3, T, colorrange = (0,1000), colormap = cmap1)
    contour!(ax1, x/1e3, z/1e3, T,levels = 125:125:1000,color = :black,linewidth = 0.85)
    pltobj2 = heatmap!(ax2, x/1e3, z/1e3, Phi_melt, colorrange = (0,1),colormap = cmap2)

    cbar1 = Colorbar(fig, pltobj1, label="Temperature", ticks=125:125:1000)        
    cbar2 = Colorbar(fig, pltobj2, label="Melt fraction")        
    
    o = Observable(1)
    on(o) do o
        Phi_melt    =   1.0 .- Phi;     
    #   Tcoord      =   hcat(Tracers.coord...)'; 

        ax1.title=" $(round(time/kyr*100)/100) kyrs "   # update title
        pltobj1 = heatmap!(ax1, x/1e3, z/1e3, T, colorrange = (0,1000), colormap = cmap1)
                contour!(ax1, x/1e3, z/1e3, T,levels = 100:100:1000,color = :black,linewidth = 0.85)
    #    pltobj2 = heatmap!(ax2, x/1e3, z/1e3, Phi_melt, colorrange = (0,1),colormap = cmap2)
    #              scatter!(Tcoord[:,1]/1e3,Tcoord[:,2]/1e3,color=:green, markersize = 1)  
    #    limits!(ax2, 0,25,-20,0)
        
        pltobj2 = plot(ax2, T[:,1], z/1e3)

        fig[1,1] = ax1
        fig[1,2] = cbar1
        fig[1,3] = ax2
        #fig[1,4] = cbar2
        display(fig)
        
        save("./viz2D_out/$(num+10000).png", fig)    # also save as separate figures 
    end

num=0;
=#

T_init = copy(T);
time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);

Makie.inline!(false)
fig = Figure(resolution = (1960,1450))
fig[1,1] = Axis(fig, xlabel = "Temperature [Celius]", ylabel = "Depth [km]", title = "temp")
lines!(fig[1,1],T[1,:],z/1e3)
display(fig)
Phi_melt    =   1.0 .- Phi;   
time=0.0

#record(fig, "sill_intrusion.gif") do io  # we will save the animation in the gif

    for it = 1:nt   # Time loop

        
        if floor(time/InjectionInterval)> dike_inj              # Add new dike every X years
            dike_inj            =   floor(time/InjectionInterval)        # Keeps track on what was injected already
            Angle_rand          =   0;
            dike                =   Dike(dike, Center=cen[:],Angle=[Angle_rand]);                # Specify dike with random location/angle but fixed size/T 
            Tracers, T, Vol     =   InjectDike(Tracers, T, Grid, dike, nTr_dike);       # Add dike, move hostrocks
            InjectVol           +=  Vol                                                 # Keep track of injected volume
            Qrate           = InjectVol/time
            Qrate_km3_yr    = Qrate*SecYear/(1e3^3)
            Qrate_km3_yr_km2    = Qrate_km3_yr/(pi*(W_in/2/1e3)^2)
            
            @printf "Added new dike; total injected magma volume = %.2f km³; rate Q=%.2f m³s⁻¹ = %.2e km³yr⁻¹ = %.2e km³yr⁻¹km⁻² \n" InjectVol/km³ Qrate Qrate_km3_yr Qrate_km3_yr_km2
            @show maximum(T)
        #   println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(Qrate,digits=2)) m³/s, Q=$(Qrate_km3_yr) km³/yr = $(Qrate_km3_yr_km2)km³/yr/km^2")
        end
        
        T_K         =  (T .+ 273.15).*K     # all GeoParams routines expect T in K
        compute_meltfraction!(Phi_melt, Mat_tup, Phases, P, ustrip.(T_K)) 

        Xs         =   1.0 .- Phi_melt;             # solid fraction
        dXs_dt     =   (Phi-Phi_o)./dt              # change of solid fraction with time

        # Update rho, cp and k for current P/T conditions
        compute_density!(Rho, Mat_tup, Phases, P,     ustrip.(T_K))
        compute_heatcapacity!(Cp, Mat_tup, Phases, P, ustrip.(T_K))
        compute_conductivity!(Kc, Mat_tup, Phases, P, ustrip.(T_K))
        
        # Perform a diffusion step

        #@parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kr, Kz, Rho, Cp, dt, dr, dz, La, dPhi_dt);  
        a=Mat_tup[1]
        if ~isempty(a.EnergySourceTerms)
            La =  NumValue(a.EnergySourceTerms[1].Q_L);             # latent heat 
        else
            La = 0.
        end
        @parallel diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, Kc, Kr, Kz, Rho, Cp, dt, dx, dz, La, dXs_dt) 

        
        @parallel (1:size(T,2)) bc2D_x!(Tnew);                  # set lateral boundary conditions (flux-free)

        if flux_free_bottom_BC==true
            @parallel (1:size(T,1)) bc2D_z_bottom!(Tnew);       # flux-free bottom BC                                                    
        else
            Tnew[:,1] = T[:,1]                                  # constant T bottom BC   
        end
                            
        # Fixed temperature top BC
        Tnew[:,end]         =   T[:,end];                       # fixed top temperature (constant)

        Tracers             =   UpdateTracers(Tracers, Grid, Tnew, Phi);                            # Update info on tracers 
        T, Tnew             =   Tnew, T;                                                            # Update temperature
        time                =   time + dt;                                                          # Keep track of evolved time
        Melt_Time[it]       =   sum( 1.0 .- Phi)/(Nx*Nz)                                            # Melt fraction in crust    
        Time_vec[it]        =   time;                                                               # Vector with time

        if mod(it,2000)==0  # Visualisation
           # @show it, size(T), size(z)
          #  num = num+1;

            #o[] = num;      # Creates the updated figure
            #on[] = num;
            
         #   lineplot.input_args[1] .= T[1,:]
            time_Myrs = time/Myr;

            # ---------------------------------
            # Create plot (using Makie)
            # 1D figure with cross-sections
            time_Myrs_rnd = round(time_Myrs,digits=3)
            fig = Figure(resolution = (1960,1450))
            Axis(fig[1,1], xlabel = "Temperature [Celcius]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs")
            lines!(fig[1,1],T[1,:],z/1e3,label="Center")
            lines!(fig[1,1],T[end,:],z/1e3,label="Side")
            lines!(fig[1,1],T_init[end,:],z/1e3,label="Initial")

            # 2D temperature plot:
            ax=Axis(fig[1, 2],xlabel = "Width [km]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs, Geneva model; Temperature [ᵒC]")
            limits!(ax, 0, 20, -20, 0)
            co = contourf!(fig[1, 2], x/1e3, z/1e3, T, levels = 0:50:1000,colormap = :jet)
            co1 = contour!(fig[1, 2], x/1e3, z/1e3, T, levels = 690:690)       # solidus
            Colorbar(fig[1, 3], co)
            
            display(fig)
            #
            # ---------------------------------

          #  current_figure()

            println(" Timestep $it = $(round(time/kyr*100)/100) kyrs, maxT = $(maximum(T)) C")
        end
    end
#end
#save("test.gif", stream) 

#gif(anim, "Example2D.gif", fps = 15)   # create gif animation
return x,z,T, Time_vec, Melt_Time;
end # end of main function





# Define material parameters for the simulation. 
# By using GeoParams we can easily change parameters without having to modify the main code
CharDim      = GEO_units(length=1000km, temperature=1000C, stress=10MPa, viscosity=1e20Pas);
MatParam     = Array{MaterialParams, 1}(undef, 2);
MatParam[1]  = SetMaterialParams(Name="Host Rock", Phase=1, 
                                 Density    = ConstantDensity(ρ=2700/m^3),
                          EnergySourceTerms = ConstantLatentHeat(Q_L=313/kg),
                               #Conductivity = ConstantConductivity(k=2.8Watt/K/m),     # in case we use constant k
                               Conductivity = T_Conductivity_Whittington_parameterised(),
                               HeatCapacity = ConstantHeatCapacity(cp=1000/kg/K),
                                    Melting = MeltingParam_4thOrder())

# It is unclear to me that thus is
MatParam[2]  = SetMaterialParams(Name="Melt", Phase=2, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                             EnergySourceTerms = ConstantLatentHeat(Q_L=313kJ/kg),
                                  Conductivity = T_Conductivity_Whittington_parameterised(),
                                  HeatCapacity = ConstantHeatCapacity(cp=1000/kg/K),
                                       Melting = MeltingParam_4thOrder())
Mat_tup = Tuple(MatParam)


x,z,T, Time_vec,Melt_Time = MainCode_2D(Mat_tup); # start the main code
#plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot
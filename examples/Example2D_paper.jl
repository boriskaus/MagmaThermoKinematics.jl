using MagmaThermoKinematics
using MagmaThermoKinematics.Diffusion2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using CairoMakie    # plotting
using GeoParams     # material parameters
using Printf        # print    
using MAT           # saves files in matlab format
using Parameters

using Statistics
using LinearAlgebra: norm

# Initialize 
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D

# These are useful parameters                                       
SecYear     = 3600*24*365.25
kyr         = 1000*SecYear
Myr         = 1e6*SecYear  
km³         = 1000^3

"""
    Holds numerical parameters for the overall simulation (and sets defaults), 
"""
@with_kw struct NumParam
    SimName::String = "Zassy_UCLA_ellipticalIntrusion"    # name of simulation
    FigTitle                    =   "UCLA setup"
    Nx::Int64                   =   201
    Nz::Int64                   =   201
    W::Float64                  =   20e3
    H::Float64                  =   20e3
    dx::Float64                 =   W/(Nx-1)
    dz::Float64                 =   H/(Nz-1)        # grid spacing in z
    Tsurface_Celcius::Float64   =   0               # Surface T in celcius
    Geotherm::Float64           =   40/1e3          # in K/m
    maxTime_Myrs::Float64       =   1.5             # maximum timestep 
    maxTime::Float64            =   maxTime_Myrs*SecYear*1e6 # maximum timestep  in seconds
    SaveOutput_steps::Int64     =   1e3;            # saves output every x steps 
    CreateFig_steps::Int64      =   500;            # Create a figure every X steps
    flux_free_bottom_BC::Bool   =   false           # zero flux bottom BC?
    deactivate_La_at_depth::Bool=   false           # deactivate latent heating @ the bottom of the model box?
    plot_tracers::Bool          =   true            # adds passive tracers to the plot
    advect_polygon::Bool        =   false           # adds a polygon around the intrusion area
    κ_time::Float64             =   3.3/(1000*2700) # κ to determine the stable timestep 
    fac_dt::Float64             =   0.4;            # prefactor with which dt is multiplied   
    dt::Float64                 =   fac_dt*min(dx^2, dz^2)./κ_time/4;   # timestep
    nt::Int64                   =   floor(maxTime/dt);
end

"""
    This holds various parameters related to the dike intrusion
"""
@with_kw struct DikeParam
    Type::String                    =   "CylindricalDike_TopAccretion"
    Center::Vector{Float64}         =   [0.; -7.0e3 - 0/2];     # Center of dike 
    T_in_Celsius::Float64           =   1000;                   # Temperature of injected magma  
    W_in::Float64                   =   20e3                    # Diameter of dike
    H_in::Float64                   =   74.6269                 # Thickness   
    AspectRatio::Float64            =   H_in/W_in;              # Aspect ratio                   
    SillRadius::Float64             =   W_in/2                  # Sill radius            
    SillArea::Float64               =   pi*SillRadius^2         # Horizontal area  of sill
    InjectionInterval_year::Float64 =   10e3;                   # Injection interval [years]
    InjectionInterval::Float64      =   InjectionInterval_year*SecYear;           # Injection interval [s]
    nTr_dike::Int64                 =   300                     # Number of tracers 
end

# This performs one diffusion timestep while doing nonlinear iterations (for latent heat and conductivity which depends on T)
function Nonlinear_Diffusion_step!(Tnew, T,  T_K, T_it_old, Mat_tup, Phi_melt, Phases, 
                P, R, Rc, qr, qz, Kc, Kr, Kz, Rho, Cp, La, dϕdT, Z, Num)
   
    err, iter = 1., 1
    @parallel assign!(T_K, T, 273.15)
    @parallel assign!(T_it_old, T)
    while err>1e-6 && iter<20
        
        # Update material properties (as some are a function of T)
        compute_meltfraction!(Phi_melt, Mat_tup, Phases, P, T_K) 
        compute_density!(Rho, Mat_tup, Phases, P,     T_K)
        compute_heatcapacity!(Cp, Mat_tup, Phases, P, T_K)
        compute_conductivity!(Kc, Mat_tup, Phases, P, T_K)
        compute_dϕdT!(dϕdT, Mat_tup, Phases, P,       T_K)

        # Switch off latent heat below a certain depth 
        if Num.deactivate_La_at_depth==true
            ind = findall(Z.<-15e3)
            dϕdT[ind] .= 0.0
        end
        
        # Diffusion step:
        @parallel diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, Kc, Kr, Kz, Rho, Cp, Num.dt, Num.dx, Num.dz, La, dϕdT) # diffusion step
        @parallel (1:size(T,2)) bc2D_x!(Tnew);                      # flux-free lateral boundary conditions
        if Num.flux_free_bottom_BC==true
            @parallel (1:size(T,1)) bc2D_z_bottom!(Tnew);           # flux-free bottom BC  (if false=isothermal)
        end
        
        # Update T_K (used above to compute material properties)
        @parallel assign!(T_K, Tnew,  273.15)   # all GeoParams routines expect T in K
        err     = norm(Tnew-T_it_old)           # compute error
    
        @parallel assign!(T_it_old, Tnew)       # Store Tnew of last iteration step
        iter   += 1
    end

    return nothing
end

#------------------------------------------------------------------------------------------
@views function MainCode_2D(Mat_tup, Num, Dikes);
    
    # Retrieve some parameters
    @unpack Nx,Nz, nt = Num 

    if ~isempty(Mat_tup[1].EnergySourceTerms)
        La =  NumValue(Mat_tup[1].EnergySourceTerms[1].Q_L);             # latent heat 
    else
        La = 0.
    end
    
    nInjections             =   Num.maxTime/Dikes.InjectionInterval   
    @show Num.dt/SecYear

    # Array initializations
    T                       =   @zeros(Nx,   Nz);   
    T_K                     =   @zeros(Nx,   Nz);   
    Kc                      =   @ones(Nx,    Nz);
    Rho                     =   @ones(Nx,    Nz);       
    Cp                      =   @ones(Nx,    Nz);
    Phi_melt, dϕdT,dϕdT_o   =   @zeros(Nx,   Nz), @zeros(Nx,   Nz  ), @zeros(Nx,   Nz  )                   # Melt fraction and derivative of melt fraction vs T

    # Work array initialization
    Tnew,T_init,qr,qz,Kr,Kz =   @zeros(Nx,   Nz), @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
    T_it_old                =   @zeros(Nx,   Nz)
    R,Rc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints

    # Set up model geometry & initial T structure
    x,z                     =   (0:Nx-1)*Num.dx, (-(Nz-1):0)*Num.dz;                    # 1-D coordinate arrays
    crd                     =   collect(Iterators.product(x,z))                         # Generate coordinates from 1D coordinate vectors   
    R,Z                     =   (x->x[1]).(crd), (x->x[2]).(crd);                       # Transfer coords to 3D arrays
    Rc                      =   (R[2:end,:] + R[1:end-1,:])/2 
    Grid                    =   (x,z);                                                  # Grid 
    Tracers                 =   StructArray{Tracer}(undef, 1)                           # Initialize tracers   
    dike                    =   Dike(W=Dikes.W_in,H=Dikes.H_in,Type=Dikes.Type,T=Dikes.T_in_Celsius, Center=Dikes.Center[:]);               # "Reference" dike with given thickness,radius and T
    T_init                 .=   Num.Tsurface_Celcius .- Z.*Num.Geotherm;                # Initial (linear) temperature profile

    # Set initial sill in temperature structure for Geneva type models
    dike_poly   = []
    if Dikes.Type  == "CylindricalDike_TopAccretion"
        ind = findall(R.<=Dikes.W_in/2 .&& abs.(Z.-Dikes.Center[2]) .< Dikes.H_in/2 );
        T_init[ind] .= Dikes.T_in_Celsius;
        if Num.advect_polygon==true
            dike_poly   =   CreateDikePolygon(dike);
        end
    end

    @parallel assign!(Tnew, T_init)
    @parallel assign!(T, T_init)

    P                       =   @zeros(Nx,Nz);
    Phases                  =   ones(Int64,Nx,Nz)
    time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);


    if isdir(Num.SimName)==false mkdir(Num.SimName) end;    # create simulation directory if needed
    global VEL
    for it = 1:nt   # Time loop

        # Add new dike every X years:
        if floor(time/Dikes.InjectionInterval)> dike_inj      
            dike_inj            =   floor(time/Dikes.InjectionInterval)                     # Keeps track on what was injected already
            dike                =   Dike(dike, Center=Dikes.Center[:],Angle=[0]);           # Specify dike with random location/angle but fixed size/T 
            Tracers, T,Vol,dike_poly  =   InjectDike(Tracers, T, Grid, dike, Dikes.nTr_dike, dike_poly=dike_poly);     # Add dike, move hostrocks
            InjectVol           +=  Vol                                                     # Keep track of injected volume
            Qrate               =   InjectVol/time
            Qrate_km3_yr        =   Qrate*SecYear/km³
            Qrate_km3_yr_km2    =   Qrate_km3_yr/(pi*(Dike_params.W_in/2/1e3)^2)
            @printf "  Added new dike; time=%.3f kyrs, total injected magma volume = %.2f km³; rate Q= %.2e km³yr⁻¹  \n" time/kyr InjectVol/km³ Qrate_km3_yr 
            
            if Num.advect_polygon==true && isempty(dike_poly)
                dike_poly   =   CreateDikePolygon(dike);            # create dike for the 1th time
            end
        end

        # Do a diffusion step, while taking T-dependencies into account
        Nonlinear_Diffusion_step!(Tnew, T,  T_K, T_it_old, Mat_tup, Phi_melt, Phases, P, R, Rc, qr, qz, Kc, Kr, Kz, Rho, Cp, La, dϕdT, Z, Num)
        
        # Update variables
        Tracers             =   UpdateTracers(Tracers, Grid, Tnew, Phi_melt);      # Update info on tracers 
        @parallel assign!(T, Tnew)
        @parallel assign!(Tnew, T)
        time                =   time + Num.dt;                                     # Keep track of evolved time
        Melt_Time[it]       =   sum( Phi_melt)/(Nx*Nz)                             # Melt fraction in crust    
        Time_vec[it]        =   time;                                              # Vector with time
        
        # Visualize results
        if mod(it,Num.CreateFig_steps)==0  || it==nt 
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
            limits!(ax1, 0, 1100, -20, 0)
        
            # 2D temperature plot 
            #ax2=Axis(fig[1, 2],xlabel = "Width [km]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs, Geneva model; Temperature [ᵒC]")
            ax2=Axis(fig[1, 2],xlabel = "Width [km]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs, $(Num.FigTitle); Temperature [ᵒC]")
            
            co = contourf!(fig[1, 2], x/1e3, z/1e3, T, levels = 0:50:1050,colormap = :jet)
        
            if maximum(T)>691
            co1 = contour!(fig[1, 2], x/1e3, z/1e3, T, levels = 690:691)       # solidus
            end
            limits!(ax2, 0, 20, -20, 0)
            Colorbar(fig[1, 3], co)
            
            # 2D melt fraction plots:
            ax3=Axis(fig[1, 4],xlabel = "Width [km]", ylabel = "Depth [km]", title = " ϕ (melt fraction)")
            co = heatmap!(fig[1, 4], x/1e3, z/1e3, Phi_melt, colormap = :vik, colorrange=(0, 1))
            if Num.plot_tracers==true
                # Add tracers to plot
                scatter!(fig[1, 4],getindex.(Tracers.coord,1)/1e3, getindex.(Tracers.coord,2)/1e3, color=:white)
            end
            if Num.advect_polygon==true 
                # Add polygon
                pl = lines!(fig[1, 4], dike_poly[1]/1e3, dike_poly[2]/1e3,   color = :yellow, linestyle=:dot, linewidth=1.0)
            end
            limits!(ax3, 0, 20, -20, 0)
            Colorbar(fig[1, 5], co)

            # Save figure
            save("$(Num.SimName)/$(Num.SimName)_$it.png", fig)
            #
            # ---------------------------------
            println("Timestep $it = $(round(time/kyr*100)/100) kyrs, maxT = $(round(maximum(T),digits=3)) ᵒC")
        end

        # Save output to disk once in a while
        if mod(it,Num.SaveOutput_steps)==0  || it==nt 
            filename = "$(Num.SimName)/$(Num.SimName)_$it.mat"
            matwrite(filename, 
                            Dict("Tnew"=> Tnew, 
                                "time"=> time, 
                                "x"   => Vector(x), 
                                "z"   => Vector(z))
                )
            println("  Saved matlab output to $filename")    
        end

    end

    return x,z,T, Time_vec, Melt_Time, Tracers, VEL;
end # end of main function


# Define material parameters for the simulation. 

if 1==0
    # 1D, Geneva-type models without magmatic injections (for comparison with 1D code)
    Num         = NumParam(Nx=21, Nz=269, SimName="Zassy_Geneva_zeroFlux_1D_variablek_1", 
                            flux_free_bottom_BC=true, deactivate_La_at_depth=false, 
                            SaveOutput_steps=1e4, CreateFig_steps=1000, FigTitle = "Geneva setup");
    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", InjectionInterval = 1e40, W_in=40e3)

    MatParam    = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700/m^3),
                             EnergySourceTerms = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),     # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = MeltingParam_4thOrder()),                     # Marxer & Ulmer data
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end

if 1==0
    # 2D, Geneva-type models  
    Num         = NumParam(Nx=269, Nz=269, SimName="Zassy_Geneva_zeroFlux_variable_k_1", 
                            flux_free_bottom_BC=true, deactivate_La_at_depth=true, 
                            SaveOutput_steps=1e4, CreateFig_steps=1000, plot_tracers=false, advect_polygon=false,
                            FigTitle="Geneva Models");

    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", InjectionInterval_year = 10e3, 
                            W_in=20e3, H_in=74.6269)

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700/m^3),
                             EnergySourceTerms = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = MeltingParam_4thOrder()),                     # Marxer & Ulmer data
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end


if 1==1
    # 2D, UCLA-type models (WiP)
    Num          = NumParam(Nx=101, Nz=101, SimName="Zassy_UCLA_ellipticalIntrusion", 
                            SaveOutput_steps=1e4, CreateFig_steps=100,
                            maxTime_Myrs=1.13,
                            FigTitle="UCLA Models", plot_tracers=false, advect_polygon=true);

    #                             
    Flux         = 7.5e-6;                              # in km3/km2/a 
    Total_r_km   = 10;                                  # final radius of area
    V_inject_km3 = 10;                                  # injection volume per sill injection
    
    Total_A_km2  = pi*Total_r_km^2;                     # final area in km^2
    Flux_km3_a   = Flux*Total_A_km2;                    # flux in km3/year

    V_total_km3  = Flux_km3_a*Num.maxTime_Myrs*1e6;
    h_total_km   = V_total_km3/(Total_A_km2)            # final height [km]
    
    mid_depth_km = -6.5-h_total_km/2;                   # mid depth of injection area [km]
    r_h          = Total_r_km/h_total_km;               # aspect ratio of spheroid [] (well strictly speaking not, but following the Excel spreadsheet)

    V_final_a    = (3/2*V_total_km3*r_h/pi)^(1/3)         # a axis in km
    V_final_b    = V_final_a/r_h*0.5;                     # b axis in km                        
    V_fin_check  = 4/3*pi*V_final_a^2*V_final_b;          # final area in km3 (just checking, should be == V_total_km3 )

    V_inj_a      = (3/2*V_inject_km3*r_h/pi)^(1/3)        # a axis in km of injected ellipsoid
    V_inj_b      = V_inj_a/r_h*0.5;                    # b axis in km                        
    V_inj_check  = 4/3*pi*V_inj_a^2*V_inj_b;              # checking

    nInjections     =   V_total_km3/V_inject_km3                # the number of required injections
    InjectionInterval_yr = Num.maxTime_Myrs*1e6/nInjections;    # Time inbetween injections

    # Use the parameters. Note that we specify the diameter of the ellipse in here
    Dike_params  = DikeParam(Type="ElipticalIntrusion", InjectionInterval_year = InjectionInterval_yr, 
                            W_in=V_inj_a*2*1e3, 
                            H_in=V_inj_b*2*1e3, 
                            Center=[0, mid_depth_km*1e3])

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700/m^3),
                             EnergySourceTerms = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                 #Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                  Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = MeltingParam_4thOrder()),                     # Marxer & Ulmer data
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end

# Call the main code with the specified material parameters
x,z,T, Time_vec,Melt_Time, Tracers, VEL = MainCode_2D(MatParam, Num, Dike_params); # start the main code


#plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot
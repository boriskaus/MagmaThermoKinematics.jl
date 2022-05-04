# This example reproduces the cases shown in the ZASSy manuscript, which we are currently preparing.
#  It includes comparisons with 2D simulations done by the Geneva (Gregor Weber, Luca Caricchi) & UCLA (Oscar Lovera) Tracers_SimParams
#
# 
const USE_GPU=false;
using MagmaThermoKinematics
if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
end
using MagmaThermoKinematics.Diffusion2D # to load AFTER calling environment!()

using CairoMakie    # plotting
using Printf        # pretty print    
using MAT, JLD2     # saves files in matlab format & JLD2 (hdf5) format

using TimerOutputs

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
    flux_bottom_BC::Bool        =   false           # flux bottom BC?
    flux_bottom::Float64        =   167e-3          # Flux in W/m2 in case flux_bottom_BC=true
    deactivate_La_at_depth::Bool=   false           # deactivate latent heating @ the bottom of the model box?
    deactivationDepth::Float64  =   -15e3           # deactivation depth
    plot_tracers::Bool          =   true            # adds passive tracers to the plot
    advect_polygon::Bool        =   false           # adds a polygon around the intrusion area
    axisymmetric::Bool          =   true            # axisymmetric (if true) of 2D geometry?
    κ_time::Float64             =   3.3/(1000*2700) # κ to determine the stable timestep 
    fac_dt::Float64             =   0.4;            # prefactor with which dt is multiplied   
    dt::Float64                 =   fac_dt*min(dx^2, dz^2)./κ_time/4;   # timestep
    nt::Int64                   =   floor(maxTime/dt);
    ω::Float64                  =   0.8;            # relaxation parameter for nonlinear iterations    
    max_iter::Int64             =   5000;           # max. number of nonlinear iterations        
    verbose::Bool               =   false;    
    convergence::Float64        =   1e-5;           # nonlinear convergence criteria      
    AnalyticalInitialGeo::Bool  =   false;      
    qs_anal::Float64            =   170e-3;
    qm_anal::Float64            =   167e-3;
    hr_anal::Float64            =   10e3;
    k_anal::Float64             =   3.35;
    InitialEllipse::Bool        =   false;
    a_init::Float64             =   2.5e3;
    b_init::Float64             =   1.5e3;
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

#------------------------------------------------------------------------------------------
@views function MainCode_2D(Mat_tup, Num, Dikes);
    
    # Array & grid initializations ---------------
    Arrays = CreateArrays(Dict( (Num.Nx,  Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0,dϕdT_o=0, R=0, Z=0, P=0),
                                (Num.Nx-1,Num.Nz  )=>(qx=0,Kx=0, Rc=0), 
                                (Num.Nx  ,Num.Nz-1)=>(qz=0,Kz=0 )
                                ))

    # Set up model geometry & initial T structure
    Grid    = CreateGrid(size=(Num.Nx,Num.Nz), extent=(Num.W, Num.H))
    
    @parallel (1:Num.Nx, 1:Num.Nz) GridArray!(Arrays.R,  Arrays.Z, Grid.coord1D[1], Grid.coord1D[2])   
    Arrays.Rc              .=   (Arrays.R[2:end,:] + Arrays.R[1:end-1,:])/2         # center points in x
    # --------------------------------------------
    
    Tracers                 =   StructArray{Tracer}(undef, 1)                       # Initialize tracers   
    dike                    =   Dike(W=Dikes.W_in,H=Dikes.H_in,Type=Dikes.Type,T=Dikes.T_in_Celsius, Center=Dikes.Center[:]);               # "Reference" dike with given thickness,radius and T

    # Set initial geotherm -----------------------
    if Num.AnalyticalInitialGeo
        # Turcotte & Schubert  analytical geotherm which takes depth-dependent radioactive heating into account
        # This is used in the UCLA setup. Parameters in Mat_tup should be consistent with this (we don't check for that)
        Arrays.T_init      .=  @. Num.Tsurface_Celcius - (Num.qm_anal/Num.k_anal)*Arrays.Z + (Num.qs_anal-Num.qm_anal)*Num.hr_anal/Num.k_anal*( 1.0 - exp(Arrays.Z/Num.hr_anal)) 

        Geothermalgradient_K_km = (maximum(Arrays.T_init) - minimum(Arrays.T_init))/(maximum(Arrays.Z) - minimum(Arrays.Z))*1e3
        # check that this is selected
        H0  = (Num.qs_anal-Num.qm_anal)/Num.hr_anal
        H0_num = Value(Mat_tup[1].RadioactiveHeat[1].H_0) 
        println("Employing analytical initial geotherm that takes radioactive heating into account with H0=$(H0), H0_num=$(H0_num)")
        println(" This results in an effective geothermal gradient of $(Geothermalgradient_K_km) K/km")

    else
        Arrays.T_init      .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm;                # Initial (linear) temperature profile
    end
    # --------------------------------------------
    
    # Update buffer & phases arrays --------------
    if USE_GPU
        # CPU buffers for advection
        Tnew_cpu        =   Matrix{Float64}(undef, Num.Nx, Num.Nz)
        Phi_melt_cpu    =   similar(Tnew_cpu)
        Phases          =   CUDA.ones(Int64,Num.Nx,Num.Nz)
    else
        Tnew_cpu        =   similar(Arrays.T)
        Phi_melt_cpu    =   similar(Arrays.ϕ)
        Phases          =   @ones(Num.Nx,Num.Nz)
    end
    # --------------------------------------------
        
    # Optionally set initial sill in models ------
    dike_poly   = []
    if Dikes.Type  == "CylindricalDike_TopAccretion"
        ind = findall( (Arrays.R.<=Dikes.W_in/2) .& (abs.(Arrays.Z.-Dikes.Center[2]) .< Dikes.H_in/2) );
        Arrays.T_init[ind] .= Dikes.T_in_Celsius;
        if Num.advect_polygon==true
            dike_poly   =   CreateDikePolygon(dike);
        end
    end
    if Num.InitialEllipse
        ind =  findall( ((Arrays.R.^2.0)/(Num.a_init^2.0) .+ ((Arrays.Z.-Dikes.Center[2]).^2.0)/((Num.b_init)^2.0)) .< 1.0); # ellipse
        Arrays.T_init[ind] .= Dikes.T_in_Celsius;
        dike_poly   =   CreateDikePolygon(Dike(dike,W=Num.a_init*2, H=Num.b_init*2));
    end
    # --------------------------------------------
    
    # Initialise arrays --------------------------
    @parallel assign!(Arrays.Tnew, Arrays.T_init)
    @parallel assign!(Arrays.T, Arrays.T_init)
    time, dike_inj, InjectVol, Time_vec,Melt_Time,Tav_magma_Time = 0.0, 0.0, 0.0,zeros(Num.nt,1),zeros(Num.nt,1),zeros(Num.nt,1);

    if isdir(Num.SimName)==false mkdir(Num.SimName) end;    # create simulation directory if needed
    # --------------------------------------------

    for it = 1:Num.nt   # Time loop

        # Add new dike every X years -----------------
        if floor(time/Dikes.InjectionInterval)> dike_inj      
            dike_inj            =   floor(time/Dikes.InjectionInterval)                     # Keeps track on what was injected already
            dike                =   Dike(dike, Center=Dikes.Center[:],Angle=[0]);           # Specify dike with random location/angle but fixed size/T 
            Tnew_cpu           .=   Array(Arrays.T)
            @timeit to "Dike intrusion" Tracers, Tnew_cpu,Vol,dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, Dikes.nTr_dike, dike_poly=dike_poly);     # Add dike, move hostrocks
            Arrays.T           .=   Data.Array(Tnew_cpu)
            InjectVol          +=   Vol                                                     # Keep track of injected volume
            Qrate               =   InjectVol/time
            Qrate_km3_yr        =   Qrate*SecYear/km³
            Qrate_km3_yr_km2    =   Qrate_km3_yr/(pi*(Dike_params.W_in/2/1e3)^2)

            @printf "  Added new dike; time=%.3f kyrs, total injected magma volume = %.2f km³; rate Q= %.2e km³yr⁻¹  \n" time/kyr InjectVol/km³ Qrate_km3_yr 
            
            if Num.advect_polygon==true && isempty(dike_poly)
                dike_poly   =   CreateDikePolygon(dike);            # create dike for the 1th time
            end
        end
        # --------------------------------------------

        # Do a diffusion step, while taking T-dependencies into account
        @timeit to "Diffusion solver" Nonlinear_Diffusion_step_2D!(Arrays, Mat_tup, Phases, Grid, Num.dt, Num)
        # --------------------------------------------

        # Update variables ---------------------------
        # copy to cpu
        Tnew_cpu      .= Array(Arrays.Tnew)
        Phi_melt_cpu  .= Array(Arrays.ϕ)
        
        @timeit to "Update tracers"  UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers 

        # copy back to gpu
        Arrays.Tnew          .= Data.Array(Tnew_cpu)
        Arrays.ϕ      .= Data.Array(Phi_melt_cpu)

        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)
        time                =   time + Num.dt;                                     # Keep track of evolved time
        Melt_Time[it]       =   sum( Arrays.ϕ)/(Num.Nx*Num.Nz)              # Melt fraction in crust    
        
        ind = findall(Arrays.T.>700);          
        if ~isempty(ind)
            Tav_magma_Time[it] = sum(Arrays.T[ind])/length(ind)                            # average T of part with magma
        else
            Tav_magma_Time[it] = NaN;
        end

        Time_vec[it]        =   time;                                              # Vector with time

        if mod(it,10)==0
            update_Tvec!(Tracers, time/SecYear*1e-6)                                 # update T & time vectors on tracers
        end
        # --------------------------------------------

        # Visualize results --------------------------
        if mod(it,Num.CreateFig_steps)==0  || it==Num.nt 
            @timeit to "visualisation" begin
                time_Myrs = time/Myr;

            T_plot = Array(Arrays.T)
            T_init_plot = Array(Arrays.T_init)
            Phi_melt_plot = Array(Arrays.ϕ)
            # ---------------------------------
            # Create plot (using Makie)
            fig = Figure(resolution = (2000,1000))

            # 1D figure with cross-sections
            time_Myrs_rnd = round(time_Myrs,digits=3)
            ax1 = Axis(fig[1,1], xlabel = "Temperature [ᵒC]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs")
            lines!(fig[1,1],T_plot[1,:],Grid.coord1D[2]/1e3,label="Center")
            lines!(fig[1,1],T_plot[end,:],Grid.coord1D[2]/1e3,label="Side")
            lines!(fig[1,1],T_init_plot[end,:],Grid.coord1D[2]/1e3,label="Initial")
            axislegend(ax1)
            limits!(ax1, 0, 1100, -20, 0)
        
            # 2D temperature plot 
            #ax2=Axis(fig[1, 2],xlabel = "Width [km]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs, Geneva model; Temperature [ᵒC]")
            ax2=Axis(fig[1, 2],xlabel = "Width [km]", ylabel = "Depth [km]", title = "Time= $time_Myrs_rnd Myrs, $(Num.FigTitle); Temperature [ᵒC]")
            
            co = contourf!(fig[1, 2], Grid.coord1D[1]/1e3, Grid.coord1D[2]/1e3, T_plot, levels = 0:50:1050,colormap = :jet)
        
            if maximum(Arrays.T)>691
            co1 = contour!(fig[1, 2], Grid.coord1D[1]/1e3, Grid.coord1D[2]/1e3, T_plot, levels = 690:691)       # solidus
            end
            limits!(ax2, 0, 20, -20, 0)
            Colorbar(fig[1, 3], co)
            
            # 2D melt fraction plots:
            ax3=Axis(fig[1, 4],xlabel = "Width [km]", ylabel = "Depth [km]", title = " ϕ (melt fraction)")
            co = heatmap!(fig[1, 4], Grid.coord1D[1]/1e3, Grid.coord1D[2]/1e3, Phi_melt_plot, colormap = :vik, colorrange=(0, 1))
            if Num.plot_tracers==true & !isempty(Tracers)
                # Add tracers to plot
                scatter!(fig[1, 4],getindex.(Tracers.coord,1)/1e3, getindex.(Tracers.coord,2)/1e3, color=:white)
            end
            if (Num.advect_polygon==true) & (~isempty(dike_poly)) & (~isempty(Tracers)) 
                # Add polygon
                pl = lines!(fig[1, 4], dike_poly[1]/1e3, dike_poly[2]/1e3,   color = :yellow, linestyle=:dot, linewidth=1.5)
            end
            limits!(ax3, 0, 20, -20, 0)
            Colorbar(fig[1, 5], co)

            # Save figure
            save("$(Num.SimName)/$(Num.SimName)_$it.png", fig)
            #
            # ---------------------------------

            # compute average T of molten zone (1% melt)
            ind = findall(Arrays.ϕ.>0.01);
            if ~isempty(ind)
                T_av_melt = round(sum(Arrays.T[ind])/length(ind), digits=3)
            else
                T_av_melt = NaN;
            end
            # print results:  
            println("Timestep $it = $(round(time/kyr*100)/100) kyrs, max(T)=$(round(maximum(Arrays.T),digits=3))ᵒC, Taverage_magma=$(T_av_melt)ᵒC, max(ϕ)=$(round(maximum(Arrays.ϕ),digits=2))")
            end
        end
        # --------------------------------------------

        # Save output to disk once in a while --------
        if mod(it,Num.SaveOutput_steps)==0  || it==Num.nt 
            filename = "$(Num.SimName)/$(Num.SimName)_$it.mat"
            matwrite(filename, 
                            Dict("Tnew"=> Array(Arrays.Tnew), 
                                "time"=> time, 
                                "x"   => Vector(Grid.coord1D[1]), 
                                "z"   => Vector(Grid.coord1D[2]),
                                "dike_poly" => dike_poly)
                )
            println("  Saved matlab output to $filename")    

            if it==Num.nt   
                # save tracers & material parameters of the simulation in jld2 format so we can reproduce this
                filename = "$(Num.SimName)/Tracers_SimParams.jld2"
                jldsave(filename; Tracers, Dikes, Num, Mat_tup, Time_vec, Melt_Time, Tav_magma_Time)
                println("  Saved Tracers & simulation parameters to file $filename ")    

                filename = "$(Num.SimName)/Tracers.mat"
                matwrite(filename,  Dict("Tracers"=> Tracers))
                println("  Saved Tracers to file $filename ")    
            end

        end
        # --------------------------------------------

    end
    x,z = Grid.coord1D[1],Grid.coord1D[2]
    return x,z,Arrays.T, Time_vec, Melt_Time, Tracers, dike_poly;
end # end of main function


# Define material parameters for the simulation. 

if 1==0
    # 1D test, Geneva-type models without magmatic injections (for comparison with 1D code)
    Num         = NumParam(Nx=21, Nz=269, SimName="Zassy_Geneva_zeroFlux_1D_variablek_2", 
                            flux_bottom_BC=true, flux_bottom=0, deactivate_La_at_depth=false, 
                            SaveOutput_steps=1e4, CreateFig_steps=1000, FigTitle = "Geneva setup", plot_tracers=false);

    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", InjectionInterval = 1e40, W_in=40e3)

    MatParam    = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),     # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = MeltingParam_4thOrder()),                     # Marxer & Ulmer data
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end

if 1==0
    # 2D, Geneva-type models with Greg's parameters 
    Num         = NumParam(Nx=269, Nz=269, SimName="ZASSy_Geneva_zeroFlux_variable_k_4thordermelt", 
                            maxTime_Myrs=1.5,
                            flux_bottom_BC=true, flux_bottom=0, deactivate_La_at_depth=true, 
                            SaveOutput_steps=1e4, CreateFig_steps=1000, plot_tracers=false, advect_polygon=true,
                            FigTitle="Geneva Models");

    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", InjectionInterval_year = 10e3, 
                            W_in=20e3, H_in=74.6269)

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = MeltingParam_4thOrder()),                     # Marxer & Ulmer data
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end

if 1==0

    # 2D, run 02.2 Geneva-type models with Greg's original parameters 
    Num         = NumParam(Nx=269, Nz=269, SimName="ZASSy_Geneva_zeroFlux_variable_k_run02.2_withlatent_depth", 
                            maxTime_Myrs=1.5, fac_dt=0.05, ω=0.9, verbose=false,
                            flux_bottom_BC=true, flux_bottom=0, deactivate_La_at_depth=false, 
                            SaveOutput_steps=4000, CreateFig_steps=1000, plot_tracers=false, advect_polygon=true,
                            FigTitle="Geneva Models");

    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", InjectionInterval_year = 10e3, 
                            W_in=20e3, H_in=74.6269)

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = MeltingParam_4thOrder()),                     # Marxer & Ulmer data
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end

if 1==0

    # 2D, run 02.2 Geneva-type models with Greg's smooth melting parameterisation
    Num         = NumParam(Nx=269, Nz=269, SimName="ZASSy_Geneva_zeroFlux_variable_k_run02.2_withlatent_depth_smooth_dt0_4_v2", 
                            maxTime_Myrs=1.5, fac_dt=0.05, ω=0.5, verbose=false,
                            flux_bottom_BC=true, flux_bottom=0, deactivate_La_at_depth=false, 
                            SaveOutput_steps=4000, CreateFig_steps=1000, plot_tracers=false, advect_polygon=true,
                            FigTitle="Geneva Models");

    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", InjectionInterval_year = 10e3, 
                            W_in=20e3, H_in=74.6269)

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = SmoothMelting(MeltingParam_4thOrder())),                     # Marxer & Ulmer data
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end

if 1==0
    # 2D, Geneva-type models with Caricchi parameters (as described in ZASSy paper)
    Num         = NumParam(Nx=269, Nz=269, SimName="Zassy_Geneva_zeroFlux_variable_k_CaricchiMelting", 
                            maxTime_Myrs=1.5,
                            flux_bottom_BC=true, flux_bottom=0, deactivate_La_at_depth=true, 
                            SaveOutput_steps=1e4, CreateFig_steps=1000, plot_tracers=false, advect_polygon=true,
                            FigTitle="Geneva Models");

    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", InjectionInterval_year = 10e3, 
                            W_in=20e3, H_in=74.6269)

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = MeltingParam_Caricchi()),                     # Caricchi melting
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
end

if 1==1
   # 2D, UCLA-type models
    #
    # Benchmark case with Oscar Lovera, who uses an exponential depth-dependent radioactive heating term combined with flux lower BC
    #
    #  For the parameters he gave, k=3.35 W/mK  qs=170 mW/m2    qm=167mW/m2  hr=10e3m
    #  H0  = (qs-qm)/hr= 3.0000e-07

    Num          = NumParam(Nx=301, Nz=201, W=30e3, SimName="ZASSy_UCLA_ellipticalIntrusion_constant_k_radioactiveheating_smoothQuad_initialEllipse_2", 
                            SaveOutput_steps=2000, CreateFig_steps=1000, axisymmetric=false,
                            flux_bottom_BC=true, flux_bottom=167e-3, fac_dt=0.2, ω=0.6, verbose=false,
                            maxTime_Myrs=0.7, 
                            AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=170e-3, qm_anal=167e-3, hr_anal=10e3, k_anal=3.3453,
                            InitialEllipse =   true, a_init= 2.5e3,  b_init  =   1.5e3,
                            FigTitle="UCLA Models", plot_tracers=false, advect_polygon=true);                            
                                 
    mid_depth_km = -7.0;                   # mid depth of injection area [km]
    
    V_inj_a      = 1.29135;                                 # a axis in km of injected ellipsoid
    V_inj_b      = 0.7745;                                  # b axis in km                        
    V_inj_check  = 4/3*pi*V_inj_a^2*V_inj_b;                # checking

    InjectionInterval_yr = 5000;

    # Use the parameters. Note that we specify the diameter of the ellipse in here
    Dike_params  = DikeParam(Type="EllipticalIntrusion", InjectionInterval_year = InjectionInterval_yr, 
                            W_in=V_inj_a*2*1e3, 
                            H_in=V_inj_b*2*1e3, 
                            Center=[0, mid_depth_km*1e3])

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=3345.3kg/m^3),                    # used in the parameterisation of Whittington 
                                    LatentHeat = ConstantLatentHeat(Q_L=2.67e5J/kg),
                               RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=3e-7Watt/m^3),
                                  Conductivity = ConstantConductivity(k=3.3453Watt/K/m),            # in case we use constant k 
                                  HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                               # Conductivity = T_Conductivity_Whittington(),                       # T-dependent k
                                  #HeatCapacity = T_HeatCapacity_Whittington(),                                                       # T-dependent cp
                                       Melting = SmoothMelting(MeltingParam_Quadratic(T_s=(700+273.15)K, T_l=(1100+273.15)K))),       # Quadratic parameterization as in Tierney et al.
                    )
end


if 1==0
    # 2D, UCLA-type models

    # In the UCLA model, the following geotherm is used, with Tsurface as top BC & qm (mantle heatflux) as bottom BC.
    # Exponential depth-dependent radioactive heating (in W/m3) is assumed to follow:
    #  H_r = H0 * exp(-z/h)r)
    # with hr=10km & qs,qm are manipulated to obtain an approximately linear initial geotherm over the first 20 km:
    #
    # The analytical solution for this case, assuming constant k, comes from Turcotte & Schubert:  
    #  T = Tsurface + (qm/k)*z + (qs-qm)hr/k *(1.-exp(-z/hr)) 
    #
    # which can also be written as:
    #  T = Tsurface + (qm/k)*z + H0*hr^2/k *(1.-exp(-z/hr)) 
    #  so: H0  = (qs-qm)/hr
    #
    # If we manipulate this, we obtain a good fit with qm=76e-3, qs=86e-3, H0=1e-6, k=1.9 (~value that Whittington gives for 1000C)

    #Num          = NumParam(Nx=301, Nz=201, W=30e3, SimName="Zassy_UCLA_ellipticalIntrusion_variable_k_radioactiveheating", 
    #                        SaveOutput_steps=400, CreateFig_steps=100, axisymmetric=true,
    #                        flux_bottom_BC=true, flux_bottom=40/1e3*1.9,
    #                        maxTime_Myrs=1.13, Tsurface_Celcius=25, Geotherm=(801.12-25)/20e3,
    #                        FigTitle="UCLA Models", plot_tracers=false, advect_polygon=true);

    Num          = NumParam(Nx=301, Nz=201, W=30e3, SimName="Zassy_UCLA_ellipticalIntrusion_variable_k_radioactiveheating_1", 
                            SaveOutput_steps=2000, CreateFig_steps=1000, axisymmetric=false,
                            flux_bottom_BC=true, flux_bottom=38.7/1e3*3.35, fac_dt=0.2, ω=0.6, max_iter=100, verbose=false,
                            maxTime_Myrs=1.13, Tsurface_Celcius=25, Geotherm=(801.12-25)/20e3,
                            FigTitle="UCLA Models", plot_tracers=false, advect_polygon=true);                            
                                 
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
    V_inj_b      = V_inj_a/r_h*0.5;                       # b axis in km                        
    V_inj_check  = 4/3*pi*V_inj_a^2*V_inj_b;              # checking

    nInjections     =   V_total_km3/V_inject_km3                # the number of required injections
    InjectionInterval_yr = Num.maxTime_Myrs*1e6/nInjections;    # Time inbetween injections


    # Use the parameters. Note that we specify the diameter of the ellipse in here
    Dike_params  = DikeParam(Type="EllipticalIntrusion", InjectionInterval_year = InjectionInterval_yr, 
                            W_in=V_inj_a*2*1e3, 
                            H_in=V_inj_b*2*1e3, 
                            Center=[0, mid_depth_km*1e3])

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),                # used in the parameterisation of Whittington 
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                               RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=1e-6Watt/m^3),
                                  Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  #Conductivity = ConstantConductivity(k=3.35Watt/K/m),        # in case we use constant k
                                  HeatCapacity = T_HeatCapacity_Whittington(),                 # T-dependent cp
                                 # HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                       Melting = SmoothMelting(MeltingParam_Quadratic())),                    # Quadratic parameterization as in Tierney et al.
                    )
end

# Keep track of time
const to = TimerOutput()
reset_timer!(to)

# Call the main code with the specified material parameters
x,z,T, Time_vec,Melt_Time, Tracers, dike_poly = MainCode_2D(MatParam, Num, Dike_params); # start the main code

@show(to)

#plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot
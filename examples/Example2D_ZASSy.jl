# This example reproduces the cases shown in the ZASSy manuscript, which we are currently preparing.
#  It includes comparisons with 2D simulations done by the Geneva (Gregor Weber, Luca Caricchi) & UCLA (Oscar Lovera) Tracers_SimParams
#
#
const USE_GPU=true;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end
using ParallelStencil, ParallelStencil.FiniteDifferences2D

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
    @init_parallel_stencil(CUDA, Float64, 2)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
    @init_parallel_stencil(Threads, Float64, 2)
end
using MagmaThermoKinematics.Diffusion2D # to load AFTER calling environment!()
using GeophysicalModelGenerator, GeoParams
using MagmaThermoKinematics.Fields2D

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
    TrackTracersOnGrid::Bool    =   true;
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

"""
    Analytical geotherm used for the UCLA setups, which includes radioactive heating
"""
function AnalyticalGeotherm!(T, Z, Tsurf, qm, qs, k, hr)

    T      .=  @. Tsurf - (qm/k)*Z + (qs-qm)*hr/k*( 1.0 - exp(Z/hr))

    return nothing
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
    GridArray!(Arrays.R, Arrays.Z, Grid)
    Arrays.Rc              .=   (Arrays.R[2:end,:] + Arrays.R[1:end-1,:])/2         # center points in x
    Rc_CPU                  = Array(Arrays.Rc);                                 # on CPU
    # --------------------------------------------

    println("Timestep Δt= $(Num.dt/SecYear) ")

    Tracers                 =   StructArray{Tracer}(undef, 1)                       # Initialize tracers
    dike                    =   Dike(W=Dikes.W_in,H=Dikes.H_in,Type=Dikes.Type,T=Dikes.T_in_Celsius, Center=Dikes.Center[:]);               # "Reference" dike with given thickness,radius and T

    # Set initial geotherm -----------------------
    if Num.AnalyticalInitialGeo
        # Turcotte & Schubert  analytical geotherm which takes depth-dependent radioactive heating into account
        # This is used in the UCLA setup. Parameters in Mat_tup should be consistent with this (we don't check for that)
        Arrays.T_init      .=  @. Num.Tsurface_Celcius - (Num.qm_anal/Num.k_anal)*Arrays.Z + (Num.qs_anal-Num.qm_anal)*Num.hr_anal/Num.k_anal*( 1.0 - exp(Arrays.Z/Num.hr_anal))
        #AnalyticalGeotherm!(Arrays.T_init, Arrays.Z, Num.Tsurface_Celcius, Num.qm_anal, Num.qs_anal, Num.k_anal, Num.hr_anal)

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
        Phases          =   ones(Int64,Num.Nx,Num.Nz)
    end
    # --------------------------------------------

    # Optionally set initial sill in models ------
    InjectVol = 0.0;
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
        #dike_poly   =   CreateDikePolygon(Dike(dike,W=Num.a_init*2, H=Num.b_init*2));

        #InjectVol += 4/3*pi*(Num.a_init)^2*Num.b_init;


        # Inject initial dike to the tracers
        dike_initial        =   Dike(dike, Center=Dikes.Center[:],Angle=[0],W=Num.a_init*2, H=Num.b_init*2);           # Specify dike with random location/angle but fixed size/T
        Tnew_cpu           .=   Array(Arrays.T)
        @timeit to "Dike intrusion" Tracers, Tnew_cpu,Vol,dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike_initial, Dikes.nTr_dike, dike_poly=dike_poly);     # Add dike, move hostrocks

        Arrays.T           .=   Data.Array(Tnew_cpu)
        InjectVol          +=   Vol                                                     # Keep track of injected volume
        if Num.advect_polygon==true && isempty(dike_poly)
            dike_poly   =   CreateDikePolygon(dike_initial);            # create dike for the 1th time
        end
        @printf "  Added initial dike; total injected magma volume = %.2f km³ \n"  InjectVol/km³


    end
    # --------------------------------------------

    # Initialise arrays --------------------------
    @parallel assign!(Arrays.Tnew, Arrays.T_init)
    @parallel assign!(Arrays.T, Arrays.T_init)
    time, dike_inj, Time_vec,Melt_Time,Tav_magma_Time, Tav_3D_magma_Time, VolMelt_time,
    Tav_all_Time, Tav_3D_all_Time, Tav_Phase2_Time, Tav_3D_Phase2_Time = 0.0, 0.0,zeros(Num.nt,1),zeros(Num.nt,1),zeros(Num.nt,1),
                    zeros(Num.nt,1), zeros(Num.nt,1), zeros(Num.nt,1), zeros(Num.nt,1), zeros(Num.nt,1), zeros(Num.nt,1);

    if isdir(Num.SimName)==false mkdir(Num.SimName) end;    # create simulation directory if needed
    # --------------------------------------------

    # Initialize sample points on the grid -------
    #  This tracks Tt evolution on fixed grid points in the same manner as the other codes do it (these tracers remain fixed in space)
    if Num.TrackTracersOnGrid==true
        X,Z = Array(Arrays.R), Array(Arrays.Z)
        Tracers_grid     =   StructArray{Tracer}(undef, 1)
        for i in eachindex(X)
            Tracers0 = Tracer(coord=[X[i]-1e-3,Z[i]])   #
            push!(Tracers_grid, Tracers0);
        end
        MagmaThermoKinematics.StructArrays.foreachfield(v -> deleteat!(v, 1), Tracers_grid)         # Delete first (undefined) row of tracer StructArray.
        Tnew_cpu      .= Array(Arrays.T_init)
        Phi_melt_cpu  .= Array(Arrays.ϕ)

        UpdateTracers_T_ϕ!(Tracers_grid, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);      # Initialize info on grid trcers

        @show length(Tracers_grid)
    end
    # --------------------------------------------

    for it = 1:Num.nt   # Time loop
        time                =   time + Num.dt;                                     # Keep track of evolved time

        # Add new dike every X years -----------------
        if floor(time/Dikes.InjectionInterval)> dike_inj
            dike_inj            =   floor(time/Dikes.InjectionInterval)                     # Keeps track on what was injected already
            dike                =   Dike(dike, Center=Dikes.Center[:],Angle=[0]);           # Specify dike with random location/angle but fixed size/T
            Tnew_cpu           .=   Array(Arrays.T)
            @timeit to "Dike intrusion" Tracers, Tnew_cpu,Vol,dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, Dikes.nTr_dike, dike_poly=dike_poly);     # Add dike, move hostrocks

            if Num.flux_bottom_BC==false
                # Keep bottom T absolutey constant (advection modifies this)
                Z               = Array(Arrays.Z)
                Tnew_cpu[:,1]   .=   @. Num.Tsurface_Celcius - Z[:,1]*Num.Geotherm
            end
            Arrays.T           .=   Data.Array(Tnew_cpu)
            InjectVol          +=   Vol                                                     # Keep track of injected volume
            Qrate               =   InjectVol/time
            Qrate_km3_yr        =   Qrate*SecYear/km³
            Qrate_km3_yr_km2    =   Qrate_km3_yr/(pi*(Dike_params.W_in/2/1e3)^2)

            @printf "  Added new dike; time=%.3f kyrs, total injected magma volume = %.2f km³; rate Q= %.2e km³yr⁻¹  \n" time/kyr InjectVol/km³ Qrate_km3_yr

            if Num.advect_polygon==true && isempty(dike_poly)
                dike_poly   =   CreateDikePolygon(dike);            # create dike for the 1th time
            end

            if length(Mat_tup)>1
                @timeit to "PhasesFromTracers"  PhasesFromTracers!(Array(Phases), Grid, Tracers, BackgroundPhase=1, InterpolationMethod="Constant");    # update phases from grid
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

        @timeit to "Update tracers"  UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);     # Update info on tracers

        if (Num.TrackTracersOnGrid==true) &&  (mod(it,100)==0)
            UpdateTracers_T_ϕ!(Tracers_grid, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);                             # Initialize info on grid tracers
            update_Tvec!(Tracers_grid, time/SecYear*1e-6)                                                        # update T & time vectors on tracers
        end
        # copy back to gpu
        Arrays.Tnew   .= Data.Array(Tnew_cpu)
        Arrays.ϕ      .= Data.Array(Phi_melt_cpu)

        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)
        Melt_Time[it]       =   sum( Arrays.ϕ)/(Num.Nx*Num.Nz)                      # Average melt fraction in crust

        ind = findall(Arrays.T.>700);
        if ~isempty(ind)
            Tav_magma_Time[it] = sum(Arrays.T[ind])/length(ind)                     # average T of part with magma
        else
            Tav_magma_Time[it] = NaN;
        end

        ind = findall((Arrays.T.>700) .& (Phases.==2));
        if ~isempty(ind)
            Tav_Phase2_Time[it] = sum(Arrays.T[ind])/length(ind)                     # average T of part with magma
        else
            Tav_Phase2_Time[it] = NaN;
        end
        Tav_all_Time[it] = sum(Arrays.T)/length(Arrays.T)

        # Store volume of melt (with T>0)
        ind                 = findall(Tnew_cpu .> 700);
        ix                  = [ind[i][1] for i=1:length(ind)]
        rc                  = Grid.coord1D[1][ix]
        VolCells            = 2*π*rc*Grid.Δ[1]*Grid.Δ[2];
        VolMelt_time[it]    = sum(VolCells);
        Tav_3D_magma_Time[it] = sum(VolCells.*Tnew_cpu[ind])/sum(VolCells)           # 3D average Temperature


        ind                 = findall(Tnew_cpu .> 0);
        ix                  = [ind[i][1] for i=1:length(ind)]
        rc                  = Grid.coord1D[1][ix]
        VolCells            = 2*π*rc*Grid.Δ[1]*Grid.Δ[2];
        Tav_3D_all_Time[it] = sum(VolCells.*Tnew_cpu[ind])/sum(VolCells)           # 3D average Temperature

        ind                 = findall((Tnew_cpu .> 700) .& (Array(Phases).==2));
        if ~isempty(ind)
            ix                  = [ind[i][1] for i=1:length(ind)]
            rc                  = Grid.coord1D[1][ix]
            VolCells            = 2*π*rc*Grid.Δ[1]*Grid.Δ[2];
            Tav_3D_Phase2_Time[it] = sum(VolCells.*Tnew_cpu[ind])/sum(VolCells)         # 3D average Temperature of phase 1
        else
            Tav_3D_Phase2_Time[it] = NaN                                                # 3D average Temperature of phase 2
        end

        Time_vec[it]        =   time;                                               # Vector with time

        if mod(it,10)==0
            update_Tvec!(Tracers, time/SecYear*1e-6)                                # update T & time vectors on tracers
        end
        # --------------------------------------------

        # Visualize results --------------------------
        if mod(it,Num.CreateFig_steps)==0  || it==Num.nt
            @timeit to "visualisation" begin
            time_Myrs = time/SecYear*1e-6;

            T_plot          = Array(Arrays.Tnew)
            T_init_plot     = Array(Arrays.T_init)
            Phi_melt_plot   = Array(Arrays.ϕ)
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
            println("Timestep $it = $(round(time/kyr*100)/100) kyrs, max(T)=$(round(maximum(Arrays.T),digits=3))ᵒC, Tav_magma=$(T_av_melt)ᵒC, max(ϕ)=$(round(maximum(Arrays.ϕ),digits=2)), Vol_magma_3D = $(round(VolMelt_time[it]/1e9))km³, Tav_magma_3D=$(round(Tav_3D_magma_Time[it]))ᵒC")
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
                if Num.TrackTracersOnGrid==true
                    # Only keep Tracers_grid, which are still partially molten (saved diskspace)
                    ind = findall(Tracers_grid.Phi .> 0.0)
                                 Tracers_grid = Tracers_grid[ind]
                end

                # save tracers & material parameters of the simulation in jld2 format so we can reproduce this
                filename = "$(Num.SimName)/Tracers_SimParams.jld2"
                Phases_float = Float64.(Phases)
                if Num.TrackTracersOnGrid
                    jldsave(filename; Tracers, Dikes, Num, Mat_tup, Time_vec, Melt_Time, Tav_magma_Time, Tav_3D_magma_Time, Tav_Phase2_Time, Tav_3D_Phase2_Time, Tav_3D_all_Time, Tav_all_Time, VolMelt_time,  Phases_float, Tnew_cpu, Tracers_grid, Grid)
                else
                    jldsave(filename; Tracers, Dikes, Num, Mat_tup, Time_vec, Melt_Time, Tav_magma_Time, Tav_3D_magma_Time, Tav_Phase2_Time, Tav_3D_Phase2_Time, Tav_3D_all_Time, Tav_all_Time, VolMelt_time,  Phases_float, Tnew_cpu, Grid)
                end
                println("  Saved Tracers & simulation parameters to file $filename ")

                filename = "$(Num.SimName)/Tracers.mat"
                matwrite(filename,  Dict("Tracers"=> Tracers))      # does not allow
                println("  Saved Tracers to file $filename ")
            end

        end
        # --------------------------------------------

    end
    x,z = Grid.coord1D[1],Grid.coord1D[2]
    return x,z,Arrays.T, Time_vec, Melt_Time, Tracers, dike_poly, Grid, Phases;
end # end of main function


# Define material parameters for the simulation.
if 1==0

    # 2D, run7-9 Geneva-type models with Greg's smooth melting parameterisation but different Geother & BC;s
    # These are the final simulations for the ZASSy paper
    Num         = NumParam(Nx=269*1, Nz=269*1, SimName="ZASSy_Geneva_10_7e_6_v2", axisymmetric=true,
                            maxTime_Myrs=1.5, fac_dt=0.2, ω=0.5, verbose=false,
                            flux_bottom_BC=false, flux_bottom=0, deactivate_La_at_depth=false,
                            Geotherm=30/1e3, TrackTracersOnGrid=true,
                            SaveOutput_steps=100000, CreateFig_steps=100000, plot_tracers=false, advect_polygon=true,
                            FigTitle="Geneva Models, Geotherm 30/km");

    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion",
                            #InjectionInterval_year = 10e3,      # flux= 7.5e-6 km3/km2/yr
                            InjectionInterval_year = 7000,      # flux= 10.7e-6 km3/km2/yr
                            #InjectionInterval_year = 8200,       # flux= 9.1e-6 km3/km2/yr
                            #InjectionInterval_year = 5000,       # flux= 14.9e-6 km3/km2/yr
                            W_in=20e3, H_in=74.6269,
			                nTr_dike=300*1
			    )

    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1,
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                    #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                  Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                 #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                  HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                       Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting
                                      # Melting = MeltingParam_Caricchi()),                     # Caricchi melting

				      # add more parameters here, in case you have >1 phase in the model
                    )
end


if 1==0
   # 2D, UCLA-type models
    #
    # Benchmark case with Oscar Lovera, who uses an exponential depth-dependent radioactive heating term combined with flux lower BC
    #
    #  For the parameters he gave, k=3.35 W/mK  qs=170 mW/m2    qm=167mW/m2  hr=10e3m
    #  H0  = (qs-qm)/hr= 3.0000e-07

    # ZASSy_UCLA_ellipticalIntrusion_constant_k_radioactiveheating_smoothQuad_initialEllipse_3
    # ZASSy_UCLA_ellipticalIntrusion_variable_k_radioactiveheating_smoothQuad_initialEllipse
    # ZASSy_UCLA_ellipticalIntrusion_constant_k_radioactiveheating_Quadratic_noHostrockLatent_initialEllipse
#=
    # Constant k case:
    Num          = NumParam(Nx=301, Nz=201, W=30e3, SimName="ZASSy_UCLA_ellipticalIntrusion_constant_k_radioactiveheating_AssimilationAndQuadratic_initialEllipse_La0_Lm267",
                            SaveOutput_steps=1000, CreateFig_steps=1000, axisymmetric=false,
                            flux_bottom_BC=true, flux_bottom=167e-3, fac_dt=0.4,  ω=0.6, verbose=false, dt = 20*SecYear,
                            maxTime_Myrs=0.7,
                            AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=170e-3, qm_anal=167e-3, hr_anal=10e3, k_anal=3.3453,
                            InitialEllipse =   true, a_init= 2.5e3,  b_init  =   1.5e3,
                            FigTitle="UCLA Models", plot_tracers=false, advect_polygon=true);
  =#


    # Note: in k-dependent cases, we have a much lower k @ the bottom
    #   julia> p=T_Conductivity_Whittington();
    #   julia> compute_conductivity(p,T=1030+273.15)
    #           1.837397823380793
    #
    #   julia> p1=T_Conductivity_Whittington_parameterised();
    #   julia> compute_conductivity(p1,[1030+273.15])
    #   1-element Vector{Float64}:
    #           1.793946
    # for that reason, we have to decrease the bottom heat flux
    Num          = NumParam(Nx=301, Nz=201, W=30e3, SimName="ZASSy_UCLA_BenchmarkCase_Geotherm40_Fig12A_density2700",
                        SaveOutput_steps=2000, CreateFig_steps=1000, axisymmetric=false,
#                        flux_bottom_BC=true, flux_bottom=40/1e3*1.84, fac_dt=0.2, ω=0.6, verbose=false,

                        # Initial benchmark:
                        #maxTime_Myrs=0.5,
                        #flux_bottom_BC=true, flux_bottom=50/1e3*1.84, fac_dt=0.2, ω=0.6, verbose=false,
                        #AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=170e-3, qm_anal=167e-3, hr_anal=10e3, k_anal=3.3453,
                        #InitialEllipse =   true, a_init= 2.5e3,  b_init  =   1.5e3,

                        # Fig 12A:
                        maxTime_Myrs=0.75,
                        flux_bottom_BC=true, flux_bottom=40/1e3*1.89, fac_dt=0.2, ω=0.6, verbose=false,
                        AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=130e-3, qm_anal=130e-3, hr_anal=10e3, k_anal=3.3453,
                        InitialEllipse =   true, a_init= 2.33e3,  b_init  =   0.44e3,

                        # Fig 12B:
                        #maxTime_Myrs=1.5,
                        #flux_bottom_BC=true, flux_bottom=40/1e3*1.89, fac_dt=0.2, ω=0.6, verbose=false,
                        #AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=130e-3, qm_anal=130e-3, hr_anal=10e3, k_anal=3.3453,
                        #InitialEllipse =   true, a_init= 1.62e3,  b_init  =   0.91e3,

                        # Fig 12C:
                        #maxTime_Myrs=1.21,
                        #flux_bottom_BC=true, flux_bottom=40/1e3*1.89, fac_dt=0.2, ω=0.6, verbose=false,
                        #AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=130e-3, qm_anal=130e-3, hr_anal=10e3, k_anal=3.3453,
                        #InitialEllipse =   true, a_init= 1.64e3,  b_init  =   0.89e3,

                        FigTitle="UCLA Models", plot_tracers=false, advect_polygon=true);


    #=
    # Our original benchmark case
    mid_depth_km = -7.0;                   # mid depth of injection area [km]
    V_inj_a      = 1.29135;                                 # a axis in km of injected ellipsoid
    V_inj_b      = 0.7745;                                  # b axis in km
    V_inj_check  = 4/3*pi*V_inj_a^2*V_inj_b;                # checking
    InjectionInterval_yr = 5000;
    =#


    # Fig. 12A as provided by Oscar
    mid_depth_km = -8.4;                                # mid depth of injection area [km]
    Vol_inj      = 2*5;                                 # email gives an injection rate of 2km3/kyr = 10km3/5kyrs
    AR           = 2.33/0.44
    V_inj_a      = (Vol_inj*AR/(4/3*π))^(1/3)
    V_inj_b      = V_inj_a/AR;                          # b axis in km
    V_inj_check  = 4/3*pi*V_inj_a^2*V_inj_b;            # checking
    InjectionInterval_yr = 5000;

    # Fig. 12B as provided by Oscar
    #mid_depth_km = -12.1;                                # mid depth of injection area [km]
    #Vol_inj      = 2*5;                                 # email gives an injection rate of 2km3/kyr = 10km3/5kyrs
    #AR           = 1.62/0.91
    #V_inj_a      = (Vol_inj*AR/(4/3*π))^(1/3)
    #V_inj_b      = V_inj_a/AR;                          # b axis in km
    #V_inj_check  = 4/3*pi*V_inj_a^2*V_inj_b;            # checking
    #InjectionInterval_yr = 5000;

    # Fig. 12C as provided by Oscar
    #mid_depth_km = -11.9;                                # mid depth of injection area [km]
    #Vol_inj      = 2*5;                                 # email gives an injection rate of 2km3/kyr = 10km3/5kyrs
    #AR           = 1.64/0.89
    #V_inj_a      = (Vol_inj*AR/(4/3*π))^(1/3)
    #V_inj_b      = V_inj_a/AR;                          # b axis in km
    #V_inj_check  = 4/3*pi*V_inj_a^2*V_inj_b;            # checking
    #InjectionInterval_yr = 5000;

    # Use the parameters. Note that we specify the diameter of the ellipse in here
    Dike_params  = DikeParam(Type="EllipticalIntrusion", InjectionInterval_year = InjectionInterval_yr,
                            W_in=V_inj_a*2*1e3,
                            H_in=V_inj_b*2*1e3,
                            Center=[0, mid_depth_km*1e3],
                            nTr_dike=3000)

    MatParam     = (SetMaterialParams(Name="Host rock", Phase=1,
                                    Density    = ConstantDensity(ρ=2700.0kg/m^3),                    # used in the parameterisation of Whittington
                                    LatentHeat = ConstantLatentHeat(Q_L=2.55e5J/kg),
                               RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=3e-7Watt/m^3),
                                 Conductivity = T_Conductivity_Whittington(),                       # T-dependent k
                                 HeatCapacity = T_HeatCapacity_Whittington(),     # T-dependent cp
                                Melting = MeltingParam_Assimilation()
                                 ),       # Quadratic parameterization as in Tierney et al.
                    SetMaterialParams(Name="Intruded rocks", Phase=2,
                                    Density    = ConstantDensity(ρ=2700.0kg/m^3),                    # used in the parameterisation of Whittington
                                    LatentHeat = ConstantLatentHeat(Q_L=2.67e5J/kg),
                               RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=3e-7Watt/m^3),
                                 Conductivity = T_Conductivity_Whittington(),                       # T-dependent k
                                 HeatCapacity = T_HeatCapacity_Whittington(),                       # T-dependent cp
                                       Melting = SmoothMelting(MeltingParam_Quadratic(T_s=(700+273.15)K, T_l=(1100+273.15)K)))
            )
end

if 1==1
    # 2D, UCLA-type models as used in the ZASSy paper (see above for benchmark setups)
    # Cases as shown in the ZASSy paper, figures 11 & 12 & systematic runs in Appendix B

    # Note: in k-dependent cases, we have a much lower k @ the bottom
    #   julia> p=T_Conductivity_Whittington();
    #   julia> compute_conductivity(p,T=1030+273.15)
    #           1.837397823380793
    # for that reason, we have to decrease the bottom heat flux

    #=
    # This is the reference model used in Fig. 11 of the paper:
    Num          = NumParam(Nx=301, Nz=201, W=30e3, SimName="ZASSy_UCLA_9_1e_6_reference",
                     SaveOutput_steps=200000, CreateFig_steps=1000, axisymmetric=false,
                     flux_bottom_BC=true, flux_bottom=30/1e3*1.9, fac_dt=0.2, ω=0.5, verbose=false,
                     maxTime_Myrs=1.1,
                     AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=100e-3, qm_anal=100e-3, hr_anal=10e3, k_anal=3.3453,
                     InitialEllipse =   true, a_init= 6.7e3,  b_init  =   1.67e3,
                     FigTitle="UCLA Models", plot_tracers=true, advect_polygon=true, TrackTracersOnGrid=true);
    =#

    Num          = NumParam(Nx=301, Nz=201, W=30e3, SimName="ZASSy_UCLA_10_7e_6_v2",
                         SaveOutput_steps=200000, CreateFig_steps=1000, axisymmetric=false,
                         flux_bottom_BC=true, flux_bottom=30/1e3*1.9, fac_dt=0.2, ω=0.5, verbose=false,
                         #maxTime_Myrs=1.1,  # Fig. 11, Fig. 12B
                         #maxTime_Myrs=0.7,  # Fig. 12A
                         #maxTime_Myrs=1.3,  # Fig. 12C
                         maxTime_Myrs=1.25,  # Fig. 12C


                         AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=100e-3, qm_anal=100e-3, hr_anal=10e3, k_anal=3.3453,
                         #InitialEllipse =   true, a_init= 6.7e3,  b_init  =   1.67e3,       # reference case, Fig. 12B
                         #InitialEllipse =   true, a_init= 5.39e3,  b_init  =   1.348e3,     # Fig. 12A
                         #InitialEllipse =   true, a_init= 8.35e3,  b_init  =   2.088e3,     # Fig. 12C
                         InitialEllipse =   true, a_init= 7.38e3,  b_init  =   1.84e3,     # Fig. 12C, 10.7

                         FigTitle="UCLA Models", plot_tracers=true, advect_polygon=true, TrackTracersOnGrid=true);


    # Reference case:
    #Flux         = 9.1e-6;                              # in km3/km2/a
    #Flux         = 7.46e-6;         # Fig. 12A
   # Flux         = 1.49254e-5;      # 12C
    Flux         = 10.7e-6;     # 12C

    Total_r_km   = 10;                                  # final radius of area

    Total_A_km2  = pi*Total_r_km^2;                     # final area in km^2
    Flux_km3_a   = Flux*Total_A_km2;                    # flux in km3/year
    V_total_km3  = Flux_km3_a*Num.maxTime_Myrs*1e6

    mid_depth_km        =   -7.0;                                           # mid depth of injection area [km]
    AspectRatio         =   Num.a_init/Num.b_init;                          # Aspect ratio of initial sill (kept constant)
    V_initial           =  (4/3)*π*(Num.a_init/1e3)^2*(Num.b_init/1e3)      # initial injected volume in km3
    Vol_inj_year        =  (V_total_km3-V_initial)/(Num.maxTime_Myrs*1e6);  # Injected volume per year

    V_initial_opt_a     =   (0.1*V_total_km3*AspectRatio/((4/3)*π))^(1/3)
    V_initial_opt_b     =   V_initial_opt_a/AspectRatio

    RatioInitialTotal   =   V_initial/V_total_km3;
    println("Starting simulation with flux: $(Flux) km³km⁻²yr⁻¹, total time= $(Num.maxTime_Myrs*1e3) kyrs, max. volume=$(round(V_total_km3))km³, Initial/Total volume = $(V_initial/V_total_km3)")

    InjectionInterval_yr=   5000;                                           # Injection interval

    Vol_inj             =   Vol_inj_year*InjectionInterval_yr;              # Volume injected every injection event
    V_inj_a             =   (Vol_inj*AspectRatio/((4/3)*π))^(1/3)           # a-axis of injected ellipse
    V_inj_b             =   V_inj_a/AspectRatio;                            # b axis in km



     # Use the parameters. Note that we specify the diameter of the ellipse in here
     Dike_params  = DikeParam(Type="EllipticalIntrusion", InjectionInterval_year = InjectionInterval_yr,
                             W_in=V_inj_a*2*1e3,
                             H_in=V_inj_b*2*1e3,
                             Center=[0, mid_depth_km*1e3],
                             nTr_dike=300*4)

     MatParam     = (SetMaterialParams(Name="Host rock", Phase=1,
                                     Density    = ConstantDensity(ρ=2700kg/m^3),                    # used in the parameterisation of Whittington
                                     LatentHeat = ConstantLatentHeat(Q_L=2.55e5J/kg),
                                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                                 Conductivity = T_Conductivity_Whittington(),                       # T-dependent k
                                  HeatCapacity = T_HeatCapacity_Whittington(),                      # T-dependent cp
                                 # Conductivity = ConstantConductivity(k=3.3Watt/K/m),               # in case we use constant k
                                 # HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                 Melting = MeltingParam_Assimilation()                              # Quadratic parameterization as in Tierney et al.
                                 #Melting = MeltingParam_Caricchi()
                                  ),
                     SetMaterialParams(Name="Intruded rocks", Phase=2,
                                     Density    = ConstantDensity(ρ=2700kg/m^3),                     # used in the parameterisation of Whittington
                                     LatentHeat = ConstantLatentHeat(Q_L=2.67e5J/kg),
                                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                                  Conductivity = T_Conductivity_Whittington(),                       # T-dependent k
                                  HeatCapacity = T_HeatCapacity_Whittington(),                       # T-dependent cp
                                  #Conductivity = ConstantConductivity(k=3.3Watt/K/m),               # in case we use constant k
                                  #HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                        Melting = SmoothMelting(MeltingParam_Quadratic(T_s=(700+273.15)K, T_l=(1100+273.15)K))
                                        #Melting = MeltingParam_Caricchi()
                                        )
             )



 end


# Keep track of time
const to = TimerOutput()
reset_timer!(to)

# Call the main code with the specified material parameters
x,z,T, Time_vec,Melt_Time, Tracers, dike_poly, Grid, Phases = MainCode_2D(MatParam, Num, Dike_params); # start the main code

@show(to)

#plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot

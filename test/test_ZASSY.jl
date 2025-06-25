# This is a relatively complicated test, but is added to ensure that the simulations published in the ZASSY paper remain reproducible:
#
using Test, LinearAlgebra, SpecialFunctions, Random
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end
using ParallelStencil, ParallelStencil.FiniteDifferences2D

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(1)                     # select the GPU you use (starts @ zero)
    @init_parallel_stencil(CUDA, Float64, 2)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
    @init_parallel_stencil(Threads, Float64, 2)
end
using MagmaThermoKinematics.Diffusion2D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields2D


Random.seed!(1234);     # such that we can reproduce results


using Printf        # pretty print

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
        Tracers, Tnew_cpu,Vol,dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike_initial, Dikes.nTr_dike, dike_poly=dike_poly);     # Add dike, move hostrocks

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
    end
    # --------------------------------------------


    for it = 1:Num.nt   # Time loop
        time                =   time + Num.dt;                                     # Keep track of evolved time

        # Add new dike every X years -----------------
        if floor(time/Dikes.InjectionInterval)> dike_inj
            dike_inj            =   floor(time/Dikes.InjectionInterval)                     # Keeps track on what was injected already
            dike                =   Dike(dike, Center=Dikes.Center[:],Angle=[0]);           # Specify dike with random location/angle but fixed size/T
            Tnew_cpu           .=   Array(Arrays.T)
            Tracers, Tnew_cpu,Vol,dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, Dikes.nTr_dike, dike_poly=dike_poly);     # Add dike, move hostrocks

            if Num.flux_bottom_BC==false
                # Keep bottom T absolutey constant (advection modifies this)
                Z               = Array(Arrays.Z)
                Tnew_cpu[:,1]   .=   @. Num.Tsurface_Celcius - Z[:,1]*Num.Geotherm
            end
            Arrays.T           .=   Data.Array(Tnew_cpu)
            InjectVol          +=   Vol                                                     # Keep track of injected volume
            Qrate               =   InjectVol/time
            Qrate_km3_yr        =   Qrate*SecYear/km³
            Qrate_km3_yr_km2    =   Qrate_km3_yr/(pi*(Dikes.W_in/2/1e3)^2)

            @printf "  Added new dike; time=%.3f kyrs, total injected magma volume = %.2f km³; rate Q= %.2e km³yr⁻¹  \n" time/kyr InjectVol/km³ Qrate_km3_yr

            if Num.advect_polygon==true && isempty(dike_poly)
                dike_poly   =   CreateDikePolygon(dike);            # create dike for the 1th time
            end

            if length(Mat_tup)>1
               PhasesFromTracers!(Array(Phases), Grid, Tracers, BackgroundPhase=1, InterpolationMethod="Constant");    # update phases from grid
            end
        end
        # --------------------------------------------

        # Do a diffusion step, while taking T-dependencies into account
        Nonlinear_Diffusion_step_2D!(Arrays, Mat_tup, Phases, Grid, Num.dt, Num)
        # --------------------------------------------


        # Update variables ---------------------------
        # copy to cpu
        Tnew_cpu      .= Array(Arrays.Tnew)
        Phi_melt_cpu  .= Array(Arrays.ϕ)

        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);     # Update info on tracers

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
        # --------------------------------------------


        # Save output to disk once in a while --------
        # --------------------------------------------

    end
    x,z = Grid.coord1D[1],Grid.coord1D[2]
    return x,z,Arrays.T, Time_vec, Melt_Time, Tracers, dike_poly, Grid, Phases;
end # end of main function


# Define material parameters for the simulation.
@testset "ZASSY simulations" begin

Random.seed!(1234);     # such that we can reproduce results

if 1==1
    # Geneva setup
    println("===============================================")
    println("Testing the underaccretion ZASSy setup")
    println("===============================================")
    # These are the final simulations for the ZASSy paper, but done @ a lower resolution
    Num         = NumParam( #Nx=269*1, Nz=269*1,
                            Nx=135*1, Nz=135*1,
                            SimName="ZASSy_Geneva_9_1e_6", axisymmetric=true,
                            #maxTime_Myrs=1.5,
                            maxTime_Myrs=0.025,
                            fac_dt=0.2, ω=0.5, verbose=false,
                            flux_bottom_BC=false, flux_bottom=0, deactivate_La_at_depth=false,
                            Geotherm=30/1e3, TrackTracersOnGrid=true,
                            SaveOutput_steps=100000, CreateFig_steps=100000, plot_tracers=false, advect_polygon=true,
                            FigTitle="Geneva Models, Geotherm 30/km");
    Dike_params = DikeParam(Type="CylindricalDike_TopAccretion",
                            #InjectionInterval_year = 10e3,      # flux= 7.5e-6 km3/km2/yr
                            #InjectionInterval_year = 7000,      # flux= 10.7e-6 km3/km2/yr
                            InjectionInterval_year = 8200,       # flux= 9.1e-6 km3/km2/yr
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
    # Call the main code with the specified material parameters
    x,z,T, Time_vec,Melt_Time, Tracers, dike_poly, Grid, Phases = MainCode_2D(MatParam, Num, Dike_params); # start the main code
    @test sum(T)/prod(size(T)) ≈ 312.1505261202475  rtol= 1e-2
    @test sum(Melt_Time)  ≈ 0.16694675188794647  rtol= 1e-5


    # compute zircon ages for a few tracers
    time_vec    = Tracers.time_vec*1e6;
    T_vec       = Tracers.T_vec;
    time_vec    = time_vec[1:10];
    T_vec       = T_vec[1:10];

    #ZirconData  	=   ZirconAgeData(Tsat=820, Tmin=700, Tsol=700, Tcal_max=800, Tcal_step=1.0, max_x_zr=0.001, zircon_number=100, time_zr_growth=100);	 # note that we use a much longer zr_growth in the real calculations (700kyrs)
    #time_years, prob, ages_eruptible, number_zircons, T_av_time, T_sd_time = compute_zircons_Ttpath(time_vec, T_vec, ZirconData=ZirconData)
    #@show sum(prob), sum(number_zircons)

end

if 1==1
    # 2D, UCLA-type models as used in the ZASSy paper (see above for benchmark setups)
    println("===============================================")
    println("Testing the central intrusion ZASSy setup")
    println("===============================================")

    Num          = NumParam( #Nx=301, Nz=201,
                            Nx=151, Nz=101,
                            W=30e3, SimName="ZASSy_UCLA_10_7e_6_v2",
                         SaveOutput_steps=200000, CreateFig_steps=1000, axisymmetric=false,
                         flux_bottom_BC=true, flux_bottom=30/1e3*1.9, fac_dt=0.2,
                         ω=0.5,
                         convergence=1e-2,
                         verbose=false,
                         maxTime_Myrs= 0.025, # for testing
                         #maxTime_Myrs=1.1,  # Fig. 11, Fig. 12B
                         #maxTime_Myrs=0.7,  # Fig. 12A
                         #maxTime_Myrs=1.3,  # Fig. 12C
                         #maxTime_Myrs=1.25,  # Fig. 12C

                         AnalyticalInitialGeo=true, Tsurface_Celcius=25,   qs_anal=100e-3, qm_anal=100e-3, hr_anal=10e3, k_anal=3.3453,
                         InitialEllipse =   true, a_init= 6.7e3,  b_init  =   1.67e3,       # reference case, Fig. 12B, Fig. 11

                         FigTitle="UCLA Models", plot_tracers=true, advect_polygon=true, TrackTracersOnGrid=true);


    # Reference case:
    Flux         = 9.1e-6;                              # in km3/km2/a

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
    #V_inj_a             =   (Vol_inj*AspectRatio/((4/3)*π))^(1/3)           # a-axis of injected ellipse
    #V_inj_b             =   V_inj_a/AspectRatio;                            # b axis in km

    V_inj_a = 2.3098358683516853
    V_inj_b = 0.5757352089772111

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
                                 Melting = MeltingParam_Assimilation()                              # Quadratic parameterization as in Tierney et al.
                                  ),
                     SetMaterialParams(Name="Intruded rocks", Phase=2,
                                     Density    = ConstantDensity(ρ=2700kg/m^3),                     # used in the parameterisation of Whittington
                                     LatentHeat = ConstantLatentHeat(Q_L=2.67e5J/kg),
                                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                                  Conductivity = T_Conductivity_Whittington(),                       # T-dependent k
                                  HeatCapacity = T_HeatCapacity_Whittington(),                       # T-dependent cp
                                        Melting = SmoothMelting(MeltingParam_Quadratic(T_s=(700+273.15)K, T_l=(1100+273.15)K))
                                        )
             )

    # Call the main code with the specified material parameters
    x,z,T, Time_vec,Melt_Time, Tracers, dike_poly, Grid, Phases = MainCode_2D(MatParam, Num, Dike_params); # start the main code

    @test sum(T)/prod(size(T)) ≈ 351.6708073949723 rtol= 1e-4
    @test sum(Melt_Time)  ≈ 11.474144800106583 rtol= 1e-4


 end

end



#plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") #Create plot

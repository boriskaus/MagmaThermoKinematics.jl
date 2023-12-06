# This shows how to use the GeophysicalModelGenerator package to create a 2D thermal model and 
# run subsequent simulations using the MagmaThermoKinematics package.

#
module MTK_GMG_2D

export NumParam, DikeParam, TimeDepProps, MTK_inject_dikes, MTK_GeoParams_2D
export MTK_display_output, MTK_print_output, MTK_update_TimeDepProps!

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Parameters
using GeoParams

#using MagmaThermoKinematics.Grid
#using MagmaThermoKinematics.Fields
using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics

const SecYear = 3600*24*365.25;

"""
    Holds numerical parameters for the overall simulation (and sets defaults), 
"""
@with_kw mutable struct NumParam
    SimName::String             =   "Zassy_UCLA_ellipticalIntrusion"    # name of simulation
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
    SecYear::Float64            =   3600*24*365.25;
    maxTime::Float64            =   maxTime_Myrs*SecYear*1e6 # maximum timestep  in seconds
    SaveOutput_steps::Int64     =   1e3;            # saves output every x steps 
    CreateFig_steps::Int64      =   500;            # Create a figure every X steps
    flux_bottom_BC::Bool        =   false           # flux bottom BC?
    flux_bottom::Float64        =   167e-3          # Flux in W/m2 in case flux_bottom_BC=true
    plot_tracers::Bool          =   true            # adds passive tracers to the plot
    advect_polygon::Bool        =   false           # adds a polygon around the intrusion area
    axisymmetric::Bool          =   true            # axisymmetric (if true) of 2D geometry?
    κ_time::Float64             =   3.3/(1000*2700) # κ to determine the stable timestep 
    fac_dt::Float64             =   0.4;            # prefactor with which dt is multiplied   
    dt::Float64                 =   fac_dt*min(dx^2, dz^2)./κ_time/4;   # timestep
    time::Float64               =   0.0;            # current time          
    nt::Int64                   =   floor(maxTime/dt);
    it::Int64                   =   0;              # current iteration
    ω::Float64                  =   0.8;            # relaxation parameter for nonlinear iterations    
    max_iter::Int64             =   5000;           # max. number of nonlinear iterations        
    verbose::Bool               =   false;    
    convergence::Float64        =   1e-5;           # nonlinear convergence criteria      
    deactivate_La_at_depth::Bool=   false           # deactivate latent heating @ the bottom of the model box?
    deactivationDepth::Float64  =   -15e3           # deactivation depth
    dike_inj                    =   0.0
    USE_GPU                     =   false;

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
    SecYear                         =   3600*24*365.25;         # s/year
    InjectionInterval::Float64      =   InjectionInterval_year*SecYear;           # Injection interval [s]
    nTr_dike::Int64                 =   300                     # Number of tracers 
end


"""
    This holds various time-dependent properties of the simulation
"""
@with_kw mutable struct TimeDepProps
    Time_vec::Vector{Float64}  = [];        # Center of dike 
    MeltFraction::Vector{Float64} = [];        # Melt fraction over time
    Tav_magma::Vector{Float64} = [];   # average magma 
    Tmax::Vector{Float64} = [];        # max magma temperature
end


"""
    Analytical geotherm used for the UCLA setups, which includes radioactive heating
"""
function AnalyticalGeotherm!(T, Z, Tsurf, qm, qs, k, hr)

    T      .=  @. Tsurf - (qm/k)*Z + (qs-qm)*hr/k*( 1.0 - exp(Z/hr)) 

    return nothing
end

"""
    Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, dike, dike_poly, Tnew_cpu, InjectVol)
    function that displays injects dikes once in a while to the screen 
"""
function MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, dike, dike_poly, Tnew_cpu, InjectVol)

    if floor(Num.time/Dikes.InjectionInterval)> Num.dike_inj      
        Num.dike_inj        =   floor(Num.time/Dikes.InjectionInterval)                 # Keeps track on what was injected already
        dike                =   Dike(dike, Center=Dikes.Center[:],Angle=[0]);           # Specify dike with random location/angle but fixed size/T 
        Tnew_cpu           .=   Array(Arrays.T)
        Tracers, Tnew_cpu,Vol,dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, Dikes.nTr_dike, dike_poly=dike_poly);     # Add dike, move hostrocks
        @show length(Tracers)
        if Num.flux_bottom_BC==false
            # Keep bottom T constant (advection modifies this)
            Z               = Array(Arrays.Z)
            Tnew_cpu[:,1]   .=   @. Num.Tsurface_Celcius - Z[:,1]*Num.Geotherm
        end

        Arrays.T           .=   Data.Array(Tnew_cpu)
        InjectVol          +=   Vol                                                     # Keep track of injected volume
        Qrate               =   InjectVol/Num.time
        Qrate_km3_yr        =   Qrate*SecYear/km³
        Qrate_km3_yr_km2    =   Qrate_km3_yr/(pi*(Dikes.W_in/2/1e3)^2)

        println("  Added new dike; time=$(Num.time/kyr) kyrs, total injected magma volume = $(InjectVol/km³) km³; rate Q= $(Qrate_km3_yr) km³yr⁻¹") 
        
        if Num.advect_polygon==true && isempty(dike_poly)
            dike_poly   =   CreateDikePolygon(dike);            # create dike for the 1th time
        end

        if length(Mat_tup)>1
           PhasesFromTracers!(Phases, Grid, Tracers, BackgroundPhase=1, InterpolationMethod="Constant");    # update phases from grid 
        end

    end

    return Tracers
end

"""
    MTK_display_output(Grid, Num, Arrays, Mat_tup, Dikes)
    function that displays output to the screen 
"""
function MTK_visualize_output(Grid, Num, Arrays, Mat_tup, Dikes)

    return nothing
end

"""
    MTK_print_output(Grid, Num, Arrays, Mat_tup, Dikes)

Function that prints output to the REPL 
"""
function MTK_print_output(Grid::GridData, Num, Arrays, Mat_tup, Dikes)
    @show Num.it, maximum(Arrays.Tnew)
    
    return nothing
end


function MTK_update_TimeDepProps!(time_props, Grid, Num, Arrays, Mat_tup, Dikes)
    push!(time_props.Time_vec,      Num.time);   # time 
    push!(time_props.MeltFraction,  sum( Arrays.ϕ)/(Num.Nx*Num.Nz));    # melt fraction       


    ind = findall(Arrays.T.>700);          
    if ~isempty(ind)
        Tav_magma_Time = sum(Arrays.T[ind])/length(ind)     # average T of part with magma
    else
        Tav_magma_Time = NaN;
    end
    push!(time_props.Tav_magma, Tav_magma_Time);       # average magma T
    push!(time_props.Tmax,      maximum(Arrays.T));   # maximum magma T
    
    return nothing
end


#-----------------------------------------------------------------------------------------
"""
    Grid, time_props, Tracers, dike_poly, Phases = MTK_GeoParams_2D(Mat_tup, Num, Dikes);

Main routine that performs a 2D or 2D axisymmetric thermal diffusion simulation with injection of dikes.

Several functions are called which can be customized 

"""
@views function MTK_GeoParams_2D(Mat_tup, Num, Dikes);
    
    # Array & grid initializations ---------------
    Arrays = CreateArrays(Dict( (Num.Nx,  Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0,dϕdT_o=0, R=0, Z=0, P=0),
                                (Num.Nx-1,Num.Nz  )=>(qx=0,Kx=0, Rc=0), 
                                (Num.Nx  ,Num.Nz-1)=>(qz=0,Kz=0 )
                                ))

    # Set up model geometry & initial T structure
    Grid                    = CreateGrid(size=(Num.Nx,Num.Nz), extent=(Num.W, Num.H))   
    GridArray!(Arrays.R, Arrays.Z, Grid)        
    Arrays.Rc              .=   (Arrays.R[2:end,:] + Arrays.R[1:end-1,:])/2     # center points in x
    # --------------------------------------------

    Tracers                 =   StructArray{Tracer}(undef, 1)                       # Initialize tracers   
    dike                    =   Dike(W=Dikes.W_in,H=Dikes.H_in,Type=Dikes.Type,T=Dikes.T_in_Celsius, Center=Dikes.Center[:]);               # "Reference" dike with given thickness,radius and T

    # Set initial geotherm -----------------------
    Arrays.T_init      .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm;                # Initial (linear) temperature profile
    # --------------------------------------------
    
    # Update buffer & phases arrays --------------
    if Num.USE_GPU
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
    InjectVol   = 0.0;
    dike_poly   = []
    if Dikes.Type  == "CylindricalDike_TopAccretion"
        ind = findall( (Arrays.R.<=Dikes.W_in/2) .& (abs.(Arrays.Z.-Dikes.Center[2]) .< Dikes.H_in/2) );
        Arrays.T_init[ind] .= Dikes.T_in_Celsius;
        if Num.advect_polygon==true
            dike_poly   =   CreateDikePolygon(dike);
        end
    end
    # --------------------------------------------
    
    # Initialise arrays --------------------------
    @parallel assign!(Arrays.Tnew, Arrays.T_init)
    @parallel assign!(Arrays.T, Arrays.T_init)

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


    time_props = TimeDepProps();    # initialize time-dependent properties

    for Num.it = 1:Num.nt   # Time loop
        Num.time  += Num.dt;                                     # Keep track of evolved time

        # Add new dike every X years -----------------
        Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, dike, dike_poly, Tnew_cpu, InjectVol)
        # --------------------------------------------

        # Do a diffusion step, while taking T-dependencies into account
        Nonlinear_Diffusion_step_2D!(Arrays, Mat_tup, Phases, Grid, Num.dt, Num)
        # --------------------------------------------

        # Update variables ---------------------------
        # copy to cpu
        Tnew_cpu      .= Array(Arrays.Tnew)
        Phi_melt_cpu  .= Array(Arrays.ϕ)
        
        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);     # Update info on tracers 

        # copy back to gpu
        Arrays.Tnew   .= Data.Array(Tnew_cpu)
        Arrays.ϕ      .= Data.Array(Phi_melt_cpu)

        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)
        # --------------------------------------------
        
        #=
        ind = findall(Arrays.T.>700);          
        if ~isempty(ind)
            Tav_magma_Time[it] = sum(Arrays.T[ind])/length(ind)                     # average T of part with magma
        else
            Tav_magma_Time[it] = NaN;
        end

      
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

        =#

        if mod(Num.it,10)==0
            update_Tvec!(Tracers, Num.time/SecYear*1e-6)                                    # update T & time vectors on tracers
        end
        # --------------------------------------------

        # Update time-dependent properties -----------
        MTK_update_TimeDepProps!(time_props, Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

        # Visualize results --------------------------
        MTK_visualize_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------
          
        # Save output to disk once in a while --------

        # --------------------------------------------

        # Display output --------
        MTK_print_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

    end
    return Grid, Arrays, Tracers, dike_poly, Phases, time_props
end # end of main function


#=
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
                            #InjectionInterval_year = 8200,       # flux= 9.1e-6 km3/km2/yr
                            #InjectionInterval_year = 5000,       # flux= 14.9e-6 km3/km2/yr
                            InjectionInterval_year = 1000,       # flux= 14.9e-6 km3/km2/yr
                            
                            W_in=20e3, H_in=74.6269*10,
                            nTr_dike=300*1
                )
    MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                    Density    = ConstantDensity(ρ=2700kg/m^3),
                                    LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                    #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                    Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting     
                                    # Melting = MeltingParam_Caricchi()),                     # Caricchi melting
                    # add more parameters here, in case you have >1 phase in the model                                    
                    )
    # Call the main code with the specified material parameters
    Grid, time_props, Tracers, dike_poly, Phases = MainCode_2D(MatParam, Num, Dike_params); # start the main code
    @test sum(T)/prod(size(T)) ≈ 312.1505261202475  rtol= 1e-2
    @test sum(Melt_Time)  ≈ 0.1707068724854955  rtol= 1e-5


    # compute zircon ages for a few tracers
    time_vec    = Tracers.time_vec*1e6;
    T_vec       = Tracers.T_vec;
    time_vec    = time_vec[1:10];
    T_vec       = T_vec[1:10];

    #ZirconData  	=   ZirconAgeData(Tsat=820, Tmin=700, Tsol=700, Tcal_max=800, Tcal_step=1.0, max_x_zr=0.001, zircon_number=100, time_zr_growth=100);	 # note that we use a much longer zr_growth in the real calculations (700kyrs) 
    #time_years, prob, ages_eruptible, number_zircons, T_av_time, T_sd_time = compute_zircons_Ttpath(time_vec, T_vec, ZirconData=ZirconData)
    #@show sum(prob), sum(number_zircons)
        
end
=#

end
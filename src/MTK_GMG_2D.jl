# This shows how to use the GeophysicalModelGenerator package to create a 2D thermal model and 
# run subsequent simulations using the MagmaThermoKinematics package.

#
module MTK_GMG_2D

export NumParam, DikeParam, TimeDepProps, MTK_inject_dikes, MTK_GeoParams_2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Parameters
using GeoParams
using StructArrays

using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics


const SecYear = 3600*24*365.25;


"""
    Holds numerical parameters for the overall simulation (and sets defaults), 
"""
@with_kw mutable struct NumParam <: NumericalParameters
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
@with_kw mutable struct DikeParam <: DikeParameters
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
    InjectVol                       =   0.0;                    # injected volume
    Qrate_km3_yr                    =   0.0;                    # Dikes insertion rate
    dike_poly                       =   [];                     # polygon with dike
    dike_inj                        =   0.0
end


"""
    This holds various time-dependent properties of the simulation
"""
@with_kw mutable struct TimeDepProps <: TimeDependentProperties
    Time_vec::Vector{Float64}  = [];        # Center of dike 
    MeltFraction::Vector{Float64} = [];     # Melt fraction over time
    Tav_magma::Vector{Float64} = [];        # Average magma 
    Tmax::Vector{Float64} = [];             # Max magma temperature
end


"""
    Analytical geotherm used for the UCLA setups, which includes radioactive heating
"""
function AnalyticalGeotherm!(T, Z, Tsurf, qm, qs, k, hr)

    T      .=  @. Tsurf - (qm/k)*Z + (qs-qm)*hr/k*( 1.0 - exp(Z/hr)) 

    return nothing
end

"""
    Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, dike, Tnew_cpu)

Function that injects dikes once in a while
"""
function MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters, Tracers::StructVector, dike, Tnew_cpu)

    if floor(Num.time/Dikes.InjectionInterval)> Dikes.dike_inj      
        Dikes.dike_inj      =   floor(Num.time/Dikes.InjectionInterval)                 # Keeps track on what was injected already
        dike                =   Dike(dike, Center=Dikes.Center[:],Angle=[0]);           # Specify dike with random location/angle but fixed size/T 
        Tnew_cpu           .=   Array(Arrays.T)
        Tracers, Tnew_cpu,Vol,Dikes.dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, Dikes.nTr_dike, dike_poly=Dikes.dike_poly);     # Add dike, move hostrocks
       
        if Num.flux_bottom_BC==false
            # Keep bottom T constant (advection modifies this)
            Z               = Array(Arrays.Z)
            Tnew_cpu[:,1]   .=   @. Num.Tsurface_Celcius - Z[:,1]*Num.Geotherm
        end

        Arrays.T           .=   Data.Array(Tnew_cpu)
        Dikes.InjectVol    +=   Vol                                                     # Keep track of injected volume
        Qrate               =   Dikes.InjectVol/Num.time
        Dikes.Qrate_km3_yr  =   Qrate*SecYear/km³
        Qrate_km3_yr_km2    =   Dikes.Qrate_km3_yr/(pi*(Dikes.W_in/2/1e3)^2)
        println("  Added new dike; time=$(Num.time/kyr) kyrs, total injected magma volume = $(Dikes.InjectVol/km³) km³; rate Q= $(Dikes.Qrate_km3_yr) km³yr⁻¹") 
        
        if Num.advect_polygon==true && isempty(Dikes.dike_poly)
            Dikes.dike_poly   =   CreateDikePolygon(dike);            # create dike for the 1th time
        end

        if length(Mat_tup)>1
           PhasesFromTracers!(Arrays.Phases, Grid, Tracers, BackgroundPhase=1, InterpolationMethod="Constant");    # update phases from grid 
        end

    end

    return Tracers
end

"""
    MTK_display_output(Grid, Num, Arrays, Mat_tup, Dikes)

Function that creates plots 
"""
function MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

    return nothing
end

"""
    MTK_print_output(Grid, Num, Arrays, Mat_tup, Dikes)

Function that prints output to the REPL 
"""
function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    
    return nothing
end

"""
    MTK_update_TimeDepProps!(time_props, Grid, Num, Arrays, Mat_tup, Dikes)

Update time-dependent properties during a simulation
"""
function MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
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

"""
    MTK_update_Arrays!(Arrays, Grid, Num)

Update the arrays (in case you want to change them during a simulation)
"""
function MTK_update_Arrays!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters)
    return nothing
end

"""
    MTK_save_output(Grid, Arrays, Tracers, Dikes, time_props)

Save the output to disk
"""
function MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers, Dikes::DikeParameters, time_props::TimeDependentProperties)

    return nothing
end
#-----------------------------------------------------------------------------------------
"""
    Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(Mat_tup, Num, Dikes);

Main routine that performs a 2D or 2D axisymmetric thermal diffusion simulation with injection of dikes.

Several functions are called every timestep. which can be overwritten every timestep:

- `MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)`

"""
@views function MTK_GeoParams_2D(Mat_tup::Tuple, Num::NumericalParameters, Dikes::DikeParameters);
    
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
    Arrays = (Arrays..., Phases=Phases);
    
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
        Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, dike, Tnew_cpu)
        # --------------------------------------------

        # Do a diffusion step, while taking T-dependencies into account
        Nonlinear_Diffusion_step_2D!(Arrays, Mat_tup, Arrays.Phases, Grid, Num.dt, Num)
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
        
        if mod(Num.it,10)==0
            # Update T vector on tracers
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
        MTK_save_output(Grid, Arrays, Tracers, Dikes, time_props)
        # --------------------------------------------

        # Optionally Update arrays (such as T) -------
        MTK_update_Arrays!(Arrays, Grid, Num)
        # --------------------------------------------
        
        # Display output -----------------------------
        MTK_print_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

    end
    return Grid, Arrays, Tracers, Dikes, time_props
end # end of main function


end
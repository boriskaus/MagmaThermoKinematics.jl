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
using GeophysicalModelGenerator
using CUDA

using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics
using JLD2

const SecYear = 3600*24*365.25;


"""
This mutable structure represents numerical parameters in the program. It is used to store and manage numerical values that are used throughout the program.
    mutable struct NumParam <: NumericalParameters

# Fields

- `SimName::String`: Name of the simulation.
- `FigTitle`: Title of the figure.
- `Nx::Int64`: Number of grid points in the x direction.
- `Nz::Int64`: Number of grid points in the z direction.
- `W::Float64`: Width of the domain.
- `H::Float64`: Height of the domain.
- `dx::Float64`: Grid spacing in the x direction.
- `dz::Float64`: Grid spacing in the z direction.
- `Tsurface_Celcius::Float64`: Surface temperature in Celsius.
- `Geotherm::Float64`: Geothermal gradient in K/m.
- `maxTime_Myrs::Float64`: Maximum simulation time in Myrs.
- `SecYear::Float64`: Number of seconds in a year.
- `maxTime::Float64`: Maximum simulation time in seconds.
- `SaveOutput_steps::Int64`: Number of steps between output saves.
- `CreateFig_steps::Int64`: Number of steps between figure creations.
- `flux_bottom_BC::Bool`: Whether to apply a flux at the bottom boundary.
- `flux_bottom::Float64`: Flux at the bottom boundary in W/m^2.
- `plot_tracers::Bool`: Whether to plot tracers.
- `advect_polygon::Bool`: Whether to advect a polygon around the intrusion area.
- `axisymmetric::Bool`: Whether the simulation is axisymmetric.
- `κ_time::Float64`: Thermal diffusivity.
- `fac_dt::Float64`: Factor to multiply the time step by.
- `dt::Float64`: Time step.
- `time::Float64`: Current time.
- `nt::Int64`: Total number of time steps.
- `it::Int64`: Current iteration.
- `ω::Float64`: Relaxation parameter for nonlinear iterations.
- `max_iter::Int64`: Maximum number of nonlinear iterations.
- `verbose::Bool`: Whether to print verbose output.
- `convergence::Float64`: Convergence criterion for nonlinear iterations.
- `deactivate_La_at_depth::Bool`: Whether to deactivate latent heating at the bottom of the model box.
- `deactivationDepth::Float64`: Depth at which to deactivate latent heating.
- `USE_GPU`: Whether to use a GPU.
- `AnalyticalInitialGeo::Bool`: Whether to use an analytical initial geotherm.
- `qs_anal::Float64`: Analytical surface heat flux.
- `qm_anal::Float64`: Analytical mantle heat flux.
- `hr_anal::Float64`: Analytical radiogenic heat production.
- `k_anal::Float64`: Analytical thermal conductivity.
- `InitialEllipse::Bool`: Whether to initialize with an ellipse.
- `a_init::Float64`: Semi-major axis of initial ellipse.
- `b_init::Float64`: Semi-minor axis of initial ellipse.
- `TrackTracersOnGrid::Bool`: Whether to track tracers on the grid.

# Examples

```julia
np = NumParam(SimName="MySim", Nx=101, Nz=101, ...)
```

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
    keep_init_RockPhases::Bool  =   true;           # keep initial rock phases (if false, all phases are initialized as Dikes.BackgroundPhase)
    pvd                         =   [];             # pvd file info for paraview
    Output_VTK                  =   true;           # output VTK files in case CartData is an input?

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
    mutable struct DikeParam <: DikeParameters

This mutable structure represents parameters related to a dike in the simulation. It is used to store and manage values related to the dike's properties and behavior.

# Fields

- `Type::String`: Type of the dike.
- `Center::Vector{Float64}`: Center of the dike.
- `T_in_Celsius::Float64`: Temperature of the injected magma in Celsius.
- `W_in::Float64`: Diameter of the dike.
- `H_in::Float64`: Thickness of the dike.
- `AspectRatio::Float64`: Aspect ratio of the dike.
- `SillRadius::Float64`: Radius of the sill.
- `SillArea::Float64`: Horizontal area of the sill.
- `InjectionInterval_year::Float64`: Injection interval in years.
- `SecYear`: Number of seconds in a year.
- `InjectionInterval::Float64`: Injection interval in seconds.
- `nTr_dike::Int64`: Number of tracers in the dike.
- `InjectVol`: Injected volume into the dike.
- `Qrate_km3_yr`: Dike insertion rate in km^3/year.
- `dike_poly`: Polygon representing the dike.
- `dike_inj`: Injection into the dike.

# Examples

```julia
dp = DikeParam(Type="MyDike", Center=[0., -7.0e3], ...)
```
"""
@with_kw mutable struct DikeParam <: DikeParameters
    Type::String                    =   "CylindricalDike_TopAccretion"
    Center::Vector{Float64}         =   [0.; -7.0e3 - 0/2];     # Center of dike 
    Angle::Vector{Float64}          =   [0.0];                  # Angle of dike
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
    BackgroundPhase                 =   1;                      # Background phase  (non-dikes)
    DikePhase                       =   2;                      # Dike phase
    dike_poly                       =   [];                     # polygon with dike
    dike_inj                        =   0.0
    H_ran                           =   5000                    # Zone in which we vary the horizontal location of the dike
    W_ran                           =   2000                    # Zone in which we vary the vertical location of the dike
end


"""
    mutable struct TimeDepProps <: TimeDependentProperties

This mutable structure represents time-dependent properties in the simulation. It is used to store and manage values that change over time.

# Fields

- `Time_vec::Vector{Float64}`: Vector storing the time points.
- `MeltFraction::Vector{Float64}`: Vector storing the melt fraction at each time point.
- `Tav_magma::Vector{Float64}`: Vector storing the average magma temperature at each time point.
- `Tmax::Vector{Float64}`: Vector storing the maximum magma temperature at each time point.

# Examples

```julia
tdp = TimeDepProps(Time_vec=[0., 1., 2.], MeltFraction=[0.1, 0.2, 0.3], ...)
```

# Note:
You can use multiple dispatch on this struct in your user code as long as the new struct 

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
    Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, Tnew_cpu)

Function that injects dikes once in a while
"""
function MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters, Tracers::StructVector, Tnew_cpu)

    if floor(Num.time/Dikes.InjectionInterval)> Dikes.dike_inj      
        Dikes.dike_inj      =   floor(Num.time/Dikes.InjectionInterval)                 # Keeps track on what was injected already
        T_bottom  =   Tnew_cpu[:,1]
        dike      =   Dike(W=Dikes.W_in,H=Dikes.H_in,Type=Dikes.Type,T=Dikes.T_in_Celsius, Center=Dikes.Center[:],  Angle=Dikes.Angle, Phase=Dikes.DikePhase);               # "Reference" dike with given thickness,radius and T
        Tnew_cpu .=   Array(Arrays.T)

        Tracers, Tnew_cpu,Vol,Dikes.dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, Dikes.nTr_dike, dike_poly=Dikes.dike_poly);     # Add dike, move hostrocks
       
        if Num.flux_bottom_BC==false
            # Keep bottom T constant (advection modifies this)
            Tnew_cpu[:,1]   .=  T_bottom
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
           PhasesFromTracers!(Arrays.Phases, Grid, Tracers, BackgroundPhase=Dikes.BackgroundPhase, InterpolationMethod="Constant");    # update phases from grid 

           # Ensure that we keep the initial phase of the area (host rocks are not deformable)
           if Num.keep_init_RockPhases==true
                for i in eachindex(Arrays.Phases)
                    if Arrays.Phases[i] != Dikes.DikePhase
                        Arrays.Phases[i] = Arrays.Phases_init[i]
                    end
                end
           end
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
    MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

Function that prints output to the REPL 
"""
function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    
    return nothing
end

"""
    MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

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
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)

Initialize temperature and phases 
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)
    # Initalize T
    Arrays.T_init      .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm;                # Initial (linear) temperature profile
    
    # Initialize Phases
    return nothing
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::CartData)

Initialize temperature and phases 
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::Union{Nothing,CartData})
    # Initalize T from CartData set
    # NOTE: this almost certainly requires changes if we use GPUs

    if Num.USE_GPU
        @parallel assign!(Arrays.T_init,  CuArray(CartData_input.fields.Temp[:,:,1]))
        
        #Arrays.T_init       .= CuArray(CartData_input.fields.Temp[:,:,1]);

        Arrays.Phases       .= CuArray(CartData_input.fields.Phases[:,:,1]);
        Arrays.Phases_init  .= CuArray(CartData_input.fields.Phases[:,:,1]);
    else
        Arrays.T_init       .= CartData_input.fields.Temp[:,:,1];

        Arrays.Phases       .= CartData_input.fields.Phases[:,:,1];
        Arrays.Phases_init  .= CartData_input.fields.Phases[:,:,1];
    end

    # open pvd file if requested
    if Num.Output_VTK & !isnothing(CartData_input)
        name =  joinpath(Num.SimName,Num.SimName*".pvd")
        Num.pvd = Movie_Paraview(name=name, Initialize=true);
    end

    return nothing
end


"""
    MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::CartData)

Finalize model run
"""
function MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::Union{Nothing,CartData})
    if Num.Output_VTK & !isnothing(CartData_input)
        Movie_Paraview(pvd=Num.pvd, Finalize=true)
    end

    return nothing
end

"""
    MTK_update_Arrays!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters)

Update arrays and structs of the simulation (in case you want to change them during a simulation)
You can use this, for example, to change the size and location of an intruded dike
"""
function MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters)
    return nothing
end

"""
    MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::Union{CartData, Nothing})

Save the output to disk
"""
function MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::Union{CartData, Nothing})

    if mod(Num.it,Num.SaveOutput_steps)==0
        # Save output
        if Num.Output_VTK & !isnothing(CartData_input)
            # add datasets 
            CartData_input = add_2Ddata_CartData(CartData_input, "Temp",            Array(Arrays.Tnew));
            CartData_input = add_2Ddata_CartData(CartData_input, "Phases",          Array(Arrays.Phases));
            CartData_input = add_2Ddata_CartData(CartData_input, "MeltFraction",    Array(Arrays.ϕ));

            # Save output to CartData
            name = joinpath(Num.SimName,Num.SimName*"_$(Num.it)")
            Num.pvd  = Write_Paraview(CartData_input, name, pvd=Num.pvd,time=Num.time/SecYear/1e3);
        end
    end
    return nothing
end

function add_2Ddata_CartData(d::CartData, name::String, data::Array{_T,2})  where _T<:Real
    a = zero(d.x.val)
    a[:,:,1] .= data;
    d = AddField(d, name, a)
    return d
end

"""
    Tracers = MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters)

Updates info on tracers
"""
function MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters)

    if mod(Num.it,10)==0
        # Update T vector on tracers
        update_Tvec!(Tracers, Num.time/SecYear*1e-6)                                    # update T & time vectors on tracers
    end

    return Tracers
end


"""
    Num = Setup_Model_CartData(d, Num, Mat_tup)

Create a MTK model setup from a CartData structure generated with GeophysicalModelGenerator

"""
function Setup_Model_CartData(d, Num, Mat_tup)
    @assert size(d.x)[3] == 1

    x = extrema(d.fields.FlatCrossSection.*1e3)
    z = extrema(d.z.val.*1e3)
    
    Num.W = (x[2]-x[1])
    Num.H = (z[2]-z[1]) 
    Num.Nx = size(d.x)[1]
    Num.Nz = size(d.x)[2]
  
    dx = (x[2]-x[1])/(Num.Nx-1)
    dz = (z[2]-z[1])/(Num.Nx-1)

    # estimate maximum thermal diffusivity from Mat_tup
    κ_max = Num.κ_time
    for mm in Mat_tup
        if hasfield(typeof(mm.Conductivity[1]),:k)
            k = NumValue(mm.Conductivity[1].k)
        else
            k = 3;
        end
        if hasfield(typeof(mm.HeatCapacity[1]),:cp)
            cp = NumValue(mm.HeatCapacity[1].cp)
        else
            cp = 1050;
        end
        if hasfield(typeof(mm.Density[1]),:ρ)
            ρ = NumValue(mm.Density[1].ρ)
        else
            ρ = 2700;
        end
        κ  = k/(cp*ρ)
        if κ>κ_max
            κ_max = κ
        end
    end
    Num.κ_time = κ_max;

    dt = Num.fac_dt*min(dx^2, dz^2)./Num.κ_time/4;   # timestep
    Num.dx = dx;
    Num.dz = dz;
    Num.dt = dt;

    Num.nt = floor(Num.maxTime/dt)
    
    return Num
end


#-----------------------------------------------------------------------------------------
"""
    Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(Mat_tup::Tuple, Num::NumericalParameters, Dikes::DikeParameters; CartData_input=nothing, time_props::TimeDependentProperties = TimeDepProps());

Main routine that performs a 2D or 2D axisymmetric thermal diffusion simulation with injection of dikes.

Parameters
====
- `Mat_tup::Tuple`: Tuple of material properties.
- `Num::NumericalParameters`: Numerical parameters.
- `Dikes::DikeParameters`: Dike parameters.
- `CartData_input::CartData`: Optional input of a CartData structure generated with GeophysicalModelGenerator.
- `time_props::TimeDependentProperties`: Optional input of a `TimeDependentProperties` structure.

Customizable functions
====
There are a few functions that you can overwrite in your user code to customize the simulation:

- `MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)`
- `MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)`
- `MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input)`
- `MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters)`
- `MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::CartData)`
- `MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters, Tracers::StructVector, Tnew_cpu)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)`
- `MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::CartData)`

"""
@views function MTK_GeoParams_2D(Mat_tup::Tuple, Num::NumericalParameters, Dikes::DikeParameters; CartData_input::Union{Nothing,CartData}=nothing, time_props::TimeDependentProperties = TimeDepProps());
    
    # Change parameters based on CartData input
    if !isnothing(CartData_input)
       
        if !hasfield(typeof(CartData_input.fields),:FlatCrossSection)
           error("You should add a Field :FlatCrossSection to your data structure with Data_Cross = AddField(Data_Cross,\"FlatCrossSection\", FlattenCrossSection(Data_Cross))")
        end

        Num = Setup_Model_CartData(CartData_input, Num, Mat_tup)
    end

    # Array & grid initializations ---------------
    Arrays = CreateArrays(Dict( (Num.Nx,  Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0,dϕdT_o=0, R=0, Z=0, P=0),
                                (Num.Nx-1,Num.Nz  )=>(qx=0,Kx=0, Rc=0), 
                                (Num.Nx  ,Num.Nz-1)=>(qz=0,Kz=0 )
                                ))

    # Set up model geometry & initial T structure
    if isnothing(CartData_input)
        Grid                    = CreateGrid(size=(Num.Nx,Num.Nz), extent=(Num.W, Num.H))   
    else
        Grid                    = CreateGrid(size=(Num.Nx,Num.Nz), x=extrema(CartData_input.fields.FlatCrossSection.*1e3), z=extrema(CartData_input.z.val.*1e3))   
    end
    GridArray!(Arrays.R, Arrays.Z, Grid)        
    Arrays.Rc              .=   (Arrays.R[2:end,:] + Arrays.R[1:end-1,:])/2     # center points in x
    # --------------------------------------------

    Tracers                 =   StructArray{Tracer}(undef, 1)                       # Initialize tracers   

    # Update buffer & phases arrays --------------
    if Num.USE_GPU
        # CPU buffers for advection
        Tnew_cpu        =   Matrix{Float64}(undef, Num.Nx, Num.Nz)
        Phi_melt_cpu    =   similar(Tnew_cpu)
        Phases          =   CUDA.ones(Int64,Num.Nx,Num.Nz)
        Phases_init     =   CUDA.ones(Int64,Num.Nx,Num.Nz)
    else
        Tnew_cpu        =   similar(Arrays.T)
        Phi_melt_cpu    =   similar(Arrays.ϕ)
        Phases          =   ones(Int64,Num.Nx,Num.Nz)
        Phases_init     =   ones(Int64,Num.Nx,Num.Nz)
    end
    Arrays = (Arrays..., Phases=Phases, Phases_init=Phases_init);
    
    # Initialize Geotherm and Phases -------------
    if isnothing(CartData_input)
        MTK_initialize!(Arrays, Grid, Num, Tracers, Dikes);
    else
        MTK_initialize!(Arrays, Grid, Num, Tracers, Dikes, CartData_input);
    end
    #@parallel assign!(Arrays.Phases_init, Arrays.Phases)

    
    # check errors 
    unique_Phases = unique(Array(Arrays.Phases));
    phase_specified = []
    for mm in Mat_tup
        push!(phase_specified, mm.Phase)
    end
    for u in unique_Phases
        if !(u in phase_specified)
            error("Properties for Phase $u are not specified in Mat_tup. Please add that")
        end
    end

    if any(isnan.(Arrays.T))
        error("NaNs in T")
    end
    # --------------------------------------------
    

    # Optionally set initial sill in models ------
    if Dikes.Type  == "CylindricalDike_TopAccretion"
        ind = findall( (Arrays.R.<=Dikes.W_in/2) .& (abs.(Arrays.Z.-Dikes.Center[2]) .< Dikes.H_in/2) );
        Arrays.T_init[ind] .= Dikes.T_in_Celsius;
        if Num.advect_polygon==true
            dike              =   Dike(W=Dikes.W_in,H=Dikes.H_in,Type=Dikes.Type,T=Dikes.T_in_Celsius, Center=Dikes.Center[:],  Angle=Dikes.Angle, Phase=Dikes.DikePhase);               # "Reference" dike with given thickness,radius and T
            Dikes.dike_poly   =   CreateDikePolygon(dike);
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

    end
    # --------------------------------------------

    for Num.it = 1:Num.nt   # Time loop
        Num.time  += Num.dt;                                     # Keep track of evolved time

        # Add new dike every X years -----------------
        Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, Tnew_cpu)
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
        
        # Update info on tracers ---------------------
        Tracers = MTK_updateTracers(Grid, Arrays, Tracers, Dikes, time_props, Num);
        # --------------------------------------------

        # Update time-dependent properties -----------
        MTK_update_TimeDepProps!(time_props, Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

        # Visualize results --------------------------
        MTK_visualize_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------
          
        # Save output to disk once in a while --------
        MTK_save_output(Grid, Arrays, Tracers, Dikes, time_props, Num, CartData_input);
        # --------------------------------------------

        # Optionally update arrays and structs (such as T or Dike) -------
        MTK_update_ArraysStructs!(Arrays, Grid, Dikes, Num)
        # --------------------------------------------
        
        # Display output -----------------------------
        MTK_print_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

    end

    # Finalize simulation ------------------------
    MTK_finalize!(Arrays, Grid, Num, Tracers, Dikes, CartData_input);
    # --------------------------------------------


    return Grid, Arrays, Tracers, Dikes, time_props
end # end of main function


end
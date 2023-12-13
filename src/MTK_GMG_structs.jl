
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
    Ny::Int64                   =   0
    Nz::Int64                   =   201
    dim::Int64                  =   length([Nx, Ny, Nz].>0)
    W::Float64                  =   20e3
    L::Float64                  =   0
    H::Float64                  =   20e3
    dx::Float64                 =   W/(Nx-1)
    dy::Float64                 =   L/(Ny-1)
    dz::Float64                 =   H/(Nz-1)        # grid spacing in z
    Tsurface_Celcius::Float64   =   0               # Surface T in celcius
    Geotherm::Float64           =   40/1e3          # in K/m
    maxTime_Myrs::Float64       =   1.5             # maximum timestep 
    SecYear::Float64            =   3600*24*365.25;
    maxTime::Float64            =   maxTime_Myrs*SecYear*1e6 # maximum timestep  in seconds
    flux_bottom_BC::Bool        =   false           # flux bottom BC?
    flux_bottom::Float64        =   167e-3          # Flux in W/m2 in case flux_bottom_BC=true
    plot_tracers::Bool          =   true            # adds passive tracers to the plot
    advect_polygon::Bool        =   false           # adds a polygon around the intrusion area
    axisymmetric::Bool          =   true            # axisymmetric (if true) of 2D geometry?
    κ_time::Float64             =   3.3/(1000*2700) # κ to determine the stable timestep 
    fac_dt::Float64             =   0.4;            # prefactor with which dt is multiplied   
    Δ::Vector{Float64}          =   [dx, dy, dz];                   # grid spacing
    Δmin::Float64               =   minimum(Δ[Δ.>0]);               # minimum grid spacing
    dt::Float64                 =   fac_dt*(Δmin^2)./κ_time/4;   # timestep
    time::Float64               =   0.0;            # current time          
    nt::Int64                   =   floor(maxTime/dt);
    it::Int64                   =   0;              # current iteration
    ω::Float64                  =   0.8;            # relaxation parameter for nonlinear iterations    
    max_iter::Int64             =   5000;           # max. number of nonlinear iterations        
    verbose::Bool               =   false;    
    convergence::Float64        =   1e-5;           # nonlinear convergence criteria      
    USE_GPU                     =   false;
    keep_init_RockPhases::Bool  =   true;           # keep initial rock phases (if false, all phases are initialized as Dikes.BackgroundPhase)
    pvd                         =   [];             # pvd file info for paraview
    Output_VTK                  =   true;           # output VTK files in case CartData is an input?
    SaveOutput_steps::Int64     =   1e3;            # saves output every x steps 
    CreateFig_steps::Int64      =   500;            # Create a figure every X steps

    AddRandomSills::Bool        =   false;          # Add random sills/dikes to the model?
    RandomSills_timestep::Int64 =   10;             # After how many timesteps do we add a new sill/dike?


    # parts that can be removed @ some stage
    deactivate_La_at_depth::Bool=   false           # deactivate latent heating @ the bottom of the model box?
    deactivationDepth::Float64  =   -15e3           # deactivation depth
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

- `H_ran`:    Zone in which we vary the vertical location of the dike (if we add random dikes)
- `L_ran`:    Zone in which we vary the horizontal (x) location of the dike (if we add random dikes)
- `W_ran`:    Zone in which we vary the horizontal (y) location of the dike (if we add random dikes)

- `Dip_ran`:  maximum variation of dip (if we add random dikes)
- `Strike_ran`: maximum variation of strike (if we add random dikes)


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
    InjectVol::Float64              =   0.0;                    # injected volume
    Qrate_km3_yr::Float64           =   0.0;                    # Dikes insertion rate
    BackgroundPhase::Int64           =   1; # Background phase  (non-dikes)
    DikePhase::Int64                       =   2; # Dike phase
    dike_poly::Vector               =   [];                     # polygon with dike
    dike_inj::Float64               =   0.0

    H_ran::Float64                  =   5000.0                    # Zone in which we vary the horizontal location of the dike
    L_ran::Float64                  =   2000.0                    # Zone in which we vary the horizontal location of the dike
    W_ran::Float64                  =   2000.0                   # Zone in which we vary the vertical location of the dike
    Dip_ran::Float64                =   30.0;                     # maximum variation of dip
    Strike_ran::Float64             =   90.0;                     # maximum variation of strike
    SillsAbove::Float64             =   -15e3;                    # Sills above this depth

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

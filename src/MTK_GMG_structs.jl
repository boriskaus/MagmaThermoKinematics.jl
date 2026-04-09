using GeophysicalModelGenerator

"""
This mutable structure represents numerical parameters in the program. It is used to store and manage numerical values that are used throughout the program.
    mutable struct NumParam <: NumericalParameters

# Fields

- `SimName::String`: Name of the simulation.
- `FigTitle`: Title of the figure.
- `Nx::Int64`: Number of grid points in the x direction.
- `Nz::Int64`: Number of grid points in the z direction.
- `W::AbstractFloat`: Width of the domain.
- `H::AbstractFloat`: Height of the domain.
- `dx::AbstractFloat`: Grid spacing in the x direction.
- `dz::AbstractFloat`: Grid spacing in the z direction.
- `Tsurface_Celcius::AbstractFloat`: Surface temperature in Celsius.
- `Geotherm::AbstractFloat`: Geothermal gradient in K/m.
- `maxTime_Myrs::AbstractFloat`: Maximum simulation time in Myrs.
- `SecYear::AbstractFloat`: Number of seconds in a year.
- `maxTime::AbstractFloat`: Maximum simulation time in seconds.
- `SaveOutput_steps::Int64`: Number of steps between output saves.
- `CreateFig_steps::Int64`: Number of steps between figure creations.
- `flux_bottom_BC::Bool`: Whether to apply a flux at the bottom boundary.
- `flux_bottom::AbstractFloat`: Flux at the bottom boundary in W/m^2.
- `plot_tracers::Bool`: Whether to plot tracers.
- `advect_polygon::Bool`: Whether to advect a polygon around the intrusion area.
- `axisymmetric::Bool`: Whether the simulation is axisymmetric.
- `κ_time::AbstractFloat`: Thermal diffusivity.
- `fac_dt::AbstractFloat`: Factor to multiply the time step by.
- `dt::AbstractFloat`: Time step.
- `time::AbstractFloat`: Current time.
- `nt::Int64`: Total number of time steps.
- `it::Int64`: Current iteration.
- `ω::AbstractFloat`: Relaxation parameter for nonlinear iterations.
- `max_iter::Int64`: Maximum number of nonlinear iterations.
- `verbose::Bool`: Whether to print verbose output.
- `convergence::AbstractFloat`: Convergence criterion for nonlinear iterations.
- `deactivate_La_at_depth::Bool`: Whether to deactivate latent heating at the bottom of the model box.
- `deactivationDepth::AbstractFloat`: Depth at which to deactivate latent heating.
- `USE_GPU`: Whether to use a GPU.
- `AnalyticalInitialGeo::Bool`: Whether to use an analytical initial geotherm.
- `qs_anal::AbstractFloat`: Analytical surface heat flux.
- `qm_anal::AbstractFloat`: Analytical mantle heat flux.
- `hr_anal::AbstractFloat`: Analytical radiogenic heat production.
- `k_anal::AbstractFloat`: Analytical thermal conductivity.
- `InitialEllipse::Bool`: Whether to initialize with an ellipse.
- `a_init::AbstractFloat`: Semi-major axis of initial ellipse.
- `b_init::AbstractFloat`: Semi-minor axis of initial ellipse.
- `TrackTracersOnGrid::Bool`: Whether to track tracers on the grid.

# Examples

```julia
np = NumParam(SimName="MySim", Nx=101, Nz=101, ...)
```

"""
@with_kw mutable struct NumParam <: NumericalParameters
    SimName::String                 =   "Zassy_UCLA_ellipticalIntrusion"    # name of simulation
    FigTitle                        =   "UCLA setup"
    Nx::Int64                       =   201
    Ny::Int64                       =   0
    Nz::Int64                       =   201
    dim::Int64                      =   length([Nx, Ny, Nz].>0)
    W::AbstractFloat                =   20e3
    L::AbstractFloat                =   0
    H::AbstractFloat                =   20e3
    dx::AbstractFloat               =   W/(Nx-1)
    dy::AbstractFloat               =   L/(Ny-1)
    dz::AbstractFloat               =   H/(Nz-1)        # grid spacing in z
    Tsurface_Celcius::AbstractFloat =   0               # Surface T in celcius
    Geotherm::AbstractFloat         =   40/1e3          # in K/m
    maxTime_Myrs::AbstractFloat     =   1.5             # maximum timestep
    SecYear::AbstractFloat          =   3600*24*365.25;
    maxTime::AbstractFloat          =   maxTime_Myrs*SecYear*1e6 # maximum timestep  in seconds
    flux_bottom_BC::Bool            =   false           # flux bottom BC?
    flux_bottom::AbstractFloat      =   167e-3          # Flux in W/m2 in case flux_bottom_BC=true
    plot_tracers::Bool              =   true            # adds passive tracers to the plot
    advect_polygon::Bool            =   false           # adds a polygon around the intrusion area
    axisymmetric::Bool              =   false           # axisymmetric (if true) of 2D geometry?
    κ_time::AbstractFloat           =   3.3/(1000*2700) # κ to determine the stable timestep
    fac_dt::AbstractFloat           =   0.4;            # prefactor with which dt is multiplied
    Δ::Vector{AbstractFloat}        =   [dx, dy, dz];                   # grid spacing
    Δmin::AbstractFloat             =   minimum(Δ[Δ.>0]);               # minimum grid spacing
    dt::AbstractFloat               =   fac_dt*(Δmin^2)./κ_time/4;   # timestep
    time::AbstractFloat             =   0.0;            # current time
    nt::Int64                       =   floor(maxTime/dt);
    it::Int64                       =   0;              # current iteration
    ω::AbstractFloat                =   0.8;            # relaxation parameter for nonlinear iterations
    max_iter::Int64                 =   5000;           # max. number of nonlinear iterations
    verbose::Bool                   =   false;
    convergence::AbstractFloat      =   1e-5;           # nonlinear convergence criteria
    USE_GPU::Bool                   =   false;
    keep_init_RockPhases::Bool      =   true;           # keep initial rock phases (if false, all phases are initialized as Dikes.BackgroundPhase)
    pvd::Union{Nothing,GeophysicalModelGenerator.WriteVTK.CollectionFile}     =   nothing;             # pvd file info for paraview
    Output_VTK::Bool                =   true;           # output VTK files in case CartData is an input?
    SaveOutput_steps::Int64         =   1e3;            # saves output every x steps
    CreateFig_steps::Int64          =   500;            # Create a figure every X steps

    AddRandomSills::Bool            =   false;          # Add random sills/dikes to the model?
    RandomSills_timestep::Int64     =   10;             # After how many timesteps do we add a new sill/dike?

    # parts that can be removed @ some stage
    deactivate_La_at_depth::Bool    =   false           # deactivate latent heating @ the bottom of the model box?
    deactivationDepth::AbstractFloat=   -15e3           # deactivation depth
    AnalyticalInitialGeo::Bool      =   false;
    qs_anal::AbstractFloat          =   170e-3;
    qm_anal::AbstractFloat          =   167e-3;
    hr_anal::AbstractFloat          =   10e3;
    k_anal::AbstractFloat           =   3.35;
    InitialEllipse::Bool            =   false;
    a_init::AbstractFloat           =   2.5e3;
    b_init::AbstractFloat           =   1.5e3;
    TrackTracersOnGrid::Bool        =   true;
end

"""
    mutable struct DikeParam <: DikeParameters

This mutable structure represents parameters related to a dike in the simulation. It is used to store and manage values related to the dike's properties and behavior.

# Fields

- `Type::String`: Type of the dike.
- `Center::Vector{AbstractFloat}`: Center of the dike.
- `T_in_Celsius::AbstractFloat`: Temperature of the injected magma in Celsius.
- `W_in::AbstractFloat`: Diameter of the dike.
- `H_in::AbstractFloat`: Thickness of the dike.
- `AspectRatio::AbstractFloat`: Aspect ratio of the dike.
- `SillRadius::AbstractFloat`: Radius of the sill.
- `SillArea::AbstractFloat`: Horizontal area of the sill.
- `InjectionInterval_year::AbstractFloat`: Injection interval in years.
- `SecYear`: Number of seconds in a year.
- `InjectionInterval::AbstractFloat`: Injection interval in seconds.
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
    Center::Vector{AbstractFloat}         =   [0.; -7.0e3 - 0/2];     # Center of dike
    Angle::Vector{AbstractFloat}          =   [0.0];                  # Angle of dike
    T_in_Celsius::AbstractFloat           =   1000;                   # Temperature of injected magma
    W_in::AbstractFloat                   =   20e3                    # Diameter of dike
    H_in::AbstractFloat                   =   74.6269                 # Thickness
    AspectRatio::AbstractFloat            =   H_in/W_in;              # Aspect ratio
    SillRadius::AbstractFloat             =   W_in/2                  # Sill radius
    SillArea::AbstractFloat               =   pi*SillRadius^2         # Horizontal area  of sill
    InjectionInterval_year::AbstractFloat =   10e3;                   # Injection interval [years]
    SecYear                         =   3600*24*365.25;         # s/year
    InjectionInterval::AbstractFloat      =   InjectionInterval_year*SecYear;           # Injection interval [s]
    nTr_dike::Int64                 =   300                     # Number of tracers
    InjectVol::AbstractFloat              =   0.0;                    # injected volume
    Qrate_km3_yr::AbstractFloat           =   0.0;                    # Dikes insertion rate
    BackgroundPhase::Int64          =   1; # Background phase  (non-dikes)
    DikePhase::Int64                =   2; # Dike phase
    dike_poly::Vector               =   [];                     # polygon with dike
    dike_inj::AbstractFloat               =   0.0

    H_ran::AbstractFloat                  =   5000.0                    # Zone in which we vary the horizontal location of the dike
    L_ran::AbstractFloat                  =   2000.0                    # Zone in which we vary the horizontal location of the dike
    W_ran::AbstractFloat                  =   2000.0                   # Zone in which we vary the vertical location of the dike
    Dip_ran::AbstractFloat                =   30.0;                     # maximum variation of dip
    Strike_ran::AbstractFloat             =   90.0;                     # maximum variation of strike
    SillsAbove::AbstractFloat             =   -15e3;                    # Sills above this depth

end


"""
    mutable struct TimeDepProps <: TimeDependentProperties

This mutable structure represents time-dependent properties in the simulation. It is used to store and manage values that change over time.

# Fields

- `Time_vec::Vector{AbstractFloat}`: Vector storing the time points.
- `MeltFraction::Vector{AbstractFloat}`: Vector storing the melt fraction at each time point.
- `Tav_magma::Vector{AbstractFloat}`: Vector storing the average magma temperature at each time point.
- `Tmax::Vector{AbstractFloat}`: Vector storing the maximum magma temperature at each time point.

# Examples

```julia
tdp = TimeDepProps(Time_vec=[0., 1., 2.], MeltFraction=[0.1, 0.2, 0.3], ...)
```

# Note:
You can use multiple dispatch on this struct in your user code as long as the new struct

"""
@with_kw mutable struct TimeDepProps <: TimeDependentProperties
    Time_vec::Vector{AbstractFloat}  = [];        # Center of dike
    MeltFraction::Vector{AbstractFloat} = [];     # Melt fraction over time
    Tav_magma::Vector{AbstractFloat} = [];        # Average magma
    Tmax::Vector{AbstractFloat} = [];             # Max magma temperature
end

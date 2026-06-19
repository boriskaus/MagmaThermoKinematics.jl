using GeophysicalModelGenerator

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
    SimName::String                 =   "Zassy_UCLA_ellipticalIntrusion"    # name of simulation
    FigTitle                        =   "UCLA setup"
    Nx::Int64                       =   201
    Ny::Int64                       =   0
    Nz::Int64                       =   201
    dim::Int64                      =   length([Nx, Ny, Nz].>0)
    W::Float64                      =   20e3
    L::Float64                      =   0
    H::Float64                      =   20e3
    dx::Float64                     =   W/(Nx-1)
    dy::Float64                     =   L/(Ny-1)
    dz::Float64                     =   H/(Nz-1)        # grid spacing in z
    Tsurface_Celcius::Float64       =   0               # Surface T in celcius
    Geotherm::Float64               =   40/1e3          # in K/m
    maxTime_Myrs::Float64           =   1.5             # maximum timestep
    SecYear::Float64                =   3600*24*365.25;
    maxTime::Float64                =   maxTime_Myrs*SecYear*1e6 # maximum timestep  in seconds
    flux_bottom_BC::Bool            =   false           # flux bottom BC?
    flux_bottom::Float64            =   167e-3          # Flux in W/m2 in case flux_bottom_BC=true
    plot_tracers::Bool              =   true            # adds passive tracers to the plot
    advect_polygon::Bool            =   false           # adds a polygon around the intrusion area
    axisymmetric::Bool              =   false           # axisymmetric (if true) of 2D geometry?
    κ_time::Float64                 =   3.3/(1000*2700) # κ to determine the stable timestep
    fac_dt::Float64                 =   0.4;            # prefactor with which dt is multiplied
    Δ::Vector{Float64}              =   [dx, dy, dz];                   # grid spacing
    Δmin::Float64                   =   minimum(Δ[Δ.>0]);               # minimum grid spacing
    dt::Float64                     =   fac_dt*(Δmin^2)./κ_time/4;   # timestep
    time::Float64                   =   0.0;            # current time
    nt::Int64                       =   floor(maxTime/dt);
    it::Int64                       =   0;              # current iteration
    ω::Float64                      =   0.8;            # relaxation parameter for nonlinear iterations
    max_iter::Int64                 =   5000;           # max. number of nonlinear iterations
    verbose::Bool                   =   false;
    convergence::Float64            =   1e-5;           # nonlinear convergence criteria
    USE_GPU::Bool                   =   false;
    keep_init_RockPhases::Bool      =   true;           # keep initial rock phases (if false, all phases are initialized as Dikes.BackgroundPhase)
    deform_hostrock::Bool           =   false;          # advect the phase field with the host-rock displacement (injection inflation + eruption deflation) instead of pinning it. Auto-enabled when a free surface is active so the surface, host rock & sills move together.
    SeedHostTracers::Bool           =   false;          # seed passive host-rock tracers at init (carry initial layering + accumulate T-t paths; advected by injection/deflation, frozen on eruption). Opt-in; phase field still handled by advect_phases!.
    HostTracersDir::Int64           =   2;              # per-direction host-tracer count per cell when SeedHostTracers=true
    pvd::Union{Nothing,GeophysicalModelGenerator.WriteVTK.CollectionFile}     =   nothing;             # pvd file info for paraview
    Output_VTK::Bool                =   true;           # output VTK files in case CartData is an input?
    SaveOutput_steps::Int64         =   1e3;            # saves output every x steps
    CreateFig_steps::Int64          =   500;            # Create a figure every X steps

    AddRandomSills::Bool            =   false;          # Add random sills/dikes to the model?
    RandomSills_timestep::Int64     =   10;             # After how many timesteps do we add a new sill/dike?

    # parts that can be removed @ some stage
    deactivate_La_at_depth::Bool    =   false           # deactivate latent heating @ the bottom of the model box?
    deactivationDepth::Float64      =   -15e3           # deactivation depth
    AnalyticalInitialGeo::Bool      =   false;
    qs_anal::Float64                =   170e-3;
    qm_anal::Float64                =   167e-3;
    hr_anal::Float64                =   10e3;
    k_anal::Float64                 =   3.35;
    InitialEllipse::Bool            =   false;
    a_init::Float64                 =   2.5e3;
    b_init::Float64                 =   1.5e3;
    TrackTracersOnGrid::Bool        =   true;
    TracerFloatType::DataType       =   Float32;    # float type for Tracer time_vec/T_vec (Float32 saves memory)
end

"""
        mutable struct SillParams <: SillParameters

This mutable structure represents parameters for sill injection in the MTK/GMG
workflow. The geometric and mechanical definition of
the intrusion is stored directly in `sill` as an `InjectSills.AbstractSill`
object (for example `InjectSills.EllipticalIntrusion` or
`InjectSills.CylindricalDikeTopAccretion`).

# Fields

- `sill::Union{Nothing, InjectSills.AbstractSill}`: Full InjectSills object that
    defines sill type, center, orientation, width, thickness, and any additional
    model parameters.
- `T_in_Celsius::Float64`: Temperature of injected magma in Celsius.
- `InjectionInterval_year::Float64`: Injection interval in years.
- `SecYear`: Number of seconds in a year.
- `InjectionInterval::Float64`: Injection interval in seconds.
- `nTr_dike::Int64`: Number of tracers inserted per injection event.
- `InjectVol::Float64`: Cumulative injected volume.
- `Qrate_km3_yr::Float64`: Time-averaged emplacement rate in km^3/yr.
- `BackgroundPhase::Int64`: Host-rock phase index.
- `SillPhase::Int64`: Injected sill phase index.
- `sill_poly::Vector`: Optional polygon representation used for advection/plotting.
- `sill_inj::Float64`: Counter that tracks how many sill injections already occurred.
- `H_ran::Float64`: Randomization range for vertical placement.
- `L_ran::Float64`: Randomization range for horizontal x placement.
- `W_ran::Float64`: Randomization range for horizontal y placement.
- `Dip_ran::Float64`: Maximum dip perturbation for randomized injections.
- `Strike_ran::Float64`: Maximum strike perturbation for randomized injections.
- `SillsAbove::Float64`: Depth threshold above which intrusions are treated as sills.

# Example

```julia
sp = SillParams(
        sill = InjectSills.EllipticalIntrusion(...),
        T_in_Celsius = 1000.0,
        InjectionInterval_year = 10e3,
)
```

"""
@with_kw mutable struct SillParams <: SillParameters
    sill::Union{Nothing, InjectSills.AbstractSill} = nothing    # InjectSills.jl sill object (used when Type="InjectSills")
    T_in_Celsius::Float64           =   1000;                   # Temperature of injected magma
    InjectionInterval_year::Float64 =   10e3;                   # Injection interval [years]
    SecYear                         =   3600*24*365.25;         # s/year
    InjectionInterval::Float64      =   InjectionInterval_year*SecYear;           # Injection interval [s]
    nTr_dike::Int64                 =   300                     # Number of tracers
    InjectVol::Float64              =   0.0;                    # injected volume
    Qrate_km3_yr::Float64           =   0.0;                    # Dikes insertion rate
    BackgroundPhase::Int64          =   1;                      # Background phase  (non-sills)
    SillPhase::Int64                =   2;                      # Sill phase
    sill_poly::Vector               =   [];                     # polygon with sill
    sill_inj::Float64               =   0.0

    H_ran::Float64                  =   5000.0                  # Zone in which we vary the horizontal location of the sill
    L_ran::Float64                  =   2000.0                  # Zone in which we vary the horizontal location of the dike
    W_ran::Float64                  =   2000.0                  # Zone in which we vary the vertical location of the dike
    Dip_ran::Float64                =   30.0;                   # maximum variation of dip
    Strike_ran::Float64             =   90.0;                   # maximum variation of strike
    SillsAbove::Float64             =   -15e3;                  # Sills above this depth
end

"""
    mutable struct EruptionParams <: EruptionParameters

Parameters controlling eruptions. An eruption is triggered when the volume of
*eruptible* magma (cells whose melt fraction exceeds `ϕ_erupt`) reaches the
critical volume `V_crit`. The eruption then removes a fraction
`erupt_efficiency` of that eruptible melt and deflates the chamber with a
negative ("withdrawal") Mogi source, keeping a running tally of erupted volume
and eruption times.

# Fields
- `erupt::Bool`: master switch (eruptions only happen when `true`).
- `ϕ_erupt::Float64`: melt fraction above which magma is considered eruptible (default 0.5).
- `EruptAbove::Float64`: depth cap on eruptibility — only cells at elevation `z ≥ EruptAbove` (i.e. shallower than this floor) can be eruptible; deeper cells are excluded even if `ϕ > ϕ_erupt`. This keeps deep background melt (e.g. partial melt of the host rock under a hot geotherm at the base of the domain) from being counted as eruptible chamber and triggering spurious eruptions. Default `-Inf` (no depth restriction). Set it to roughly the base of the magmatic system (e.g. the same value as `SillParams.SillsAbove`).
- `V_crit::Float64`: critical eruptible volume that triggers an eruption [m³] (in 2D this is a per-unit-depth volume = area·1 m).
- `erupt_efficiency::Float64`: fraction (0–1) of the eruptible melt removed per eruption.
- `ΔP::Float64`: magnitude of the deflation overpressure for the negative-Mogi source [Pa] (applied with a negative sign).
- `G::Float64`, `ν::Float64`: elastic moduli of the host rock for the deflation source.
- `T_min::Float64`: floor temperature; eruptive cooling never drops a cell below this value. This guards the linearized thermal extraction `ΔT = η·ϕ/(dϕ/dT)`, which becomes unbounded where the melt curve is flat (`dϕ/dT → 0`, i.e. near melt saturation) and would otherwise push `T` to unphysical values that destabilize the diffusion solver.
- `out_of_plane_3D::Bool`: in 2D, lift the per-unit-depth eruptible volume to a true 3D volume by assuming a Gaussian out-of-plane (y) distribution of the melt (effective out-of-plane length `√(2π)·σ`, with `σ` the melt-weighted horizontal half-width of the eruptible region). This keeps the eruptible/erupted volumes in km³ — the same convention as `V_crit` and the injected volume — so the trigger comparison is dimensionally consistent. No effect in 3D. Default `true`.
- `deflate::Bool`: also apply the kinematic chamber deflation (host-rock subsidence) in addition to the thermal melt removal.
- `deflate_percell::Bool`: deflation source model. When `false` (default) a single region-scale negative-Mogi source is used (melt-weighted centroid, radius from the bulk eruptible volume) — `O(N_grid)` per eruption and spatially symmetric. When `true`, the chamber deflation is instead a **superposition of one negative-Mogi source per eruptible cell**, each sized to the melt withdrawn from that cell (`ΔV_i = erupt_efficiency·ϕ_i·V_cell`), so the subsidence follows the (irregular) melt distribution and the total subsidence volume equals the booked erupted volume `η·Ve`. The per-cell path costs `N_eruptible × N_grid` source evaluations per eruption (`O(N²)` as the chamber grows with the grid), so it is only viable on small grids — at production resolution it stalls; prefer the default region-scale source there.
- `n_eruptions::Int64`, `erupted_volume::Float64`: cumulative bookkeeping.
- `eruption_times::Vector{Float64}`, `eruption_volumes::Vector{Float64}`: per-event record (time [s], volume [m³]).
"""
@with_kw mutable struct EruptionParams <: EruptionParameters
    erupt::Bool                       = false        # enable eruptions
    ϕ_erupt::Float64                  = 0.5          # melt fraction above which magma is "eruptible"
    EruptAbove::Float64               = -Inf         # depth cap: only cells with z ≥ EruptAbove are eruptible (excludes deep geotherm melt). Default -Inf = no cap.
    V_crit_km3::Float64               = 10.0         # critical eruptible volume [km³] (convenience)
    V_crit::Float64                   = V_crit_km3*1e9   # critical eruptible volume [m³]
    erupt_efficiency::Float64         = 0.5          # fraction of eruptible melt removed per eruption (0–1)
    ΔP::Float64                       = 20e6         # deflation overpressure magnitude [Pa] (negative-Mogi)
    G::Float64                        = 10e9         # host-rock shear modulus [Pa]
    ν::Float64                        = 0.25         # host-rock Poisson ratio
    T_min::Float64                    = 0.0          # floor T: eruptive cooling never drops a cell below this (guards ΔT = η·ϕ/dϕdT when dϕdT→0)
    out_of_plane_3D::Bool             = true         # 2D: lift per-unit-depth eruptible volume to a 3D volume via a Gaussian out-of-plane profile (km³, comparable to V_crit). No-op in 3D.
    deflate::Bool                     = true         # also apply the kinematic chamber deflation (host-rock subsidence)
    deflate_percell::Bool             = false        # deflation source: false (default) ⇒ single region-scale negative-Mogi source (O(N_grid), spatially symmetric); true ⇒ per-cell superposition (irregular, ΔV_i=η·ϕ_i·Vcell) but O(N_eruptible × N_grid) per eruption — only viable on small grids
    # --- bookkeeping (filled during the run) ---
    n_eruptions::Int64                = 0
    erupted_volume::Float64           = 0.0          # cumulative erupted melt volume [m³]
    eruption_times::Vector{Float64}   = Float64[]    # time [s] of each eruption
    eruption_volumes::Vector{Float64} = Float64[]    # erupted volume [m³] of each eruption
end

"""
    mutable struct FreeSurfaceParams <: FreeSurfaceParameters

Parameters controlling the kinematic sticky-air free surface (issue 4). The
surface is tracked as a topography `z_surf` on the fixed grid (one elevation per
surface column). Cells above the topography are treated as "air" (temperature
`Tair`, melt fraction 0, phase `air_phase`); the surface is advected vertically
by the host-rock displacement produced by sill injection (inflation) and
eruption deflation.

# Fields
- `free_surface::Bool`: master switch (the surface only moves / air is only stamped when `true`).
- `air_phase::Int64`: phase index assigned to air cells.
- `Tair::Float64`: temperature assigned to air cells [°C].
- `z0::Float64`: initial flat surface elevation [m] used to allocate `z_surf` on the first step (when no `topography` is given).
- `topography::Union{Nothing,Function,AbstractArray}`: optional non-flat initial topography used to build `z_surf` on the first step. Either an array of per-column elevations [m] (length `Nx` in 2D, size `Nx×Ny` in 3D) or a function of the column coordinates (`f(x)` in 2D, `f(x,y)` in 3D). When `nothing` (default) the surface starts flat at `z0`. See [`init_free_surface`](@ref).
- `z_surf::Union{Nothing,Array{Float64}}`: topography (allocated on first use; length `Nx` in 2D, `Nx×Ny` in 3D).
- `_last_inj::Float64`: internal — injection counter already accounted for by the surface (avoids double-counting an injection).
"""
@with_kw mutable struct FreeSurfaceParams <: FreeSurfaceParameters
    free_surface::Bool                = false        # enable the moving free surface
    air_phase::Int64                  = 0            # phase index of air cells
    Tair::Float64                     = 0.0          # temperature of air cells [°C]
    z0::Float64                       = 0.0          # initial flat surface elevation [m] (used when topography===nothing)
    topography::Union{Nothing,Function,AbstractArray} = nothing  # optional non-flat initial topography (array or f(x[,y])); flat z0 when nothing
    z_surf::Union{Nothing,Array{Float64}} = nothing  # topography (allocated on first use)
    _last_inj::Float64                = -1.0         # internal: injection counter seen by the surface
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

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
- `Tsurface_Celsius::Float64`: Surface temperature in Celsius.
- `sticky_air_BC::Bool`: hold every cell of phase `AirPhase` at `Tsurface_Celsius` at the end of each time step (default `false`). The usual alternative, a very large air conductivity, makes the air the fastest-diffusing phase and so shrinks the time step of the whole model (see `bc_sticky_air!`).
- `AirPhase::Int64`: phase number of the air (default 0), used by `sticky_air_BC`.
- `Geotherm::Float64`: Geothermal gradient in K/m.
- `maxTime_Myrs::Float64`: Maximum simulation time in Myrs.
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
- `out_of_plane::Bool`: heat loss along the out-of-plane direction of a planar 2D model (default `false`), `-factor K (T - Tbg)/L²(z)` with `L²(z)` the horizontal variance of the anomaly `T - Tbg` (`Arrays.Tbg`, by default the initial temperature) at each depth.
- `out_of_plane_factor::Float64`, `out_of_plane_smooth::Int64`, `out_of_plane_Lmin::Float64`: prefactor, averaging half-width in depth cells and lower bound [m] of the length scale.
- `ω::Float64`: Relaxation parameter for nonlinear iterations (default 0.5; values ≥ 0.7 may not converge at high resolution).
- `seed::Union{Nothing, Int64}`: seed of the random number generator at the start of a simulation (reproducible random sills); `nothing` leaves the generator unchanged.
- `print_injections::Bool`: print a line for every injected dike/sill (default true).
- `Output_VTK::Bool`: write VTK output; the output directory `SimName` is only created if this is true.
- `solver::Symbol`: `:explicit` (default) keeps the existing explicit step. `:implicit_enthalpy` solves a backward-Euler step of the enthalpy balance, linearised with Newton iterations and solved with an accelerated pseudo-transient iteration (see `apt_diffusion_solve!`); the time step factor `fac_dt` can then be much larger than 1. `:apt` is an alias.
- `apt_tol`, `apt_max_iter`, `apt_check`: tolerance, iteration limit and check interval of the pseudo-transient iterations (implicit enthalpy solver).
- `max_iter::Int64`: Maximum number of nonlinear iterations.
- `verbose::Bool`: Whether to print verbose output.
- `convergence::Float64`: Convergence criterion for nonlinear iterations.
- `deactivate_La_at_depth::Bool`: Whether to deactivate latent heating at the bottom of the model box.
- `deactivationDepth::Float64`: Depth at which to deactivate latent heating.
- `backend`: KernelAbstractions backend on which the model arrays live: `CPU()` (default), or e.g. `CUDABackend()` once CUDA.jl is loaded.
- `FloatType`: Element type of the model arrays (default `Float64`).
- `AnalyticalInitialGeo::Bool`: Whether to use an analytical initial geotherm.
- `qs_anal::Float64`: Analytical surface heat flux.
- `qm_anal::Float64`: Analytical mantle heat flux.
- `hr_anal::Float64`: Analytical radiogenic heat production.
- `k_anal::Float64`: Analytical thermal conductivity.
- `InitialEllipse::Bool`: Whether to initialize with an ellipse.
- `a_init::Float64`: Semi-major axis of initial ellipse.
- `b_init::Float64`: Semi-minor axis of initial ellipse.
- `TrackTracersOnGrid::Bool`: Whether to track tracers on the grid.
- `TtPath_steps::Int64`: Number of time steps between samples of the tracer temperature–time paths (`time_vec`, `T_vec`).

# Examples

```julia
np = NumParam(SimName="MySim", Nx=101, Nz=101, ...)
```

"""
@with_kw mutable struct NumParam <: NumericalParameters
    SimName::String = "Zassy_UCLA_ellipticalIntrusion"    # name of simulation
    FigTitle = "UCLA setup"
    Nx::Int64 = 201
    Ny::Int64 = 0
    Nz::Int64 = 201
    dim::Int64 = count(>(0), (Nx, Ny, Nz))
    W::Float64 = 20.0e3
    L::Float64 = 0
    H::Float64 = 20.0e3
    dx::Float64 = W / (Nx - 1)
    dy::Float64 = L / (Ny - 1)
    dz::Float64 = H / (Nz - 1)        # grid spacing in z
    Tsurface_Celsius::Float64 = 0               # Surface T in Celsius
    sticky_air_BC::Bool = false           # hold the air phase at `Tsurface_Celsius` every time step?
    AirPhase::Int64 = 0               # phase number of the air
    Geotherm::Float64 = 40 / 1.0e3          # in K/m
    maxTime_Myrs::Float64 = 1.5             # maximum timestep
    maxTime::Float64 = maxTime_Myrs * SecYear * 1.0e6 # maximum timestep  in seconds
    flux_bottom_BC::Bool = false           # flux bottom BC?
    flux_bottom::Float64 = 167.0e-3          # Flux in W/m2 in case flux_bottom_BC=true
    plot_tracers::Bool = true            # adds passive tracers to the plot
    advect_polygon::Bool = false           # adds a polygon around the intrusion area
    axisymmetric::Bool = false           # axisymmetric (if true) of 2D geometry?
    κ_time::Float64 = 3.3 / (1000 * 2700) # κ to determine the stable timestep
    fac_dt::Float64 = 0.4             # prefactor with which dt is multiplied
    Δ::Vector{Float64} = [dx, dy, dz]                    # grid spacing
    Δmin::Float64 = minimum(Δ[Δ .> 0])                # minimum grid spacing
    dt::Float64 = fac_dt * (Δmin^2) ./ κ_time / 4    # timestep
    time::Float64 = 0.0             # current time
    nt::Int64 = floor(maxTime / dt)
    it::Int64 = 0               # current iteration
    ω::Float64 = 0.5             # relaxation parameter for nonlinear iterations
    solver::Symbol = :explicit            # diffusion step: `:explicit` (default), or `:implicit_enthalpy` (implicit enthalpy method, Newton-linearised and solved with accelerated pseudo-transient iterations; `fac_dt` may then exceed 1; alias `:apt`)
    out_of_plane::Bool = false            # planar 2D models: add the heat loss along the out-of-plane direction, with the length scale measured from the horizontal width of the anomaly `T - Tbg` at every depth
    out_of_plane_factor::Float64 = 1.5    # prefactor of the out-of-plane loss (1 is exact for a Gaussian anomaly; 1.5 reproduces 3D models of a long-lived intrusion best)
    out_of_plane_smooth::Int64 = 10       # the width is averaged over ± this many depth cells
    out_of_plane_Lmin::Float64 = 1.0e3    # lower bound of the length scale [m]
    apt_tol::Float64 = 1.0e-6             # implicit enthalpy solver: tolerance of the pseudo-transient iterations (temperature error relative to max(|T|, 1))
    apt_max_iter::Int64 = 50_000          # implicit enthalpy solver: maximum number of pseudo-transient iterations per solve
    apt_check::Int64 = 10                 # implicit enthalpy solver: check convergence every `apt_check` iterations
    max_iter::Int64 = 5000            # max. number of nonlinear iterations
    verbose::Bool = false
    convergence::Float64 = 1.0e-5            # nonlinear convergence criteria
    backend::KernelAbstractions.Backend = CPU()      # KernelAbstractions backend of the model arrays
    FloatType::DataType = Float64         # element type of the model arrays
    keep_init_RockPhases::Bool = true            # keep initial rock phases (if false, all phases are initialized as Dikes.BackgroundPhase)
    pvd::Union{Nothing, GeophysicalModelGenerator.WriteVTK.CollectionFile} = nothing              # pvd file info for paraview
    Output_VTK::Bool = true            # output VTK files in case CartData is an input?
    SaveOutput_steps::Int64 = 1.0e3             # saves output every x steps
    CreateFig_steps::Int64 = 500             # Create a figure every X steps

    AddRandomSills::Bool = false           # Add random sills/dikes to the model?
    RandomSills_timestep::Int64 = 10              # After how many timesteps do we add a new sill/dike?

    # parts that can be removed @ some stage
    deactivate_La_at_depth::Bool = false           # deactivate latent heating @ the bottom of the model box?
    deactivationDepth::Float64 = -15.0e3           # deactivation depth
    AnalyticalInitialGeo::Bool = false
    qs_anal::Float64 = 170.0e-3
    qm_anal::Float64 = 167.0e-3
    hr_anal::Float64 = 10.0e3
    k_anal::Float64 = 3.35
    InitialEllipse::Bool = false
    a_init::Float64 = 2.5e3
    b_init::Float64 = 1.5e3
    TrackTracersOnGrid::Bool = true
    TtPath_steps::Int64 = 10              # time steps between samples of the tracer T-t paths
    seed::Union{Nothing, Int64} = nothing        # seed of the (task-local) random number generator at the start of the simulation (`nothing`: leave it unchanged)
    print_injections::Bool = true        # print a line for every injected dike/sill
    TracerFloatType::DataType = Float32     # float type for Tracer time_vec/T_vec (Float32 saves memory)
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
    sill::Union{Nothing, InjectSills.AbstractSill} = nothing    # InjectSills.jl sill object
    T_in_Celsius::Float64 = 1000                    # Temperature of injected magma
    InjectionInterval_year::Float64 = 10.0e3                    # Injection interval [years]
    InjectionInterval::Float64 = InjectionInterval_year * SecYear            # Injection interval [s]
    nTr_dike::Int64 = 300                     # Number of tracers
    InjectVol::Float64 = 0.0                     # injected volume
    Qrate_km3_yr::Float64 = 0.0                     # Dikes insertion rate
    BackgroundPhase::Int64 = 1                       # Background phase  (non-sills)
    SillPhase::Int64 = 2                       # Sill phase
    sill_poly::Vector = []                      # polygon with sill
    sill_inj::Float64 = 0.0

    H_ran::Float64 = 5000.0                  # Zone in which we vary the horizontal location of the sill
    L_ran::Float64 = 2000.0                  # Zone in which we vary the horizontal location of the dike
    W_ran::Float64 = 2000.0                  # Zone in which we vary the vertical location of the dike
    Dip_ran::Float64 = 30.0                    # maximum variation of dip
    Strike_ran::Float64 = 90.0                    # maximum variation of strike
    SillsAbove::Float64 = -15.0e3                   # Sills above this depth
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
    Time_vec::Vector{Float64} = []         # Center of dike
    MeltFraction::Vector{Float64} = []      # Melt fraction over time
    Tav_magma::Vector{Float64} = []         # Average magma
    Tmax::Vector{Float64} = []              # Max magma temperature
end


"""
    copy_params(Num::NumParam; kwargs...) -> NumParam

Independent copy of the numerical parameters with the given fields changed, e.g. `copy_params(Num; Nx = 101, fac_dt = 0.2)`.
Fields derived from other fields (grid spacing, time step, number of steps, dimension) are recomputed when one of their inputs
is changed, unless they are given explicitly; the state of a finished run (time, iteration counter, VTK collection) is reset.
"""
function copy_params(Num::NumParam; kwargs...)
    N = deepcopy(Num)
    N.time, N.it, N.pvd = 0.0, 0, nothing
    for (k, v) in kwargs
        hasfield(NumParam, k) || throw(ArgumentError("NumParam has no field :$k"))
        setfield!(N, k, convert(fieldtype(NumParam, k), v))
    end
    given(k) = haskey(kwargs, k)
    if any(given, (:Nx, :Ny, :Nz, :W, :L, :H, :fac_dt, :κ_time, :maxTime_Myrs))
        given(:dim) || (N.dim = count(>(0), (N.Nx, N.Ny, N.Nz)))
        given(:dx) || (N.dx = N.W / (N.Nx - 1))
        given(:dy) || (N.dy = N.L / (N.Ny - 1))
        given(:dz) || (N.dz = N.H / (N.Nz - 1))
        given(:Δ) || (N.Δ = [N.dx, N.dy, N.dz])
        given(:Δmin) || (N.Δmin = minimum(N.Δ[N.Δ .> 0]))
        given(:maxTime) || (N.maxTime = N.maxTime_Myrs * SecYear * 1.0e6)
        given(:dt) || (N.dt = N.fac_dt * (N.Δmin^2) / N.κ_time / 4)
        given(:nt) || (N.nt = floor(Int, N.maxTime / N.dt))
    end
    return N
end

"""
    copy_params(Dikes::SillParams; kwargs...) -> SillParams

Independent copy of the sill parameters with the given fields changed, e.g. `copy_params(Dikes; T_in_Celsius = 1100.0)`.
The bookkeeping of a finished run (injected volume, injection counter, polygon) is reset.
"""
function copy_params(Dikes::SillParams; kwargs...)
    D = deepcopy(Dikes)
    D.InjectVol, D.Qrate_km3_yr, D.sill_inj = 0.0, 0.0, 0.0
    empty!(D.sill_poly)
    for (k, v) in kwargs
        hasfield(SillParams, k) || throw(ArgumentError("SillParams has no field :$k"))
        setfield!(D, k, convert(fieldtype(SillParams, k), v))
    end
    haskey(kwargs, :InjectionInterval_year) && !haskey(kwargs, :InjectionInterval) && (D.InjectionInterval = D.InjectionInterval_year * SecYear)
    return D
end

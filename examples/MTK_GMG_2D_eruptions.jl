# Overpressure-driven eruptions of a magma chamber in a planar 2D section with a moving free surface.
#
# A volumetric source feeds a region at depth until a melt-rich chamber has formed. The chamber is the largest connected region with
# a melt fraction ≥ 0.5; its overpressure follows a lumped mass balance, and whenever it exceeds a critical value a dike reaches the
# surface and the chamber drains. The drained volume is removed through a host-rock displacement field (Mindlin point sources of
# a body of revolution about the chamber axis, so the planar section behaves like a 3D chamber), which advects the temperature and
# moves the free surface. The same displacement is applied for the inflation by the source.
#
# The source below is a placeholder: a box that is heated (up to the magma temperature) and a constant magma recharge of the
# chamber. Replace `MTK_source!`, the recharge rate `A_DOT` and the inflation by your own source of magma.

using MagmaThermoKinematics
backend = CPU()
using GeophysicalModelGenerator
using GeoParams, Random
using MagmaThermoKinematics.MTK_GMG     # Allow overwriting user routines

Random.seed!(1234)

# --- Model: x ∈ [-15, 15] km, z ∈ [-20, 3] km, air above z = 0 -------------------------------------------------------------
xs, zs = -15:0.2:15, -20:0.2:3
X = [x for x in xs, z in zs, k in 1:1]; Z = [z for x in xs, z in zs, k in 1:1]       # a 2D section is stored as (Nx, Nz, 1)
Data_2D = CartData(X, zeros(size(X)), Z, (Phases = ones(Int64, size(X)), Temp = zeros(size(X)), FlatCrossSection = copy(X)))
Data_2D.fields.Phases[Data_2D.z.val .> 0.0] .= 0                                   # air
Data_2D.fields.Temp .= max.(-Data_2D.z.val .* 25.0, 10.0)                           # 25 K/km geotherm [°C]
T_GEO(depth) = max(25.0e-3 * depth, 10.0)                             # undisturbed geotherm [°C] at a depth [m]

# --- Placeholder source: a heated box and a constant recharge of the chamber ------------------------------------------------
const T_IN = 950.0                                                                  # temperature of the magma [°C]
const R_BOX, Z_BOX, H_BOX = 3.0e3, -7.0e3, 1.0e3                                   # half-width, centre depth and thickness of the box [m]
const H_BOX_HEAT = 0.02                                                             # heat production in the box [W/m³], while T < T_IN
const A_DOT = 0.02 / SecYear                                                        # magma recharge per footprint area of the chamber [m/s]
const Q_3D = π * R_BOX^2 * H_BOX / (8.0e3 * SecYear)                                # 3D injection rate of the inflation sills [m³/s]
const INBOX = Ref{Any}(nothing)

function MTK_GMG.MTK_source!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Mat_tup::Tuple, Dikes::SillParameters)
    INBOX[] === nothing && (INBOX[] = (abs.(Arrays.R) .<= R_BOX) .& (abs.(Arrays.Z .- Z_BOX) .<= H_BOX / 2))
    @. Arrays.Qsrc = ifelse(INBOX[] & (Arrays.T < T_IN), H_BOX_HEAT, 0.0)
    return nothing
end

# --- Eruption model and the hooks that apply it ---------------------------------------------------------------------------
const EP = EruptionParams(
    ΔP_crit = 15.0e6, σ_barrier = 5.0e6, w_dike = 5.0, μ_shear = 1.0e10, m_w = 0.04,
    h_melt_min = 400.0,                                                             # minimum melt thickness of a chamber [m]
    melt_viscosity = LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.9606K, η0 = 1Pas),
    solubility = Liu2005_Solubility(),
)
const STATE = EruptionState()
const EVENTS = Vector{NTuple{5, Float64}}()                                         # (time [kyr], drained fraction, 3D volume [km³], chamber depth [km], surface max [m])
const INFL = Ref{Any}(nothing)                                                      # inflation displacement rate [m/s]
const INFL_EVERY = 5                                                                # apply the inflation every this many steps
const PLITH = Ref{Any}(nothing)                                                     # lithostatic pressure and work array

function setup_inflation!(Arrays, Grid)
    # ensemble of penny-shaped sills (radius R_s, thickness H_s) of the source: the caller defines where they are
    rng = MersenneTwister(7); n = 64
    R_s, H_s = 1.5e3, 40.0
    offset = R_BOX .* sqrt.(rand(rng, n)); zc = Z_BOX .+ H_BOX .* (rand(rng, n) .- 0.5)
    interval = (2 / 3 * π * R_s^2 * H_s) / Q_3D                                      # mean sill volume / injection rate [s]
    return revolved_inflation_rate((Arrays.R, Arrays.Z), zeros(n), offset, zc, fill(R_s, n), interval; thickness = H_s, nθ = 8)
end

function erupt_and_inflate(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters)
    Tracers = MTK_GMG.MTK_updateTracers(Grid, Arrays, Tracers, Dikes, time_props, Num)         # default tracer update
    coords = Tuple(Grid.coord1D)
    zsurf = max(0.0, maximum(filter(!isnan, Array(Arrays.Zsurf)); init = 0.0))                 # the kernels are valid up to the free surface
    # inflation of the source region: host rock moves outward, the surface rises
    if mod(Num.it, INFL_EVERY) == 0
        INFL[] === nothing && (INFL[] = setup_inflation!(Arrays, Grid))
        U = map(u -> u .* (INFL_EVERY * Num.dt), INFL[])
        displace_temperature!(Arrays.T, similar(Arrays.T), coords, U)
        move_free_surface!(Arrays.Phases, Arrays.Phases_init, Arrays.T, Arrays.Zsurf, coords, U; air = Num.AirPhase, fill = Dikes.BackgroundPhase)
        Arrays.Tnew .= Arrays.T
    end
    # lithostatic pressure: one top-down integration of the density, with the air above the free surface at zero density
    PLITH[] === nothing && (PLITH[] = (similar(Arrays.P), similar(Arrays.Rho)))
    Pl, ρ_rock = PLITH[]
    @. ρ_rock = ifelse(Arrays.Phases == Num.AirPhase, 0.0, Arrays.Rho)
    MagmaThermoKinematics.lithostatic_pressure!(Pl, ρ_rock, 9.81, step(coords[end]))       # (GeophysicalModelGenerator exports a function of the same name)
    # chamber overpressure; `recharge` is the magma volume that the source adds to the chamber per second [m²/s]
    recharge = STATE.init ? A_DOT * length(unique(I[1] for I in STATE.cells)) * step(coords[1]) : 0.0       # ȧ × chamber width
    ev = chamber_step!(STATE, EP, Arrays.ϕ, Arrays.T, coords; recharge, Δt = Num.dt, time = Num.time, T_background = depth -> T_GEO(depth), margin = 500.0, resolve = 0.5, P = Pl, Zsurf = Arrays.Zsurf)
    ev === nothing && return Tracers
    zsurf = max(0.0, maximum(filter(!isnan, Array(Arrays.Zsurf)); init = 0.0))
    U, _ = erupt!(Arrays.T, similar(Arrays.T), Tracers, coords, ev; revolve = true, nθ = 8, coarsen = 2, z_surface = zsurf, Zsurf = Arrays.Zsurf)
    move_free_surface!(Arrays.Phases, Arrays.Phases_init, Arrays.T, Arrays.Zsurf, coords, U; air = Num.AirPhase, fill = Dikes.BackgroundPhase)
    Arrays.Tnew .= Arrays.T
    push!(EVENTS, (Num.time / SecYear / 1.0e3, ev.f, ev.f * π * ev.geo.R^2 * ev.geo.h / km³, -ev.geo.zc / 1.0e3, maximum(filter(!isnan, Array(Arrays.Zsurf)))))
    return Tracers
end

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it, 100) == 0
        ΔP = STATE.init ? (STATE.P - STATE.P_lith) / 1.0e6 : NaN
        println("$(Num.it), t = $(round(Num.time / SecYear / 1.0e3, digits = 1)) kyr; max(T) = $(round(maximum(Arrays.Tnew))) °C; overpressure $(round(ΔP, digits = 2)) MPa; eruptions: $(length(EVENTS))")
    end
    return nothing
end
MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters) = nothing

# --- Parameters ----------------------------------------------------------------------------------------------------------
Num = NumParam(
    SimName = "Eruptions2D", maxTime_Myrs = 0.05, fac_dt = 0.4, Output_VTK = false, SaveOutput_steps = 10^9, CreateFig_steps = 10^9,
    Tsurface_Celsius = 10.0, sticky_air_BC = true, AirPhase = 0,
    move_free_surface = true,                                                       # the free surface moves with the displacement of the injections
    backend = backend,
)
Sill_params = SillParams(                                                           # sills are not injected: the placeholder source is the only supply
    sill = PennyShapedSill(Center = Point2(0.0, -7.0e3)m, R = 1.0e3m, H = 100m, E = 1.5e10Pa, ν = 0.3NoUnits),
    InjectionInterval_year = 1.0e12, nTr_dike = 1, SillPhase = 2, T_in_Celsius = T_IN, SillsAbove = -12.0e3,
)
rock(phase, name) = SetMaterialParams(
    Name = name, Phase = phase, Density = ConstantDensity(ρ = 2700kg / m^3), LatentHeat = ConstantLatentHeat(Q_L = 2.55e5J / kg),
    Conductivity = ConstantConductivity(k = 2.5Watt / K / m), HeatCapacity = ConstantHeatCapacity(Cp = 1050J / kg / K),
    Melting = SmoothMelting(MeltingParam_Quadratic(T_s = (700 + 273.15)K, T_l = (1100 + 273.15)K)),
)
MatParam = (rock(0, "Air"), rock(1, "Host rock"), rock(2, "Magma"))

Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params; CartData_input = Data_2D, callbacks = (; updateTracers = erupt_and_inflate))

for e in EVENTS[(end - min(length(EVENTS), 5) + 1):end]
    println("eruption at $(round(e[1], digits = 1)) kyr: drained fraction $(round(e[2], digits = 3)), 3D volume $(round(e[3], digits = 2)) km³, chamber depth $(round(e[4], digits = 1)) km")
end
println("\n$(length(EVENTS)) eruptions; erupted 3D volume $(round(sum(e[3] for e in EVENTS; init = 0.0), digits = 2)) km³; max. surface elevation $(round(maximum(filter(!isnan, Array(Arrays.Zsurf))))) m")

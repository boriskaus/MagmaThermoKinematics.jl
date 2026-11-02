# High-resolution 2D showcase: the last 150 kyr of a magmatic system below Unzen volcano (Japan), on a W–E section through the
# present topography. It combines most features of the package:
#  - sills injected in two zones (shallow and deep), with the host rock displaced by the opening of every sill (InjectSills.jl),
#  - a magma chamber that builds up overpressure and erupts, removing magma and moving the host rock and the free surface,
#  - the viscosity of the shell around the chamber from the temperature field and a layered host rock (one creep law per layer),
#  - parameterised convection of melt-rich regions, out-of-plane heat loss, the implicit enthalpy solver, sticky air,
#  - tracers that record temperature–time paths for zircon ages (see HighRes_Unzen_plots.jl),
#  - passive markers of the initial depth, which show how the crust deforms.
# The run starts from a closed-form fit to the temperature and surface of a lower-resolution model at 150 kyr before present.
#
# Usage:  RES=50 julia --project HighRes_Unzen_2D.jl          (RES: grid spacing [m]; GPU=1 runs on an NVIDIA GPU)
using MagmaThermoKinematics, MagmaThermoKinematics.MTK_GMG
using GeophysicalModelGenerator, GeoParams, JLD2, Random
const GPU = get(ENV, "GPU", "0") == "1"
GPU && using CUDA
backend = GPU ? CUDABackend() : CPU()
const RES = parse(Float64, get(ENV, "RES", "100"))                                   # grid spacing [m]
const OUT = get(ENV, "OUT", "HighRes2D_$(round(Int, RES))m"); mkpath(OUT)
const DT_YR = parse(Float64, get(ENV, "DT_YR", "25"))                               # time step [yr]
const SNAP_KYR = parse(Float64, get(ENV, "SNAP_KYR", "0.5"))                        # snapshot interval [kyr] (movie frames)
const SNAP_STRIDE = parse(Int, get(ENV, "SNAP_STRIDE", string(max(1, round(Int, 20 / RES)))))   # store every n-th grid point (default: a 20 m grid; the final state is stored in full)
const T_RUN = parse(Float64, get(ENV, "T_RUN_KYR", "150"))                           # simulated time [kyr]
rng = MersenneTwister(2206)
const T_START = time()

# --- Initial state: a fit to a lower-resolution model of the preceding evolution, 150 kyr before present ---------------------
const Y_SEC = 4.0                                                                   # the 2D section lies 4 km north of the centre of the topography window [km]
T_BG(d) = 73.6 + 12.58d + 0.292d^2 - 0.00588d^3                                     # background temperature [°C] at the depth d [km] below sea level
const WARM = (                                                                      # warm bodies: amplitude [K], centre (x, z) [km], horizontal and vertical e-folding widths [km]
    (A = 355.0, x = 5.9, z = -4.9, a = 4.5, b = 3.7),
    (A = 324.0, x = 0.1, z = -12.6, a = 5.9, b = 5.8),
    (A = 370.0, x = 4.2, z = -5.6, a = 8.0, b = 8.5),
)
T_init(x, y, z) = T_BG(-z) + sum(w.A * exp(-((x - w.x)^2 + (y - Y_SEC)^2) / w.a^2 - ((z - w.z) / w.b)^2) for w in WARM)    # [°C] at x, y, z [km]
uplift(x, y) = 154.0 * exp(-((x - 4.8)^2 + (y - Y_SEC)^2) / 5.2^2)                  # uplift of the surface during the run [m]: the initial surface is lower
T_GEO(depth) = T_BG(depth / 1.0e3)                                                  # undisturbed geotherm [°C] at a depth [m]
Topo = load_GMG(joinpath(pkgdir(MagmaThermoKinematics), "examples", "Topo_cart"))  # present topography [km]
function topo(x, y)                                                                 # bilinear interpolation of the topography [km] at x, y [km]
    xt, yt, zt = Topo.x.val[:, 1, 1], Topo.y.val[1, :, 1], Topo.z.val[:, :, 1]
    i = clamp(searchsortedlast(xt, x), 1, length(xt) - 1); j = clamp(searchsortedlast(yt, y), 1, length(yt) - 1)
    u = clamp((x - xt[i]) / (xt[i + 1] - xt[i]), 0, 1); v = clamp((y - yt[j]) / (yt[j + 1] - yt[j]), 0, 1)
    return (1 - u) * (1 - v) * zt[i, j] + u * (1 - v) * zt[i + 1, j] + (1 - u) * v * zt[i, j + 1] + u * v * zt[i + 1, j + 1]
end

# --- Grid and layers -------------------------------------------------------------------------------------------------------
xs, zs = -20.0e3:RES:20.0e3, -20.0e3:RES:5.0e3                                      # W–E section [m]
Zinit = [1.0e3 * topo(x / 1.0e3, Y_SEC) - uplift(x / 1.0e3, Y_SEC) for x in xs]    # initial surface [m]
X = [x / 1.0e3 for x in xs, z in zs, _ in 1:1]; Z = [z / 1.0e3 for x in xs, z in zs, _ in 1:1]
Temp = T_init.(X, Y_SEC, Z)
Data_2D = CartData(X, fill(Y_SEC, size(X)), Z, (Phases = zeros(Int64, size(X)), Temp = Temp, Tbg = T_BG.(-Z), FlatCrossSection = copy(X)))   # Tbg: the out-of-plane loss acts on the warm anomaly too
const Z_LAYERS = (-2.0e3, -12.0e3)                                                  # volcanic cover / upper crust / lower crust [m]; air 0, magma 4
add_layer!(Data_2D.fields.Phases, Temp, Data_2D; zlim = (Z_LAYERS[1] / 1.0e3, 5.0), phase = ConstantPhase(1))
add_layer!(Data_2D.fields.Phases, Temp, Data_2D; zlim = (Z_LAYERS[2] / 1.0e3, Z_LAYERS[1] / 1.0e3), phase = ConstantPhase(2))
add_layer!(Data_2D.fields.Phases, Temp, Data_2D; zlim = (-20.0, Z_LAYERS[2] / 1.0e3), phase = ConstantPhase(3))
Data_2D.fields.Phases[Z .> reshape(Zinit, :, 1, 1) ./ 1.0e3] .= 0                    # air above the initial surface

# --- Magma supply: sills and dikes in a shallow and a deep zone --------------------------------------------------------------------
const Q = parse(Float64, get(ENV, "Q_KM3", "1.5")) * km³ / (1.0e3 * SecYear)       # magma flux [m³/s] of the 3D volcano
const ZONES = (                                                                     # centre x, half-width R, depth range [m], fraction of Q
    (name = "shallow", x = 5.0e3, R = 3.2e3, z = (-7.0e3, -3.0e3), frac = 0.7),
    (name = "deep", x = 1.0e3, R = 4.5e3, z = (-16.0e3, -8.0e3), frac = 0.3),
)
area_rate(zone) = 2 * zone.frac * Q / (π * zone.R)                                  # 2D section through the centre of a disc-shaped zone [m²/s]
const T_IN = 1050.0                                                                 # temperature of the injected magma [°C]
const H_SILL = max(10.0, 2 * RES)                                                   # sill thickness [m]: 10 m from RES = 5 m (a sill needs about two cells)
const R_MED, R_MAX = 500.0, 2.0e3                                                   # radius of the sills and dikes: lognormal median and maximum [m]
const NEXT = [0.0, 0.0]                                                             # time of the next sill of each zone [s]
const SILLS = NamedTuple[]                                                          # every sill and dike: time [kyr], centre, radius [m], angle [°] (for the movie)
const INJECTED = Ref(0.0); const RECHARGE = Ref(0.0)                               # injected 3D volume [m³], area added to the chamber [m²]
const Z0 = Ref{Any}(nothing)                                                        # initial height of the rock [m] (advected marker)
const SILL = Ref{Any}(nothing)                                                      # 1 in intruded magma, 0 elsewhere: the phase grid of the sills, moved with the host rock
const SILL_PHASE = 4
host_phase(z0) = z0 > Z_LAYERS[1] ? 1 : z0 > Z_LAYERS[2] ? 2 : 3                   # crustal layer from the initial height of the rock

function move_all!(Arrays, Grid, Num, U, fill; sill = nothing)
    coords = Tuple(Grid.coord1D)
    displace_temperature!(Z0[], similar(Z0[]), coords, U)
    S = SILL[]; @. S = Arrays.Phases == SILL_PHASE                                   # sills as a 0/1 field, moved like Z0 and made sharp again below
    displace_temperature!(S, similar(S), coords, U)
    sill === nothing || MagmaThermoKinematics.add_dike(S, nothing, Grid.coord1D, sill, 1.0, SILL_PHASE, 0)   # the new sill
    @. Arrays.Phases = ifelse(Arrays.Phases == Num.AirPhase, Num.AirPhase, ifelse(S >= 0.5, SILL_PHASE, host_phase(Z0[])))
    move_free_surface!(Arrays.Phases, Arrays.Phases_init, Arrays.T, Arrays.Zsurf, coords, U; air = Num.AirPhase, fill)
    return nothing
end

function inject_two_zones(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructArray)
    if Z0[] === nothing                                                             # first call: markers, continuous initial surface
        Z0[] = copy(Arrays.Z); SILL[] = similar(Arrays.T); copyto!(Arrays.Zsurf, Zinit)
    end
    for (k, zone) in enumerate(ZONES)
        while NEXT[k] <= Num.time
            R = min(R_MAX, R_MED * exp(0.6 * randn(rng)))                            # lognormal radius
            xc = zone.x + zone.R * (2 * rand(rng) - 1); zc = zone.z[1] + (zone.z[2] - zone.z[1]) * rand(rng)
            angle = 180 * rand(rng) - 90                                            # random orientation: sills and dikes
            sill = PennyShapedSill(Center = Point2(xc, zc) * m, Angle = Vec1(angle) * NoUnits, R = R * m, H = H_SILL * m, E = 1.5e10Pa, ν = 0.3NoUnits)
            push!(SILLS, (time_kyr = Num.time / SecYear / 1.0e3 - T_RUN, x = xc, z = zc, R, angle))
            Tracers, _, _, _, U = inject_sills(Tracers, Arrays.T, Grid.coord1D, sill, T_IN, Dikes.SillPhase, 20)
            move_all!(Arrays, Grid, Num, U, Dikes.BackgroundPhase; sill)
            area = π / 2 * R * H_SILL
            NEXT[k] += area / area_rate(zone); INJECTED[] += area * π * zone.R / 2   # 3D volume represented by the sill
            in_chamber(xc, zc) && (RECHARGE[] += area)
        end
    end
    Arrays.Tnew .= Arrays.T
    return Tracers
end

# --- Eruptions -----------------------------------------------------------------------------------------------------------
const EP = EruptionParams(
    ΔP_crit = 10.0e6, σ_barrier = 5.0e6, w_dike = 5.0, μ_shear = 1.0e10, m_w = 0.04, h_melt_min = 400.0,
    melt_viscosity = LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.9606K, η0 = 1Pas), solubility = Liu2005_Solubility(),
    shell_from_T = true, shell_z_split = Z_LAYERS,                                  # layered host rock: one creep law per layer
    shell_rheology = (
        SetDislocationCreep(GeoParams.Dislocation.wet_quartzite_Ueda_2008), SetDislocationCreep(GeoParams.Dislocation.granite_Tirel_2008),
        SetDislocationCreep(GeoParams.Dislocation.wet_anorthite_Rybacki_2006),
    ),
)
const STATE = EruptionState()
const EVENTS = NamedTuple[]
const PLITH = Ref{Any}(nothing)
const MARGIN = 1.0e3                                                                # sills this close to the chamber recharge it [m]
const COARSEN = max(2, round(Int, 200 / RES))                                       # chamber cells are grouped into sources ≈ 200 m apart for the displacement field
const CH_X = Ref((0.0, 0.0))                                                        # centre and half-width of the chamber [m]
in_chamber(x, z) = STATE.init && STATE.z_lo - MARGIN <= z <= STATE.z_hi + MARGIN && abs(x - CH_X[][1]) <= CH_X[][2] + MARGIN

"Phases on every `s`-th grid point; a point shows the sill phase if any cell of its block holds a sill, so that sills thinner than the snapshot grid stay visible."
coarse_phases(P, s) = [any(==(SILL_PHASE), view(P, map((i, n) -> i:min(i + s - 1, n), Tuple(I), size(P))...)) ? SILL_PHASE : P[I] for I in CartesianIndices(map(n -> 1:s:n, size(P)))]

function erupt_and_save(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters)
    Tracers = MTK_GMG.MTK_updateTracers(Grid, Arrays, Tracers, Dikes, time_props, Num)
    coords = Tuple(Grid.coord1D)
    PLITH[] === nothing && (PLITH[] = (similar(Arrays.P), similar(Arrays.Rho)))
    Pl, ρ = PLITH[]
    @. ρ = ifelse(Arrays.Phases == Num.AirPhase, 0.0, Arrays.Rho)
    MagmaThermoKinematics.lithostatic_pressure!(Pl, ρ, 9.81, step(coords[end]))
    ev = chamber_step!(STATE, EP, Arrays.ϕ, Arrays.T, coords; recharge = RECHARGE[] / Num.dt, Δt = Num.dt, time = Num.time, T_background = T_GEO, margin = 500.0, resolve = 0.5, P = Pl, Zsurf = Arrays.Zsurf)
    RECHARGE[] = 0.0
    if ev !== nothing
        zsurf = maximum(filter(!isnan, Array(Arrays.Zsurf)))
        U, _ = erupt!(Arrays.T, similar(Arrays.T), Tracers, coords, ev; revolve = true, nθ = 8, coarsen = COARSEN, z_surface = zsurf, Zsurf = Arrays.Zsurf)
        move_all!(Arrays, Grid, Num, U, Dikes.BackgroundPhase)
        Arrays.Tnew .= Arrays.T
        push!(EVENTS, (time_kyr = Num.time / SecYear / 1.0e3 - T_RUN, V_km3 = ev.f * π * ev.geo.R^2 * ev.geo.h / km³, depth_km = -ev.geo.zc / 1.0e3))
    end
    if STATE.init                                                                   # horizontal extent of the chamber, for the recharge
        xcell = [coords[1][I[1]] for I in STATE.cells]
        CH_X[] = (sum(xcell) / length(xcell), (maximum(xcell) - minimum(xcell)) / 2)
    end
    if Num.it == 1 || floor(Num.time / (SNAP_KYR * 1.0e3 * SecYear)) > floor((Num.time - Num.dt) / (SNAP_KYR * 1.0e3 * SecYear))
        s = SNAP_STRIDE; sub(A) = Array(A)[1:s:end, 1:s:end]
        jldsave(
            joinpath(OUT, "snap_" * lpad(Num.it, 7, '0') * ".jld2"); time_kyr = Num.time / SecYear / 1.0e3 - T_RUN, x = coords[1][1:s:end], z = coords[2][1:s:end],
            T = Float32.(sub(Arrays.T)), ϕ = round.(UInt8, 255 .* clamp.(sub(Arrays.ϕ), 0, 1)), Z0 = Float32.(sub(Z0[])), Phases = Int8.(coarse_phases(Array(Arrays.Phases), s)), Zsurf = Array(Arrays.Zsurf)[1:s:end],
            Nu = haskey(Arrays, :Nu) ? Float32.(sub(Arrays.Nu)) : nothing, injected_km3 = INJECTED[] / km³, events = copy(EVENTS)
        )
    end
    return Tracers
end

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    mod(Num.it, 100) == 0 || return nothing
    ΔP = STATE.init ? round((STATE.P - STATE.P_lith) / 1.0e6, digits = 2) : NaN
    println("$(Num.it): t = $(round(Num.time / SecYear / 1.0e3 - T_RUN, digits = 2)) kyr, max T $(round(maximum(Arrays.Tnew))) °C, injected $(round(INJECTED[] / km³, digits = 2)) km³, overpressure $ΔP MPa, eruptions $(length(EVENTS)), wall $(round((time() - T_START) / 60, digits = 1)) min"); flush(stdout)
    return nothing
end
MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters) = nothing

# --- Material parameters and numerics ------------------------------------------------------------------------------------
melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0)     # rhyolite
rock(phase, name; Q_L = 2.55e5) = SetMaterialParams(
    Name = name, Phase = phase, Density = ConstantDensity(ρ = 2700kg / m^3), LatentHeat = ConstantLatentHeat(Q_L = Q_L * J / kg),
    Conductivity = ConstantConductivity(k = 3.0Watt / K / m), HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K), Melting = melting,
)
MatParam = (rock(0, "Air"; Q_L = 0.0), rock(1, "Volcanic cover"), rock(2, "Upper crust"), rock(3, "Lower crust"), rock(4, "Magma"))
κ = 3.0 / (2700 * 1000)
Num = NumParam(
    SimName = "HighRes2D", dim = 2, maxTime_Myrs = T_RUN * 1.0e-3, κ_time = κ, fac_dt = 4 * DT_YR * SecYear * κ / RES^2,
    solver = :implicit_enthalpy, ω = 1.0, convection = true, convection_phi = 0.6, convection_boundary_layer = false, out_of_plane = true,
    Tsurface_Celsius = 10.0, sticky_air_BC = true, AirPhase = 0, move_free_surface = true, TtPath_steps = 5,
    Output_VTK = false, SaveOutput_steps = 10^9, CreateFig_steps = 10^9, print_injections = false, backend = backend, seed = 2206,
)
Sill_params = SillParams(sill = PennyShapedSill(Center = Point2(0.0, -5.0e3)m, R = 1.0e3m, H = 10m), InjectionInterval_year = 1.0e15, SillPhase = SILL_PHASE, BackgroundPhase = 1, T_in_Celsius = T_IN)

t0 = time()
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params; CartData_input = Data_2D, callbacks = (; inject_dikes = inject_two_zones, updateTracers = erupt_and_save))
println("done: $(Num.it) steps on $(length(xs))×$(length(zs)) points in $(round((time() - t0) / 60, digits = 1)) min; injected $(round(INJECTED[] / km³, digits = 1)) km³, $(length(EVENTS)) eruptions, $(round(sum(e.V_km3 for e in EVENTS; init = 0.0), digits = 2)) km³ erupted")
jldsave(joinpath(OUT, "final.jld2"); events = EVENTS, tracers = Tracers, x = collect(xs), z = collect(zs), T = Float32.(Array(Arrays.T)), ϕ = Float32.(Array(Arrays.ϕ)), Z0 = Float32.(Array(Z0[])), Phases = Int8.(Array(Arrays.Phases)), Zsurf = Array(Arrays.Zsurf), Zinit = Zinit)
jldsave(joinpath(OUT, "sills.jld2"); sills = SILLS)

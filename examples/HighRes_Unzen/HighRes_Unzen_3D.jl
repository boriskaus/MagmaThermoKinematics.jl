# High-resolution 3D showcase: the last 150 kyr of a magmatic system below Unzen volcano (Japan), below the present topography.
# The 3D counterpart of HighRes_Unzen_2D.jl (see there for the features); intended for a GPU.
#  - sills in a shallow and a deep zone (discs below the volcano), each displacing the host rock,
#  - a magma chamber that erupts (Mindlin half-space sources), moving the host rock and the free surface,
#  - shell viscosity from the temperature field with one creep law per crustal layer (layers built with GeophysicalModelGenerator),
#  - parameterised convection, the implicit enthalpy solver, sticky air, tracers for zircon ages, markers of the initial depth.
# The initial temperature and surface: the closed-form fit of HighRes_Unzen_2D.jl, with every warm body about its own vertical axis.
#
# Usage:  julia --project HighRes_Unzen_3D.jl [res] [Q] [gpu]
#  - res: grid spacing [m] (default 200)
#  - Q:   magma flux [km³/kyr] (default 2.36)
#  - gpu: run on an NVIDIA GPU (CUDA.jl must be available in the active environment)
# The run of the documentation:  julia --project HighRes_Unzen_3D.jl 50 2.36 gpu
using MagmaThermoKinematics, MagmaThermoKinematics.MTK_GMG
using GeophysicalModelGenerator, GeoParams, JLD2, Random

res = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 200.0   # grid spacing [m]
Q_km3 = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 2.36  # magma flux [km³/kyr]: 3D loses heat in all directions, so ~1.5× the 2D flux
use_gpu = length(ARGS) > 2 && ARGS[3] == "gpu"
if use_gpu
    using CUDA
end
backend = use_gpu ? CUDABackend() : CPU()

t_run = 150.0                                           # simulated time [kyr]
dt_yr = 25.0                                            # time step [yr]
snapshot_kyr = 1.5                                      # time between two snapshots (the frames of the movie) [kyr]
snapshot_stride = max(1, round(Int, 100 / res))         # snapshots keep every snapshot_stride-th grid point (a 100 m grid); the final state is saved in full
outdir = mkpath("HighRes3D_$(isinteger(res) ? Int(res) : res)m")
rng = MersenneTwister(2206)
wall_start = time()
kyr_before_present(t) = t / SecYear / 1.0e3 - t_run     # model time [s] -> time before present [kyr] (negative)

# --- Initial state: a fit to a lower-resolution model of the preceding evolution, 150 kyr before present ---------------------
T_bg(d) = 73.6 + 12.58d + 0.292d^2 - 0.00588d^3          # background temperature [°C] at a depth d [km] below sea level
T_geotherm(depth) = T_bg(depth / 1.0e3)                 # the same at a depth [m]: the undisturbed geotherm

# warm bodies: amplitude [K], vertical axis (x, y) and centre z [km], horizontal and vertical e-folding widths [km]
warm_bodies = (
    (A = 355.0, x = 5.9, y = 4.0, z = -4.9, a = 4.5, b = 3.7),
    (A = 324.0, x = 0.1, y = 4.0, z = -12.6, a = 5.9, b = 5.8),
    (A = 370.0, x = 4.2, y = 4.0, z = -5.6, a = 8.0, b = 8.5),
)
T_init(x, y, z, bodies) = T_bg(-z) + sum(w.A * exp(-((x - w.x)^2 + (y - w.y)^2) / w.a^2 - ((z - w.z) / w.b)^2) for w in bodies)   # [°C] at x, y, z [km]
uplift(x, y) = 154.0 * exp(-((x - 4.8)^2 + (y - 4.0)^2) / 5.2^2)                    # uplift of the surface during the run [m]: the initial surface is lower

topography = load_GMG(joinpath(pkgdir(MagmaThermoKinematics), "examples", "Topo_cart"))   # present topography [km]
x_topo, y_topo, z_topo = topography.x.val[:, 1, 1], topography.y.val[1, :, 1], topography.z.val[:, :, 1]

"Height of the present topography [km] at x, y [km], interpolated bilinearly."
function topo(x, y)
    i = clamp(searchsortedlast(x_topo, x), 1, length(x_topo) - 1)
    j = clamp(searchsortedlast(y_topo, y), 1, length(y_topo) - 1)
    u = clamp((x - x_topo[i]) / (x_topo[i + 1] - x_topo[i]), 0, 1)
    v = clamp((y - y_topo[j]) / (y_topo[j + 1] - y_topo[j]), 0, 1)
    return (1 - u) * (1 - v) * z_topo[i, j] + u * (1 - v) * z_topo[i + 1, j] + (1 - u) * v * z_topo[i, j + 1] + u * v * z_topo[i + 1, j + 1]
end

# --- Grid and layers -------------------------------------------------------------------------------------------------------
xs, ys, zs = -20.0e3:res:20.0e3, -14.0e3:res:19.0e3, -20.0e3:res:5.0e3              # [m]
X, Y, Z = xyz_grid(xs ./ 1.0e3, ys ./ 1.0e3, zs ./ 1.0e3)
Zpres = [1.0e3 * topo(x / 1.0e3, y / 1.0e3) for x in xs, y in ys]                  # present surface [m]
Zinit = [Zpres[i, j] - uplift(x / 1.0e3, y / 1.0e3) for (i, x) in enumerate(xs), (j, y) in enumerate(ys)]   # initial surface [m]
Temp = T_init.(X, Y, Z, Ref(warm_bodies))
Data_3D = CartData(X, Y, Z, (Phases = zeros(Int64, size(X)), Temp = Temp))

# phases: air 0, volcanic cover 1, upper crust 2, lower crust 3, magma 4
z_layers = (-2.0e3, -12.0e3)                                                        # base of the volcanic cover and of the upper crust [m]
add_layer!(Data_3D.fields.Phases, Temp, Data_3D; zlim = (z_layers[1] / 1.0e3, 5.0), phase = ConstantPhase(1))
add_layer!(Data_3D.fields.Phases, Temp, Data_3D; zlim = (z_layers[2] / 1.0e3, z_layers[1] / 1.0e3), phase = ConstantPhase(2))
add_layer!(Data_3D.fields.Phases, Temp, Data_3D; zlim = (-20.0, z_layers[2] / 1.0e3), phase = ConstantPhase(3))
Data_3D.fields.Phases[Z .> reshape(Zinit, size(Zinit)..., 1) ./ 1.0e3] .= 0          # air above the initial surface
sill_phase = 4
host_phase(z0, z_cover, z_upper) = z0 > z_cover ? 1 : z0 > z_upper ? 2 : 3          # crustal layer from the initial height z0 of the rock

# --- Magma supply: sills and dikes in a shallow and a deep zone --------------------------------------------------------------------
Q = Q_km3 * km³ / (1.0e3 * SecYear)                                                 # magma flux [m³/s]
zones = (                                                                           # centre (x, y), radius R, depth range [m], fraction of Q
    (name = "shallow", c = (5.0e3, 4.0e3), R = 3.2e3, z = (-7.0e3, -3.0e3), frac = 0.7),
    (name = "deep", c = (1.0e3, 4.0e3), R = 4.5e3, z = (-16.0e3, -8.0e3), frac = 0.3),
)
T_magma = 1050.0                                                                    # temperature of the injected magma [°C]
H_sill = max(10.0, 2 * res)                                                         # sill thickness [m] (a sill needs about two cells)
R_median, R_max = 500.0, 2.0e3                                                      # radius of the sills and dikes: lognormal median and maximum [m]

"What the callbacks keep from one time step to the next."
Base.@kwdef mutable struct ModelState
    next_sill::Vector{Float64} = [0.0, 0.0]          # time of the next sill in each zone [s]
    injected::Float64 = 0.0                          # injected magma [m³]
    recharge::Float64 = 0.0                          # volume of the sills added to the chamber in this time step [m³]
    chamber_xy::NTuple{3, Float64} = (0.0, 0.0, 0.0) # centre (x, y) and radius of the chamber [m]
    Z0 = nothing                                     # initial height of the rock [m], moved with the rock: shows the deformation
    S = nothing                                      # 1 in sills and dikes, 0 elsewhere, moved with the rock
    P_lith = nothing                                 # work arrays for the lithostatic pressure: pressure and density
    events::Vector{NamedTuple} = NamedTuple[]        # every eruption: time [kyr], volume [km³], depth [km]
end
model = ModelState()

"Moves the markers of the initial height, the sills and the free surface with the displacement `U` of a new `sill` or of an eruption."
function move_rock!(Arrays, Grid, Num, U, fill; sill = nothing)
    coords = Tuple(Grid.coord1D)
    displace_temperature!(model.Z0, similar(model.Z0), coords, U)
    S = model.S
    @. S = Arrays.Phases == sill_phase                                              # sills as a 0/1 field, moved like Z0 and made sharp again below
    displace_temperature!(S, similar(S), coords, U)
    if sill !== nothing
        MagmaThermoKinematics.add_dike(S, nothing, Grid.coord1D, sill, 1.0, sill_phase, 0)
    end
    z_cover, z_upper = z_layers
    @. Arrays.Phases = ifelse(Arrays.Phases == Num.AirPhase, Num.AirPhase, ifelse(S >= 0.5, sill_phase, host_phase(model.Z0, z_cover, z_upper)))
    move_free_surface!(Arrays.Phases, Arrays.Phases_init, Arrays.T, Arrays.Zsurf, coords, U; air = Num.AirPhase, fill)
    return nothing
end

"Callback that injects the sills and dikes of both zones that are due by now."
function inject_two_zones(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructArray)
    if model.Z0 === nothing                                                         # first call: the markers, and the initial surface
        model.Z0 = copy(Arrays.Z)
        model.S = similar(Arrays.T)
        copyto!(Arrays.Zsurf, Zinit)
    end
    for (k, zone) in enumerate(zones)
        while model.next_sill[k] <= Num.time
            R = min(R_max, R_median * exp(0.6 * randn(rng)))                        # lognormal radius
            r, θ = zone.R * sqrt(rand(rng)), 2π * rand(rng)                         # uniform position in the disc of the zone
            xc, yc = zone.c[1] + r * cos(θ), zone.c[2] + r * sin(θ)
            zc = zone.z[1] + (zone.z[2] - zone.z[1]) * rand(rng)
            dip, strike = 90 * rand(rng), 360 * rand(rng)                           # random orientation: sills and dikes
            sill = PennyShapedSill(Center = Point3(xc, yc, zc) * m, Angle = Vec2(dip, strike) * NoUnits, R = R * m, H = H_sill * m, E = 1.5e10Pa, ν = 0.3NoUnits)
            Tracers, _, V, _, U = inject_sills(Tracers, Arrays.T, Grid.coord1D, sill, T_magma, Dikes.SillPhase, 20)
            move_rock!(Arrays, Grid, Num, U, Dikes.BackgroundPhase; sill)

            model.next_sill[k] += V / (zone.frac * Q)
            model.injected += V
            if in_chamber(xc, yc, zc)
                model.recharge += V
            end
        end
    end
    Arrays.Tnew .= Arrays.T
    return Tracers
end

# --- Eruptions -----------------------------------------------------------------------------------------------------------
eruption_params = EruptionParams(
    ΔP_crit = 10.0e6, σ_barrier = 5.0e6, w_dike = 5.0, μ_shear = 1.0e10, m_w = 0.04, h_melt_min = 400.0,
    melt_viscosity = LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.9606K, η0 = 1Pas), solubility = Liu2005_Solubility(),
    shell_from_T = true, shell_z_split = z_layers,                                  # layered host rock: one creep law per layer
    shell_rheology = (
        SetDislocationCreep(GeoParams.Dislocation.wet_quartzite_Ueda_2008),
        SetDislocationCreep(GeoParams.Dislocation.granite_Tirel_2008),
        SetDislocationCreep(GeoParams.Dislocation.wet_anorthite_Rybacki_2006),
    ),
)
chamber = EruptionState()
margin = 1.0e3                                                                      # sills this close to the chamber recharge it [m]
coarsen = max(2, round(Int, 200 / res))                                             # chamber cells are grouped into sources ≈ 200 m apart for the displacement field
function in_chamber(x, y, z)
    xc, yc, R = model.chamber_xy
    return chamber.init && chamber.z_lo - margin <= z <= chamber.z_hi + margin && hypot(x - xc, y - yc) <= R + margin
end

"Phases on every `s`-th grid point; a point shows the sill phase if any cell of its block holds a sill, so that sills thinner than the snapshot grid stay visible."
function coarse_phases(P, s, phase)
    block(I) = view(P, map((i, n) -> i:min(i + s - 1, n), Tuple(I), size(P))...)
    return [any(==(phase), block(I)) ? phase : P[I] for I in CartesianIndices(map(n -> 1:s:n, size(P)))]
end

"Saves every `snapshot_stride`-th grid point of the current state, for the movie."
function save_snapshot(Num, Arrays, coords)
    s = snapshot_stride
    sub(A) = Array(A)[1:s:end, 1:s:end, 1:s:end]
    return jldsave(
        joinpath(outdir, "snap_" * lpad(Num.it, 7, '0') * ".jld2");
        time_kyr = kyr_before_present(Num.time), x = coords[1][1:s:end], y = coords[2][1:s:end], z = coords[3][1:s:end],
        T = Float32.(sub(Arrays.T)), ϕ = round.(UInt8, 255 .* clamp.(sub(Arrays.ϕ), 0, 1)), Z0 = Float32.(sub(model.Z0)),
        Phases = Int8.(coarse_phases(Array(Arrays.Phases), s, sill_phase)), Zsurf = Array(Arrays.Zsurf)[1:s:end, 1:s:end],
        injected_km3 = model.injected / km³, events = copy(model.events),
    )
end

"Callback after every time step: the tracers, the overpressure of the chamber and its eruptions, and the snapshots."
function erupt_and_save(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters)
    Tracers = MTK_GMG.MTK_updateTracers(Grid, Arrays, Tracers, Dikes, time_props, Num)
    coords = Tuple(Grid.coord1D)

    if model.P_lith === nothing
        model.P_lith = (similar(Arrays.P), similar(Arrays.Rho))
    end
    P_lith, ρ = model.P_lith
    @. ρ = ifelse(Arrays.Phases == Num.AirPhase, 0.0, Arrays.Rho)
    MagmaThermoKinematics.lithostatic_pressure!(P_lith, ρ, 9.81, step(coords[end]))
    ev = chamber_step!(
        chamber, eruption_params, Arrays.ϕ, Arrays.T, coords; recharge = model.recharge / Num.dt, Δt = Num.dt, time = Num.time,
        T_background = T_geotherm, margin = 500.0, resolve = 0.5, P = P_lith, Zsurf = Arrays.Zsurf
    )
    model.recharge = 0.0
    if ev !== nothing
        z_surface = maximum(filter(!isnan, Array(Arrays.Zsurf)))
        U, _ = erupt!(Arrays.T, similar(Arrays.T), Tracers, coords, ev; coarsen, z_surface, Zsurf = Arrays.Zsurf, halfspace = :mindlin)
        move_rock!(Arrays, Grid, Num, U, Dikes.BackgroundPhase)
        Arrays.Tnew .= Arrays.T
        push!(model.events, (time_kyr = kyr_before_present(Num.time), V_km3 = ev.f * ev.geo.V / km³, depth_km = -ev.geo.zc / 1.0e3))
    end
    if chamber.init                                                                 # centre and radius of the chamber, for the recharge
        x_cells = [coords[1][I[1]] for I in chamber.cells]
        y_cells = [coords[2][I[2]] for I in chamber.cells]
        R = sqrt(chamber.V / π / max(chamber.z_hi - chamber.z_lo, res))
        model.chamber_xy = (sum(x_cells) / length(x_cells), sum(y_cells) / length(y_cells), R)
    end

    snapshot = 1.0e3 * SecYear * snapshot_kyr
    if Num.it == 1 || floor(Num.time / snapshot) > floor((Num.time - Num.dt) / snapshot)
        save_snapshot(Num, Arrays, coords)
    end
    return Tracers
end

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it, 50) == 0
        ΔP = chamber.init ? round((chamber.P - chamber.P_lith) / 1.0e6, digits = 2) : NaN
        t = round(kyr_before_present(Num.time), digits = 2)
        println("$(Num.it): t = $t kyr, max T $(round(maximum(Arrays.Tnew))) °C, injected $(round(model.injected / km³, digits = 2)) km³, overpressure $ΔP MPa, eruptions $(length(model.events)), wall $(round((time() - wall_start) / 60, digits = 1)) min")
        flush(stdout)
    end
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
    SimName = "HighRes3D", dim = 3, maxTime_Myrs = t_run * 1.0e-3, κ_time = κ, fac_dt = 4 * dt_yr * SecYear * κ / res^2,
    solver = :implicit_enthalpy, ω = 1.0, convection = true, convection_phi = 0.6, convection_boundary_layer = false,
    Tsurface_Celsius = 10.0, sticky_air_BC = true, AirPhase = 0, move_free_surface = true, TtPath_steps = 5,
    Output_VTK = false, SaveOutput_steps = 10^9, CreateFig_steps = 10^9, print_injections = false, backend = backend, seed = 2206,
)
Sill_params = SillParams(
    sill = PennyShapedSill(Center = Point3(0.0, 0.0, -5.0e3)m, Angle = Vec2(0.0, 0.0)NoUnits, R = 1.0e3m, H = 10.0m),   # not used: the callback injects the sills
    InjectionInterval_year = 1.0e15, SillPhase = sill_phase, BackgroundPhase = 1, T_in_Celsius = T_magma,
)

# --- Run --------------------------------------------------------------------------------------------------------------------
callbacks = (; inject_dikes = inject_two_zones, updateTracers = erupt_and_save)
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params; CartData_input = Data_3D, callbacks)
erupted = sum(e.V_km3 for e in model.events; init = 0.0)
println("done: $(Num.it) steps on $(length(xs))×$(length(ys))×$(length(zs)) points in $(round((time() - wall_start) / 60, digits = 1)) min; injected $(round(model.injected / km³, digits = 1)) km³, $(length(model.events)) eruptions, $(round(erupted, digits = 2)) km³ erupted")
jldsave(
    joinpath(outdir, "final.jld2"); events = model.events, tracers = Tracers, Zinit = Zinit, Zpres = Zpres, x = collect(xs), y = collect(ys), z = collect(zs),
    T = Float32.(Array(Arrays.T)), ϕ = Float32.(Array(Arrays.ϕ)), Z0 = Float32.(Array(model.Z0)), Phases = Int8.(Array(Arrays.Phases)), Zsurf = Array(Arrays.Zsurf),
)

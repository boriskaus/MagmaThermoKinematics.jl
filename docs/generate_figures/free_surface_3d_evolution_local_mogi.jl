# Generates docs/src/assets/movies/FreeSurface3D_evolution_local_mogi.gif for
# docs/src/man/free_surface.md.
#
# Identical setup to free_surface_3d_evolution.jl, but with
# EruptionParams.deflation_model = :local_mogi instead of the default
# :column, so the two animations can be compared directly: :column produces a
# flat-bottomed, sharp-edged pit (no lateral coupling between columns);
# :local_mogi produces a smooth, shape-tracking collapse bowl.
#
# Run from the repository root: `julia --project=. docs/generate_figures/free_surface_3d_evolution_local_mogi.jl`

using MagmaThermoKinematics
environment!(:cpu, Float64, 3)
import MagmaThermoKinematics.MTK_GMG
import MagmaThermoKinematics.MTK_GMG_3D
using GeoParams, InjectSills, Parameters
using Plots
gr()

@with_kw mutable struct TimeDepProps3DFS <: MagmaThermoKinematics.TimeDependentProperties
    Time_vec::Vector{Float64} = []
    MeltFraction::Vector{Float64} = []
    Tav_magma::Vector{Float64} = []
    Tmax::Vector{Float64} = []
    z_snap::Vector{Matrix{Float64}} = []
end

function MTK_GMG.MTK_update_TimeDepProps!(time_props::TimeDepProps3DFS, Grid::MagmaThermoKinematics.GridData,
        Num::MagmaThermoKinematics.NumericalParameters, Arrays::NamedTuple,
        Mat_tup::Tuple, Dikes::MagmaThermoKinematics.SillParameters)
    push!(time_props.Time_vec, Num.time)
    isnothing(FS_fs.z_surf) || push!(time_props.z_snap, copy(FS_fs.z_surf))
    return nothing
end

Mat_fs = (
    SetMaterialParams(Name = "Air", Phase = 0,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 0.0J / kg),
        Conductivity = ConstantConductivity(k = 3Watt / K / m),
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())),
    SetMaterialParams(Name = "Crust", Phase = 1,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
        Conductivity = T_Conductivity_Whittington_parameterised(),
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())),
    SetMaterialParams(Name = "Sill", Phase = 2,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 2.67e5J / kg),
        Conductivity = T_Conductivity_Whittington_parameterised(),
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_Quadratic(T_s = (700 + 273.15)K, T_l = (1100 + 273.15)K))),
)

Num_fs = NumParam(Nx = 41, Ny = 41, Nz = 41, W = 10e3, L = 10e3, H = 10e3,
    SimName = "docs_free_surface_3d_fig_local_mogi", maxTime_Myrs = 0.0025, fac_dt = 0.2, ω = 0.5, verbose = false,
    CreateFig_steps = 100000, SaveOutput_steps = 100000,
    plot_tracers = false, advect_polygon = false, USE_GPU = false)

sill = PennyShapedSill(Center = Point3(0.0e3, 0.0e3, -6.0e3)m, Angle = Vec2(0.0, 0.0) * NoUnits,
    W = 500m, H = 150m, E = 1.5e10Pa, ν = 0.3NoUnits)
Sill_params = SillParams(sill = sill, InjectionInterval_year = 200, nTr_dike = 200,
    SillPhase = 2, BackgroundPhase = 1, T_in_Celsius = 1000, SillsAbove = -8e3)

Erupt_fs = EruptionParams(erupt = true, ϕ_erupt = 0.5, V_crit_km3 = 5e-4,
    erupt_efficiency = 0.5, deflate = true, deflation_model = :local_mogi)
FS_fs = FreeSurfaceParams(free_surface = true, air_phase = 0, Tair = 0.0, z0 = -2.0e3)

Grid, Arrays_fs, _, Dikes_fs, tp = MTK_GMG_3D.MTK_GeoParams_3D(Mat_fs, Num_fs, Sill_params;
    Erupt = Erupt_fs, FS = FS_fs, time_props = TimeDepProps3DFS())

println("n_eruptions = ", Erupt_fs.n_eruptions, "; n_snapshots = ", length(tp.z_snap))
println("net uplift (m) = ", maximum(tp.z_snap[end]) - maximum(tp.z_snap[1]))

x_km = Grid.coord1D[1] ./ 1e3
y_km = Grid.coord1D[2] ./ 1e3

stride = max(1, length(tp.z_snap) ÷ 15)
frames = 1:stride:length(tp.z_snap)
zlims  = extrema(reduce(vcat, tp.z_snap)) ./ 1e3

anim = @animate for k in frames
    t_kyr = tp.Time_vec[k] / (3600 * 24 * 365.25 * 1e3)
    surface(x_km, y_km, tp.z_snap[k]' ./ 1e3;
        xlabel = "x (km)", ylabel = "y (km)", zlabel = "z_surf (km)",
        title = "3D free-surface evolution (:local_mogi), t=$(round(t_kyr, digits = 2)) kyr",
        c = :viridis, clims = zlims, zlims = zlims, camera = (45, 35),
        size = (700, 500), dpi = 110, colorbar = false)
end

outfile = joinpath(@__DIR__, "..", "src", "assets", "movies", "FreeSurface3D_evolution_local_mogi.gif")
mkpath(dirname(outfile))
gif(anim, outfile, fps = 3)
println("saved ", outfile)

rm(Num_fs.SimName, recursive = true, force = true)

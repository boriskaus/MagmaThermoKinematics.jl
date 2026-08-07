# Example: generic eruption + free-surface evolution in 2D, using the
# default *kinematic* trigger (`EruptionParams.overpressure = false`, the
# package default): an eruption fires once the eruptible melt volume reaches
# `V_crit`, withdrawing a fixed fraction `erupt_efficiency` of it.
#
#   • Eruptions    — `erupt_magma!` via the `MTK_erupt!` callback.
#   • Free surface — a kinematic sticky-air topography via the
#       `MTK_free_surface!` callback: sill injection inflates the ground
#       surface, eruption deflation subsides it, and cells above the
#       topography are stamped as "air".
#
# This is the baseline of a 4-script eruption-trigger series:
#   - MTK_GMG_2D_Eruption_Kinematic.jl                (this file)
#   - MTK_GMG_2D_Eruption_DegruyterHuber.jl            (same setup, physical ΔP_crit trigger)
#   - MTK_GMG_3D_Eruption_DegruyterHuber_FlatTopo.jl   (3D, physical trigger, flat topography)
#   - MTK_GMG_3D_Eruption_DegruyterHuber_Lanin.jl      (3D, physical trigger, real GMT topography)
# See docs/src/man/eruptions.md and docs/src/man/free_surface.md for the
# underlying physics.

const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading ParallelStencil
end
using ParallelStencil, ParallelStencil.FiniteDifferences2D

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
    @init_parallel_stencil(CUDA, Float64, 2)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
    @init_parallel_stencil(Threads, Float64, 2)
end
using MagmaThermoKinematics.Diffusion2D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields2D
using MagmaThermoKinematics.MTK_GMG_2D
using GeophysicalModelGenerator
using GeoParams, Random
using Plots                             # plots
using MagmaThermoKinematics.MTK_GMG     # Allow overwriting user routines

Random.seed!(1234);     # use the same random seed, such that we can reproduce results

println("Eruption (kinematic trigger) + free-surface example of the MTK - GMG integration (2D)")

# ------------------------------------------------------------------
# Overwrite some of the MTK user callbacks
# ------------------------------------------------------------------

# Print a short status line, including eruption + surface diagnostics.
# (Erupt / FS are visible here because this script closes over them below.)
function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it, 50) == 0
        zmin = isnothing(FS.z_surf) ? NaN : minimum(FS.z_surf)
        println("$(Num.it), t=$(round(Num.time/Num.SecYear/1e3,digits=2)) kyr; " *
                "max(T)=$(round(maximum(Arrays.Tnew))) °C; " *
                "n_eruptions=$(Erupt.n_eruptions), erupted=$(round(Erupt.erupted_volume/1e9,digits=3)) km³; " *
                "min(z_surf)=$(round(zmin/1e3,digits=3)) km")
    end
    return nothing
end

# Visualize temperature, phases, and the free-surface topography line.
function MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it, Num.CreateFig_steps) == 0
        x_1d       = Grid.coord1D[1]/1e3
        z_1d       = Grid.coord1D[2]/1e3
        temp_data  = Array(Arrays.Tnew)'
        phase_data = Array(Arrays.Phases)'
        t          = Num.time/SecYear/1e3

        p = plot(layout=grid(1,2))
        Plots.heatmap!(p[1], x_1d, z_1d, temp_data,  c=:viridis, xlabel="x [km]", ylabel="z [km]",
                       title="Temperature, t=$(round(t,digits=1)) kyr", aspect_ratio=:equal, ylimits=(-20,0))
        Plots.heatmap!(p[2], x_1d, z_1d, phase_data, c=:viridis, xlabel="x [km]", ylabel="z [km]",
                       title="Phases (0=air)", aspect_ratio=:equal, ylimits=(-20,0))
        if !isnothing(FS.z_surf)                              # overlay the moving surface
            Plots.plot!(p[1], x_1d, Array(FS.z_surf)/1e3, c=:red, lw=2, label="free surface")
        end
        display(p)
    end
    return nothing
end

# Initialize T and phases. Cells above the initial surface elevation are air.
function MTK_GMG.MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters)
    Arrays.T_init .= @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm          # linear geotherm

    z_surf0 = -2.0e3                                                          # initial flat surface [m]
    @views Arrays.Phases[Arrays.Z .> z_surf0] .= 0                            # air above the surface
    Arrays.Phases_init .= Arrays.Phases

    if Num.Output_VTK
        name    = joinpath(Num.SimName, Num.SimName*".pvd")
        Num.pvd = movie_paraview(name=name, Initialize=true)
    end
    return nothing
end
# ------------------------------------------------------------------

# Numerical parameters
Num = NumParam( Nx                   = 135,
                Nz                   = 135,
                W                    = 20e3,
                H                    = 20e3,
                SimName              = "Eruption2D_Kinematic",
                maxTime_Myrs         = 0.02,
                fac_dt               = 0.2,
                ω                    = 0.5,
                CreateFig_steps      = 50,
                SaveOutput_steps     = 200,
                USE_GPU              = USE_GPU,
                AddRandomSills       = true,
                RandomSills_timestep = 10)

# Sill: a penny-shaped crack (Sun 1969 solution) at 7 km depth.
sill = PennyShapedSill(Center=Point2(10.0e3, -7.0e3)m, W=2.5e3m, H=250m, E=1.5e10Pa, ν=0.3NoUnits)

Sill_params = SillParams(
    sill                   = sill,
    InjectionInterval_year = 100,
    nTr_dike               = 300*2,
    SillPhase              = 2,
    BackgroundPhase        = 1,
    T_in_Celsius           = 1000,
    SillsAbove             = -12e3,
)

# `AddRandomSills` re-injects every `RandomSills_timestep` diffusion steps, so
# the actual injection cadence tracks the (resolution-dependent) timestep
# rather than the fixed `InjectionInterval_year` above — this is what lets the
# eruptible melt volume actually build up past `V_crit` within a short run.
if Num.AddRandomSills
    Sill_params.InjectionInterval = Num.dt * Num.RandomSills_timestep
    Sill_params.InjectionInterval_year = Sill_params.InjectionInterval / SecYear
end

# Eruption parameters: erupt + deflate once the eruptible volume reaches
# `V_crit` — the default *kinematic* trigger (`overpressure = false`).
# Volumes are in km³ (3D) — even in this 2D run, because `out_of_plane_3D` lifts
# the per-unit-depth eruptible volume to a true 3D volume via a Gaussian out-of-
# plane profile, so `V_crit_km3` is directly comparable to the injected volume.
Erupt = EruptionParams(
    erupt            = true,
    ϕ_erupt          = 0.5,        # mobile-melt threshold
    EruptAbove       = -12e3,      # only melt shallower than this is eruptible (excludes deep geotherm melt)
    V_crit_km3       = 10e0,       # critical eruptible volume [km³, 3D]
    erupt_efficiency = 0.5,        # fraction of mobile melt withdrawn per event
    deflate          = true,       # also subside the chamber (deflation_model defaults to :local_mogi)
    out_of_plane_3D  = true,       # 2D eruptible volume → 3D via Gaussian out-of-plane (km³)
    T_min            = 0.0,        # floor for the thermal-extraction cooling
)

# Free-surface parameters: sticky-air topography, starts flat at -2 km
FS = FreeSurfaceParams(
    free_surface = true,
    air_phase    = 0,
    Tair         = 0.0,            # = Tsurface
    z0           = -2.0e3,
)

# Material properties: phase 0 = air, 1 = host rock, 2 = intruded sills
MatParam = (SetMaterialParams(Name="Air", Phase=0,
                              Density      = ConstantDensity(ρ=2700kg/m^3),
                              LatentHeat   = ConstantLatentHeat(Q_L=0.0J/kg),
                              Conductivity = ConstantConductivity(k=3Watt/K/m),
                              HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K)),
            SetMaterialParams(Name="Host rock", Phase=1,
                              Density      = ConstantDensity(ρ=2700kg/m^3),
                              LatentHeat   = ConstantLatentHeat(Q_L=2.55e5J/kg),
                              Conductivity = T_Conductivity_Whittington_parameterised(),
                              HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                              Melting      = SmoothMelting(MeltingParam_4thOrder())),
            SetMaterialParams(Name="Intruded rocks", Phase=2,
                              Density      = ConstantDensity(ρ=2700kg/m^3),
                              LatentHeat   = ConstantLatentHeat(Q_L=2.67e5J/kg),
                              Conductivity = T_Conductivity_Whittington_parameterised(),
                              HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                              Melting      = SmoothMelting(MeltingParam_Quadratic(T_s=(700+273.15)K, T_l=(1100+273.15)K))))

# Call the main code, passing the eruption + free-surface parameters
Grid, Arrays, Tracers, Dikes, time_props =
    MTK_GeoParams_2D(MatParam, Num, Sill_params; Erupt=Erupt, FS=FS)

println("Done: $(Erupt.n_eruptions) eruptions, total erupted melt = " *
        "$(round(Erupt.erupted_volume/1e9, digits=3)) km³; " *
        "final surface min/max = $(round(minimum(FS.z_surf)/1e3,digits=3)) / " *
        "$(round(maximum(FS.z_surf)/1e3,digits=3)) km")

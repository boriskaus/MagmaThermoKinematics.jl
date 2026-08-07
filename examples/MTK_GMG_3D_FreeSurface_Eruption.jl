# Example: a 3D MTK – GMG run with the two newer capabilities wired into the
# time loop (the 3D analogue of examples/MTK_GMG_2D_FreeSurface_Eruption.jl):
#
#   • Eruptions      (issue 2) — `erupt_magma!` via the `MTK_erupt!` callback:
#       once the eruptible melt volume reaches `V_crit`, a fraction of the mobile
#       melt is withdrawn (thermally) and the chamber deflates (column subsidence).
#       In 3D the eruptible volume is already a true volume, so `V_crit_km3` is a
#       genuine km³ (the 2D Gaussian out-of-plane lift, `out_of_plane_3D`, is a
#       no-op here).
#   • Free surface   (issue 4) — a kinematic sticky-air topography via the
#       `MTK_free_surface!` callback: sill injection inflates the ground surface,
#       eruption deflation subsides it, and cells above the topography are
#       stamped as "air". A moving free surface auto-enables `deform_hostrock`,
#       so the phase column (host rock + sills) moves with the surface and the
#       air/rock interface tracks the topography.

const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading ParallelStencil
end
using ParallelStencil, ParallelStencil.FiniteDifferences3D

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 3)      # initialize parallel stencil in 3D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
    @init_parallel_stencil(CUDA, Float64, 3)
else
    environment!(:cpu, Float64, 3)      # initialize parallel stencil in 3D
    @init_parallel_stencil(Threads, Float64, 3)
end
using MagmaThermoKinematics.Diffusion3D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields3D
using MagmaThermoKinematics.MTK_GMG_3D
using GeophysicalModelGenerator
using GeoParams, Random
using Plots                             # plots
using MagmaThermoKinematics.MTK_GMG     # Allow overwriting user routines

Random.seed!(1234);     # use the same random seed, such that we can reproduce results

println("Free-surface + eruption example of the MTK - GMG integration (3D)")

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

# Visualize a vertical cross-section (y = 0 mid-plane) of temperature, phases,
# and the free-surface topography line along that slice.
function MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it, Num.CreateFig_steps) == 0
        x_1d       = Grid.coord1D[1]/1e3
        z_1d       = Grid.coord1D[3]/1e3
        jmid       = (Num.Ny + 1) ÷ 2                          # y = 0 slice
        temp_data  = Array(Arrays.Tnew)[:, jmid, :]'
        phase_data = Array(Arrays.Phases)[:, jmid, :]'
        t          = Num.time/SecYear/1e3

        p = plot(layout=grid(1,2))
        Plots.heatmap!(p[1], x_1d, z_1d, temp_data,  c=:viridis, xlabel="x [km]", ylabel="z [km]",
                       title="Temperature (y=0), t=$(round(t,digits=1)) kyr", aspect_ratio=:equal, ylimits=(-20,0))
        Plots.heatmap!(p[2], x_1d, z_1d, phase_data, c=:viridis, xlabel="x [km]", ylabel="z [km]",
                       title="Phases (0=air)", aspect_ratio=:equal, ylimits=(-20,0))
        if !isnothing(FS.z_surf)                               # overlay the moving surface (y=0 slice)
            Plots.plot!(p[1], x_1d, Array(FS.z_surf)[:, jmid]/1e3, c=:red, lw=2, label="free surface")
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

# Numerical parameters (3D ⇒ Ny > 0). The 3D grid is centred in x and y:
# x ∈ [-W/2, W/2], y ∈ [-L/2, L/2], z ∈ [-H, 0].
Num = NumParam( Nx                   = 100,
                Ny                   = 100,
                Nz                   = 100,
                W                    = 20e3,
                L                    = 20e3,
                H                    = 20e3,
                SimName              = "FreeSurface_Eruption_3D",
                maxTime_Myrs         = 0.02,
                fac_dt               = 0.2,
                ω                    = 0.5,
                CreateFig_steps      = 50,
                SaveOutput_steps     = 200,
                USE_GPU              = USE_GPU)

# Sill: a penny-shaped crack at 7 km depth, centred at x = y = 0.
# NOTE: this uses `PennyShapedSill` (the Sun-1969 penny-crack solution, like
# examples/MTK_GMG_3D_example.jl) as the well-behaved 3D source (max host-rock
# displacement ≈ H/2). In InjectSills ≥0.2.1, `W` is a radius and `H` the full
# thickness: volume = (4/3)π·W²·(H/2). The 3D `EllipticalIntrusion.hostrock_displacement`
# was grossly oversized in InjectSills 0.2.0 (≫ the H/2 opening it should
# produce, inflating the free surface by kilometres per injection); this is
# fixed in InjectSills ≥0.2.1, where EllipticalIntrusion is a usable 3D source
# once MTK's compat is bumped.
sill = PennyShapedSill(Center=Point3(0.0e3, 0.0e3, -7.0e3)m, Angle=Vec2(0.0, 0.0)*NoUnits,
                       W=2.5e3m, H=1000m, E=1.5e10Pa, ν=0.3NoUnits)

Sill_params = SillParams(
    sill                   = sill,
    InjectionInterval_year = 1000,
    nTr_dike               = 300*2,
    SillPhase              = 2,
    BackgroundPhase        = 1,
    T_in_Celsius           = 1000,
    SillsAbove             = -12e3,
)

# Eruption parameters: erupt + deflate once the eruptible volume is reached.
# Volumes are in km³ (3D). `out_of_plane_3D` is irrelevant in 3D (the eruptible
# volume is already a true volume) — it only matters for 2D runs.
Erupt = EruptionParams(
    erupt            = true,
    ϕ_erupt          = 0.5,        # mobile-melt threshold
    EruptAbove       = -12e3,      # only melt shallower than this is eruptible (excludes deep geotherm melt)
    V_crit_km3       = 5.0,        # critical eruptible volume [km³]
    erupt_efficiency = 0.5,        # fraction of mobile melt withdrawn per event
    deflate          = true,       # also subside the chamber (deflation_model defaults to :local_mogi)
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
                              HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                              ),
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
    MTK_GeoParams_3D(MatParam, Num, Sill_params; Erupt=Erupt, FS=FS)

println("Done: $(Erupt.n_eruptions) eruptions, total erupted melt = " *
        "$(round(Erupt.erupted_volume/1e9, digits=3)) km³; " *
        "final surface min/max = $(round(minimum(FS.z_surf)/1e3,digits=3)) / " *
        "$(round(maximum(FS.z_surf)/1e3,digits=3)) km")

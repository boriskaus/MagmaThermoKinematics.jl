# Example: a 3D MTK – GMG run with eruption + free-surface evolution, using
# the *physical* Degruyter & Huber (2014) chamber-overpressure trigger
# (`EruptionParams.overpressure = true`) and a flat initial free-surface
# topography (no real terrain — see MTK_GMG_3D_Eruption_DegruyterHuber_Lanin.jl
# for a version that starts from a downloaded GMT topography).
#
# 3D analogue of MTK_GMG_2D_Eruption_DegruyterHuber.jl. In 3D the eruptible
# volume is already a true volume, so `V_crit_km3` (unused by the physical
# trigger) would be a genuine km³ and the 2D Gaussian out-of-plane lift,
# `out_of_plane_3D`, is a no-op.
#
# Part of a 4-script eruption-trigger series:
#   - MTK_GMG_2D_Eruption_Kinematic.jl                (2D, kinematic V_crit trigger)
#   - MTK_GMG_2D_Eruption_DegruyterHuber.jl            (2D, physical ΔP_crit trigger)
#   - MTK_GMG_3D_Eruption_DegruyterHuber_FlatTopo.jl   (this file)
#   - MTK_GMG_3D_Eruption_DegruyterHuber_Lanin.jl      (3D, physical trigger, real GMT topography)

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

println("Eruption (Degruyter & Huber ΔP_crit trigger) + free-surface example of the MTK - GMG integration (3D, flat topography)")

# ------------------------------------------------------------------
# Overwrite some of the MTK user callbacks
# ------------------------------------------------------------------

# Print a short status line, including eruption + surface + chamber diagnostics.
# (Erupt / FS are visible here because this script closes over them below.)
function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it, 50) == 0
        zmin = isnothing(FS.z_surf) ? NaN : minimum(FS.z_surf)
        ΔP   = Erupt.chamber.P - Erupt.chamber.P_lith
        println("$(Num.it), t=$(round(Num.time/Num.SecYear/1e3,digits=2)) kyr; " *
                "max(T)=$(round(maximum(Arrays.Tnew))) °C; " *
                "n_eruptions=$(Erupt.n_eruptions), erupted=$(round(Erupt.erupted_volume/1e9,digits=3)) km³; " *
                "ΔP=$(round(ΔP/1e6,digits=2)) MPa (crit=$(Erupt.ΔP_crit/1e6) MPa); " *
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
                SimName              = "Eruption3D_DegruyterHuber_FlatTopo",
                maxTime_Myrs         = 0.02,
                fac_dt               = 0.2,
                ω                    = 0.5,
                CreateFig_steps      = 50,
                SaveOutput_steps     = 200,
                USE_GPU              = USE_GPU)

# Sill: a penny-shaped crack (Sun 1969 solution) at 7 km depth, centred at x = y = 0.
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

# Eruption parameters: physical ΔP_crit trigger. `β_r`/`η_r` are chosen soft
# (not physically calibrated) so the chamber reaches ΔP_crit within this
# short demo run — for a real study, calibrate them against the host-rock
# elastic modulus and a wall-relaxation timescale for your system.
# `magma_phase` must match a `MatParam` phase whose `Density` drives the
# chamber-overpressure ODE — here phase 2 ("Intruded rocks"), whose `Density`
# is a `ThreePhase_Density` (melt+crystal+gas) and `Solubility` a
# `Liu2005_Solubility`. `m_h2o_total`/`X_co2` combine with that phase's
# `Solubility` (magma_density_fn) to diagnose the exsolved gas fraction from a
# 3 wt% total H₂O content, giving the second-boiling pressurization (E4) that
# a melt+crystal-only density law cannot produce.
Erupt = EruptionParams(
    erupt            = true,
    ϕ_erupt          = 0.5,        # mobile-melt threshold
    EruptAbove       = -12e3,      # only melt shallower than this is eruptible (excludes deep geotherm melt)
    deflate          = true,       # also subside the chamber (deflation_model defaults to :local_mogi)
    T_min            = 0.0,        # floor for the thermal-extraction cooling
    overpressure     = true,       # physical ΔP_crit trigger instead of kinematic V_crit
    magma_phase      = 2,          # Mat_tup phase whose Density/Solubility laws drive the ODE ("Intruded rocks")
    m_h2o_total      = 0.03,       # total dissolved+exsolved H2O content of the melt [mass fraction]
    ΔP_crit          = 20e6,       # rock-strength failure overpressure [Pa]
    β_r              = 1e9,        # host-rock elastic stiffness [Pa] (soft demo value)
    η_r              = 1e17,       # wall relaxation viscosity [Pa·s] (soft demo value)
    ΔP_relax         = 2e6,        # overpressure left right after a drain [Pa]
    ρ_melt           = 2400.0,     # melt density for the recharge mass rate [kg/m³]
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
                              # ρgas uses IdealGas_Density, not RedlichKwong_Density: the main
                              # solver's compute_density_ps! evaluates this Density law at every
                              # cell/iteration with just (T,P), unconditionally weighting all three
                              # sub-densities even where ϕ_gas=0 — RedlichKwong_Density is only
                              # fitted for 873-1173K/30-400MPa and returns NaN outside that window,
                              # which poisons the mixture even at zero weight (0*NaN = NaN).
                              Density      = ThreePhase_Density(ρmelt=ConstantDensity(ρ=2300kg/m^3),
                                                                 ρx=ConstantDensity(ρ=2700kg/m^3),
                                                                 ρgas=IdealGas_Density()),
                              Solubility   = Liu2005_Solubility(),  # H2O(-CO2) solubility law (Degruyter & Huber 2014)
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

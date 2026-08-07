# Example: a 3D MTK – GMG run with the *physical* Degruyter & Huber (2014)
# chamber-overpressure eruption trigger (`EruptionParams.overpressure = true`),
# on a real downloaded topography (Lanin volcano, Argentina/Chile) — the D&H
# analogue of examples/MTK_GMG_3D_Lanin.jl.
#
# Surface representation: like MTK_GMG_3D_Lanin.jl, the real terrain enters
# via `CartData_input` (GeophysicalModelGenerator's DEM-derived Phases/Temp
# grid, phase 0 = air above the topography), not the separate kinematic
# `FreeSurfaceParams` z_surf-array tracker used in the flat-topography example
# — that combination (CartData_input real terrain *plus* a tracked z_surf
# array) is not exercised elsewhere in this package, so it is deliberately
# not attempted here. `deform_hostrock = true` is enabled (unlike
# MTK_GMG_3D_Lanin.jl) so the air/rock interface still visibly evolves as
# sills inject and D&H eruptions deflate the chamber.
#
# Requires the `GMT` package and internet access on the *first* run only (the
# downloaded/projected topography is cached to `Topo_cart_Lanin3D.jld2` next
# to this script and reused — the same cache file MTK_GMG_3D_Lanin.jl uses,
# since both examples cover the same location).
#
# Part of a 4-script eruption-trigger series:
#   - MTK_GMG_2D_Eruption_Kinematic.jl                (2D, kinematic V_crit trigger)
#   - MTK_GMG_2D_Eruption_DegruyterHuber.jl            (2D, physical ΔP_crit trigger)
#   - MTK_GMG_3D_Eruption_DegruyterHuber_FlatTopo.jl   (3D, physical trigger, flat topography)
#   - MTK_GMG_3D_Eruption_DegruyterHuber_Lanin.jl      (this file)

using Random
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading ParallelStencil
end

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 3)      # initialize parallel stencil in 3D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
else
    environment!(:cpu, Float64, 3)      # initialize parallel stencil in 3D
end
using MagmaThermoKinematics.Diffusion3D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields3D
using MagmaThermoKinematics.MTK_GMG
using MagmaThermoKinematics.MTK_GMG_3D
using Random, GeoParams, GeophysicalModelGenerator

const rng = Random.seed!(1234);     # same seed such that we can reproduce results

# Print a short status line, including eruption + chamber diagnostics.
# (Erupt is visible here because this script closes over it below.)
function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it, 10) == 0
        ΔP = Erupt.chamber.P - Erupt.chamber.P_lith
        println("$(Num.it), $(round(Num.time/SecYear/1e3,digits=3)) kyrs; max(T)=$(round(maximum(Arrays.Tnew))); " *
                "n_eruptions=$(Erupt.n_eruptions), erupted=$(round(Erupt.erupted_volume/1e9,digits=3)) km³; " *
                "ΔP=$(round(ΔP/1e6,digits=2)) MPa (crit=$(Erupt.ΔP_crit/1e6) MPa)")
    end
    return nothing
end

println("===============================================")
println("     MTK model: D&H eruption trigger @ Lanin    ")
println("===============================================")
# -----------------------------

# Create 3D grid of the region (identical to MTK_GMG_3D_Lanin.jl, so the
# cached topography file can be shared between the two examples).
if !isfile(joinpath(@__DIR__,"Topo_cart_Lanin3D.jld2"))
    using GMT, Statistics
    println("Creating topography grid from GMG for Lanin 3D example...")
    Topo       =   import_topo(lon = [-71.9, -71.1], lat=[-39.95, -39.35], file="@earth_relief_01s.grd")
    proj       =   ProjectionPoint(; Lat=mean(Topo.lat.val), Lon=mean(Topo.lon.val))
    Topo_cart  =   convert2CartData(Topo, proj)

    Xt,Yt,Zt   =   xyz_grid(-20:.025:20,-20:.025:20,0)
    Topo_cart  =   project_CartData(CartData(Xt,Yt,Zt,(Zt=Zt,)), Topo, proj)
    write_paraview(Topo_cart,joinpath(@__DIR__,"Topo_cart_Lanin3D"));

    save_GMG(joinpath(@__DIR__,"Topo_cart_Lanin3D"), Topo_cart)
end

Topo_cart = load_GMG(joinpath(@__DIR__,"Topo_cart_Lanin3D"))
x_range     =   (-20,20)
z_range     =   (-40,5)
Nx          =   100
Ny          =   100
Nz          =   100
X,Y,Z       =   xyz_grid(range(x_range[1],x_range[2], length=Nx),range(x_range[1],x_range[2], length=Ny),range(z_range[1],z_range[2], length=Nz))
Data_3D     =   CartData(X,Y,Z,(Phases=zeros(Int64,size(X)),Temp=zeros(size(X))));       # 3D dataset

# Intersect with topography: phase 0 = air above the real terrain
Below = below_surface(Data_3D, Topo_cart)
Data_3D.fields.Phases[Below] .= 1

# Set Moho
ind = findall(Data_3D.z.val .< -30.0)
Data_3D.fields.Phases[ind] .= 2

# Set T (linear geotherm, floored at the surface temperature)
gradient = 30
Data_3D.fields.Temp .= -Data_3D.z.val*gradient
ind = findall(Data_3D.fields.Temp .< 10.0)
Data_3D.fields.Temp[ind] .= 10.0

# Define numerical parameters
Num = NumParam( SimName="Eruption3D_DegruyterHuber_Lanin", axisymmetric=false,
                maxTime_Myrs=0.02,
                Nx = Nx, Ny = Ny, Nz = Nz,
                fac_dt=0.2,
                SaveOutput_steps=20, CreateFig_steps=1000, plot_tracers=false, advect_polygon=false,
                USE_GPU=USE_GPU,
                AddRandomSills = true, RandomSills_timestep=5,
                deform_hostrock = true);    # let the air/rock interface evolve with injections + eruptions

# Sill: a penny-shaped crack (Sun 1969 solution) at 7 km depth.
sill = PennyShapedSill(Center=Point3(0.0, 0.0, -7.0e3) * m, Angle=Vec2(0.0, 0.0) * NoUnits, W=2.5e3 * m, H=1000 * m, E=1.5e10 * Pa, ν=0.3 * NoUnits)

Sill_params = SillParams(
    sill                    = sill,
    InjectionInterval_year  = 500,
    nTr_dike                = 300,
    H_ran                   = 5000,
    W_ran                   = 5000,
    SillPhase               = 3,
    BackgroundPhase         = 1,
)

# Keep random sill relocation and actual injection in sync.
if Num.AddRandomSills
    Sill_params.InjectionInterval = Num.dt * Num.RandomSills_timestep
    Sill_params.InjectionInterval_year = Sill_params.InjectionInterval / SecYear
end

# Eruption parameters: physical ΔP_crit trigger. `β_r`/`η_r` are chosen soft
# (not physically calibrated) so the chamber reaches ΔP_crit within this
# short demo run — for a real study, calibrate them against the host-rock
# elastic modulus and a wall-relaxation timescale for your system.
# `magma_phase` must match a `MatParam` phase whose `Density` drives the
# chamber-overpressure ODE — here phase 3 ("Dikes", the injected material),
# whose `Density` is a `ThreePhase_Density` (melt+crystal+gas) and
# `Solubility` a `Liu2005_Solubility`. `m_h2o_total`/`X_co2` combine with
# that phase's `Solubility` (magma_density_fn) to diagnose the exsolved gas
# fraction from a 3 wt% total H₂O content, giving the second-boiling
# pressurization (E4) that a melt+crystal-only density law cannot produce.
Erupt = EruptionParams(
    erupt            = true,
    ϕ_erupt          = 0.5,        # mobile-melt threshold
    deflate          = true,       # also subside the chamber (deflation_model defaults to :local_mogi)
    T_min            = 0.0,        # floor for the thermal-extraction cooling
    overpressure     = true,       # physical ΔP_crit trigger instead of kinematic V_crit
    magma_phase      = 3,          # Mat_tup phase whose Density/Solubility laws drive the ODE ("Dikes")
    m_h2o_total      = 0.03,       # total dissolved+exsolved H2O content of the melt [mass fraction]
    ΔP_crit          = 20e6,       # rock-strength failure overpressure [Pa]
    β_r              = 1e9,        # host-rock elastic stiffness [Pa] (soft demo value)
    η_r              = 1e17,       # wall relaxation viscosity [Pa·s] (soft demo value)
    ΔP_relax         = 2e6,        # overpressure left right after a drain [Pa]
    ρ_melt           = 2400.0,     # melt density for the recharge mass rate [kg/m³]
)

# Define parameters for the different phases
MatParam     = (SetMaterialParams(Name="Air", Phase=0,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                                Conductivity = ConstantConductivity(k=3Watt/K/m),          # in case we use constant k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),         # Marxer & Ulmer melting

                SetMaterialParams(Name="Crust", Phase=1,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting

                SetMaterialParams(Name="Mantle", Phase=2,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K)),

                SetMaterialParams(Name="Dikes", Phase=3,
                                # ρgas uses IdealGas_Density, not RedlichKwong_Density: the main
                                # solver's compute_density_ps! evaluates this Density law at every
                                # cell/iteration with just (T,P), unconditionally weighting all three
                                # sub-densities even where ϕ_gas=0 — RedlichKwong_Density is only
                                # fitted for 873-1173K/30-400MPa and returns NaN outside that window,
                                # which poisons the mixture even at zero weight (0*NaN = NaN).
                                Density    = ThreePhase_Density(ρmelt=ConstantDensity(ρ=2300kg/m^3),
                                                                 ρx=ConstantDensity(ρ=2700kg/m^3),
                                                                 ρgas=IdealGas_Density()),
                                Solubility = Liu2005_Solubility(),  # H2O(-CO2) solubility law (Degruyter & Huber 2014)
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder()))      # Marxer & Ulmer melting
                )

# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GMG_3D.MTK_GeoParams_3D(MatParam, Num, Sill_params, CartData_input=Data_3D; Erupt=Erupt); # start the main code

println("Done: $(Erupt.n_eruptions) eruptions, total erupted melt = " *
        "$(round(Erupt.erupted_volume/1e9, digits=3)) km³")

Data_set3D_out = Data_3D;
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "Temperature[C]",  Float32.(Array(Arrays.Tnew )));
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "Temp",         Float32.(Array(Arrays.Tnew)));
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "Phases",       Int32.(Array(Arrays.Phases)));
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "MeltFraction", Float32.(Array(Arrays.ϕ)));
save_GMG(joinpath(Num.SimName,"Lanin3D_DegruyterHuber_final"), Data_set3D_out)
write_paraview(Data_3D, joinpath(Num.SimName,"Lanin3D_DegruyterHuber_final"))

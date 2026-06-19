using Test, LinearAlgebra, SpecialFunctions, Random
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end
using InjectSills

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
end
import MagmaThermoKinematics.Diffusion2D # load module AFTER calling environment!()
import MagmaThermoKinematics.MTK_GMG_2D
using GeophysicalModelGenerator, GeoParams

const rng = Random.seed!(1234);     # same seed such that we can reproduce results

# Import a few routines, so we can overwrite them below
import MagmaThermoKinematics.MTK_GMG
import MagmaThermoKinematics: inject_sills, km³, kyr, PhasesFromTracers!

function _build_injectsill(Dikes)
    if Dikes.Type == "CylindricalDike_TopAccretion"
        return CylindricalDikeTopAccretion(Center=Point2(Dikes.Center[1], Dikes.Center[2]) * m,
                                           Angle=Vec1(Dikes.Angle[1]) * NoUnits,
                                           W=Dikes.W_in * m,
                                           H=Dikes.H_in * m)
    elseif Dikes.Type == "EllipticalIntrusion" || Dikes.Type == "ElasticDike"
        return EllipticalIntrusion(Center=Point2(Dikes.Center[1], Dikes.Center[2]) * m,
                                   Angle=Vec1(Dikes.Angle[1]) * NoUnits,
                                   W=Dikes.W_in * m,
                                   H=Dikes.H_in * m)
    elseif Dikes.Type == "InjectSills"
        isnothing(Dikes.sill) && error("Dikes.Type='InjectSills' requires Dikes.sill to be set")
        return Dikes.sill
    else
        error("Unsupported Dikes.Type for InjectSills callback in test_MTK_GMG_2D: $(Dikes.Type)")
    end
end

@eval MTK_GMG begin
function MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructVector, Tnew_cpu)
    inj_counter = hasproperty(Dikes, :sill_inj) ? Dikes.sill_inj : Dikes.dike_inj
    if floor(Num.time / Dikes.InjectionInterval) > inj_counter
        inj_counter = floor(Num.time / Dikes.InjectionInterval)
        if hasproperty(Dikes, :sill_inj)
            Dikes.sill_inj = inj_counter
        else
            Dikes.dike_inj = inj_counter
        end

        if Num.dim == 2
            T_bottom = Array(@view Arrays.T[:, 1])
        else
            T_bottom = Array(@view Arrays.T[:, :, 1])
        end

        IS = getproperty(parentmodule(@__MODULE__), :InjectSills)
        m_unit = getproperty(parentmodule(@__MODULE__), :m)
        no_unit = getproperty(parentmodule(@__MODULE__), :NoUnits)

        if hasproperty(Dikes, :sill) && !isnothing(Dikes.sill)
            sill = Dikes.sill
        elseif Dikes.Type == "CylindricalDike_TopAccretion"
            sill = IS.CylindricalDikeTopAccretion(Center=IS.Point2(Dikes.Center[1], Dikes.Center[2]) * m_unit,
                                                 Angle=IS.Vec1(Dikes.Angle[1]) * no_unit,
                                                 W=Dikes.W_in * m_unit,
                                                 H=Dikes.H_in * m_unit)
        elseif Dikes.Type == "EllipticalIntrusion" || Dikes.Type == "ElasticDike"
            sill = IS.EllipticalIntrusion(Center=IS.Point2(Dikes.Center[1], Dikes.Center[2]) * m_unit,
                                         Angle=IS.Vec1(Dikes.Angle[1]) * no_unit,
                                         W=Dikes.W_in * m_unit,
                                         H=Dikes.H_in * m_unit)
        else
            error("Unsupported Dikes.Type for InjectSills callback in test_MTK_GMG_2D: $(Dikes.Type)")
        end
        poly = hasproperty(Dikes, :sill_poly) ? Dikes.sill_poly : Dikes.dike_poly
        if Num.advect_polygon == true && isempty(poly)
            if hasproperty(Dikes, :sill_poly)
                Dikes.sill_poly = InjectSills.dike_polygon(sill)
            else
                Dikes.dike_poly = InjectSills.dike_polygon(sill)
            end
        end

        copyto!(Tnew_cpu, Arrays.T)
        intrusion_phase = hasproperty(Dikes, :SillPhase) ? Dikes.SillPhase : Dikes.DikePhase
        Tracers, Tnew_cpu, Vol, _, _ = getproperty(parentmodule(@__MODULE__), :inject_sills)(Tracers, Tnew_cpu, Grid.coord1D, sill, Dikes.T_in_Celsius, intrusion_phase, Dikes.nTr_dike)

        if Num.flux_bottom_BC == false
            if Num.dim == 2
                Tnew_cpu[:, 1] .= T_bottom
            else
                Tnew_cpu[:, :, 1] .= T_bottom
            end
        end

        Arrays.T .= DataArray(Tnew_cpu)
        Dikes.InjectVol += Vol
        Qrate = Dikes.InjectVol / Num.time
        Dikes.Qrate_km3_yr = Qrate * SecYear / km³
        println("  Added new dike; time=$(Num.time / kyr) kyrs, total injected magma volume = $(Dikes.InjectVol / km³) km³; rate Q= $(Dikes.Qrate_km3_yr) km³yr⁻¹")

        if length(Mat_tup) > 1
            Phases = Array(Arrays.Phases)               # CPU copy — must be captured + written back
            PhasesFromTracers!(Phases, Grid, Tracers, BackgroundPhase=Dikes.BackgroundPhase, InterpolationMethod="Constant")

            if Num.keep_init_RockPhases == true
                Phases_init = Array(Arrays.Phases_init)
                for i in eachindex(Phases)
                    if Phases[i] != intrusion_phase
                        Phases[i] = Phases_init[i]
                    end
                end
            end
            Arrays.Phases .= DataArray(Phases)
        end
    end

    return Tracers
end
end

@testset "MTK_GMG_2D" begin
#=
    function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
        println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
        return nothing
    end
=#
# Test setup
println("===============================================")
println("Testing the MTK - GMG integration")
println("===============================================")

# These are the final simulations for the ZASSy paper, but done @ a lower resolution
Num         = NumParam( #Nx=269*1, Nz=269*1,
                        Nx=65*1, Nz=65*1,
                        SimName="ZASSy_Geneva_9_1e_6", axisymmetric=false,
                        #maxTime_Myrs=1.5,
                        maxTime_Myrs=0.025,
                        fac_dt=0.2, ω=0.5, verbose=false,
                        flux_bottom_BC=false, flux_bottom=0, deactivate_La_at_depth=false,
                        Geotherm=30/1e3, TrackTracersOnGrid=true,
                        SaveOutput_steps=100000, CreateFig_steps=100000, plot_tracers=false, advect_polygon=false,
                        FigTitle="Geneva Models, Geotherm 30/km",
                        USE_GPU=USE_GPU);

Sill_params = SillParams(
            sill=CylindricalDikeTopAccretion(Center=Point2(0.0, -7.0e3) * m, W=20e3 * m, H=74.6269 * m),
            InjectionInterval_year = 5000,       # flux= 14.9e-6 km3/km2/yr
            nTr_dike=300*1
        )

MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                        #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                            Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                            #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                            HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting
                                # Melting = MeltingParam_Caricchi()),                     # Caricchi melting
                # add more parameters here, in case you have >1 phase in the model
                )

# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GMG_2D.MTK_GeoParams_2D(MatParam, Num, Sill_params); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 296.4607300089425  rtol= 1e-4
@test sum(time_props.MeltFraction)  ≈ 0.0  rtol= 1e-5



# -----------------------------


Topo_cart = load_GMG(normpath(joinpath(@__DIR__, "..", "examples", "Topo_cart")))       # Note: Laacher seee is around [10,20]

# Create 3D grid of the region
X,Y,Z       =   xyz_grid(-23:.1:23,-19:.1:19,-20:.1:5)
Data_set3D  =   CartData(X,Y,Z,(Phases=zeros(Int64,size(X)),Temp=zeros(size(X))));       # 3D dataset

# Create 2D cross-section
Nx          =   Num.Nx;  # resolution in x
Nz          =   Num.Nz;
Data_2D     =   cross_section(Data_set3D, Start=(-20,4), End=(20,4), dims=(Nx, Nz))
Data_2D     =   addfield(Data_2D,"FlatCrossSection", flatten_cross_section(Data_2D))
Data_2D     =   addfield(Data_2D,"Phases", Int64.(Data_2D.fields.Phases))

# Intersect with topography
Below = below_surface(Data_2D, Topo_cart)
Data_2D.fields.Phases[Below] .= 1

# Set Moho
ind = findall(Data_2D.z.val .< -30.0)
Data_2D.fields.Phases[ind] .= 2

# Set T:
gradient = 30
Data_2D.fields.Temp .= -Data_2D.z.val*gradient
ind = findall(Data_2D.fields.Temp .< 10.0)
Data_2D.fields.Temp[ind] .= 10.0

# Set thermal anomaly
x_c, z_c, r = -10, -15, 2.5
Volume  = 4/3*pi*r^3 # equivalent 3D volume of the anomaly [km^3]
ind = findall((Data_2D.x.val .- x_c).^2 .+ (Data_2D.z.val .- z_c).^2 .< r^2)
Data_2D.fields.Temp[ind] .= 800.0

"""
Randomly change orientation and location of a dike
"""
function MTK_GMG.MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::SillParameters, Num::NumericalParameters)
    if mod(Num.it,10)==0
        cen       =     (Grid.max .+ Grid.min)./2 .+ 0*rand(rng, -0.5:1e-3:0.5, 2).*[Dikes.W_ran; Dikes.H_ran];    # Randomly vary center of dike
        if cen[end]<-15e3;  Angle_rand = 0*rand(rng, 80.0:0.1:100.0)                                              # Orientation: near-vertical @ depth
        else                Angle_rand = 0*rand(rng,-10.0:0.1:10.0); end

        if hasproperty(Dikes, :sill) && !isnothing(Dikes.sill)
            Dikes.sill = InjectSills.update_abstractsill(Dikes.sill;
                                                         Center=Point2(cen[1], cen[2]) * m,
                                                         Angle=Vec1(Angle_rand) * NoUnits)
        else
            Dikes.Center = cen;
            Dikes.Angle = [Angle_rand];
        end
    end
    return nothing
end

# Define numerical parameters
Num         = NumParam( SimName="Unzen1", axisymmetric=false,
                        maxTime_Myrs=0.005,
                        fac_dt=0.2, ω=0.5, verbose=false,
                        SaveOutput_steps=10000, CreateFig_steps=1000, plot_tracers=false, advect_polygon=false,
                        USE_GPU=USE_GPU);

# dike parameters
Sill_params = SillParams(
            sill=EllipticalIntrusion(Center=Point2(0.0, -7.0e3) * m, W=5e3 * m, H=250 * m),
            InjectionInterval_year = 1000,       # flux= 14.9e-6 km3/km2/yr
            nTr_dike=300*1,
            H_ran = 5000, W_ran = 5000,
            SillPhase=3, BackgroundPhase=1,
        )

# Define parameters for the different phases
MatParam     = (SetMaterialParams(Name="Air", Phase=0,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                                Conductivity = ConstantConductivity(k=3Watt/K/m),          # in case we use constant k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),          # Marxer & Ulmer melting

                SetMaterialParams(Name="Crust", Phase=1,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting

                SetMaterialParams(Name="Mantle", Phase=2,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K)),

                SetMaterialParams(Name="Dikes", Phase=3,
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                        #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder()))      # Marxer & Ulmer melting

                )


# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GMG_2D.MTK_GeoParams_2D(MatParam, Num, Sill_params, CartData_input=Data_2D); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 251.58206620240594  rtol= 1e-4
@test sum(time_props.MeltFraction)  ≈  0.2238128607809668 rtol= 1e-5
# Regression: the injected dike's phase must actually land in the phase array.
# (PhasesFromTracers! was being called on a throwaway copy — Array(Arrays.Phases) —
#  so SillPhase=3 never reached Arrays.Phases.)
@test any(Array(Arrays.Phases) .== Sill_params.SillPhase)

# -----------------------------
# Eruption time-loop integration (issue 2): erupt_magma! wired into the loop
# via the MTK_erupt! callback. A short, self-contained run that injects hot
# (T_in=1000°C ⇒ ϕ=1) magma and erupts it once the eruptible volume exceeds a
# (deliberately tiny) critical volume.
# -----------------------------
@testset "Eruption time-loop" begin

    Num_e = NumParam(Nx=65, Nz=65, W=10e3, H=10e3,
                     SimName="Erupt2D", axisymmetric=false,
                     maxTime_Myrs=0.001, fac_dt=0.2, ω=0.5, verbose=false,
                     flux_bottom_BC=false, Geotherm=30/1e3, TrackTracersOnGrid=true,
                     SaveOutput_steps=100000, CreateFig_steps=100000,
                     plot_tracers=false, advect_polygon=false, USE_GPU=USE_GPU)

    # thick sill (H=600 m ≫ dz≈156 m) so grid nodes are actually set to T_in
    Sill_e = SillParams(
                sill=EllipticalIntrusion(Center=Point2(5.0e3, -5.0e3) * m, W=2e3 * m, H=600.0 * m),
                InjectionInterval_year=100, nTr_dike=100)

    Mat_e  = (SetMaterialParams(Name="Rock & partial melt", Phase=1,
                                Density      = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat   = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting      = SmoothMelting(MeltingParam_4thOrder())),)

    # --- eruptions ENABLED: thermal extraction only (deflate=false) ---
    Erupt_on = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit_km3=1e-6,
                              erupt_efficiency=0.5, deflate=false)
    _, Arrays_e, _, _, _ = MTK_GMG_2D.MTK_GeoParams_2D(Mat_e, Num_e, Sill_e; Erupt=Erupt_on)

    @test Erupt_on.n_eruptions ≥ 1                                   # eruptions actually fired in the run
    @test Erupt_on.erupted_volume > 0
    @test length(Erupt_on.eruption_times)   == Erupt_on.n_eruptions  # bookkeeping consistent
    @test length(Erupt_on.eruption_volumes) == Erupt_on.n_eruptions
    @test all(isfinite, Arrays_e.Tnew)                               # state stays finite

    # --- eruptions DISABLED (default): the callback is a no-op ---
    Num_off = NumParam(Nx=65, Nz=65, W=10e3, H=10e3,
                       SimName="Erupt2D_off", axisymmetric=false,
                       maxTime_Myrs=0.001, fac_dt=0.2, ω=0.5, verbose=false,
                       flux_bottom_BC=false, Geotherm=30/1e3, TrackTracersOnGrid=true,
                       SaveOutput_steps=100000, CreateFig_steps=100000,
                       plot_tracers=false, advect_polygon=false, USE_GPU=USE_GPU)
    Sill_off = SillParams(
                sill=EllipticalIntrusion(Center=Point2(5.0e3, -5.0e3) * m, W=2e3 * m, H=600.0 * m),
                InjectionInterval_year=100, nTr_dike=100)
    Erupt_off = EruptionParams()                                     # erupt=false by default
    MTK_GMG_2D.MTK_GeoParams_2D(Mat_e, Num_off, Sill_off; Erupt=Erupt_off)
    @test Erupt_off.erupt == false
    @test Erupt_off.n_eruptions == 0 && Erupt_off.erupted_volume == 0.0

    rm("Erupt2D", recursive=true, force=true)
    rm("Erupt2D_off", recursive=true, force=true)
end

# -----------------------------
# Free-surface time-loop integration (issue 4): the sticky-air surface is wired
# into the loop via MTK_free_surface! (injection inflation + air stamping) and
# MTK_erupt! (eruption deflation). The surface should be allocated, move from its
# initial flat elevation, stay finite, and stamp air (Phase 0) above topography.
# -----------------------------
@testset "Free-surface time-loop" begin

    # two phases: Air (0) + Crust (1), so air cells have valid material props
    Mat_fs = (SetMaterialParams(Name="Air", Phase=0,
                                Density      = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat   = ConstantLatentHeat(Q_L=0.0J/kg),
                                Conductivity = ConstantConductivity(k=3Watt/K/m),
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting      = SmoothMelting(MeltingParam_4thOrder())),
              SetMaterialParams(Name="Crust", Phase=1,
                                Density      = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat   = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting      = SmoothMelting(MeltingParam_4thOrder())))

    Num_fs = NumParam(Nx=65, Nz=65, W=10e3, H=10e3,
                      SimName="FreeSurf2D", axisymmetric=false,
                      maxTime_Myrs=0.001, fac_dt=0.2, ω=0.5, verbose=false,
                      flux_bottom_BC=false, Geotherm=30/1e3, TrackTracersOnGrid=true,
                      SaveOutput_steps=100000, CreateFig_steps=100000,
                      plot_tracers=false, advect_polygon=false, USE_GPU=USE_GPU)
    Sill_fs = SillParams(
                sill=EllipticalIntrusion(Center=Point2(5.0e3, -5.0e3) * m, W=2e3 * m, H=600.0 * m),
                InjectionInterval_year=100, nTr_dike=100, SillPhase=1, BackgroundPhase=1)

    # eruptions (deflate=true) so the surface subsides as well as inflates
    Erupt_fs = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit_km3=1e-6,
                              erupt_efficiency=0.5, deflate=true)
    FS_on = FreeSurfaceParams(free_surface=true, air_phase=0, Tair=0.0, z0=-2000.0)

    _, Arrays_fs, _, _, _ = MTK_GMG_2D.MTK_GeoParams_2D(Mat_fs, Num_fs, Sill_fs;
                                                        Erupt=Erupt_fs, FS=FS_on)

    @test FS_on.z_surf !== nothing                       # surface allocated
    @test length(FS_on.z_surf) == Num_fs.Nx
    @test all(isfinite, FS_on.z_surf)
    @test !all(FS_on.z_surf .≈ FS_on.z0)                 # surface moved (inflation/deflation)
    @test all(FS_on.z_surf .>= -Num_fs.H - 1e-6) && all(FS_on.z_surf .<= 1e-6)   # within domain
    @test any(Array(Arrays_fs.Phases) .== 0)             # air was stamped above topography
    @test all(isfinite, Arrays_fs.Tnew)

    # --- free surface DISABLED (default): callback is a no-op, no surface ---
    Num_off = NumParam(Nx=65, Nz=65, W=10e3, H=10e3,
                       SimName="FreeSurf2D_off", axisymmetric=false,
                       maxTime_Myrs=0.001, fac_dt=0.2, ω=0.5, verbose=false,
                       flux_bottom_BC=false, Geotherm=30/1e3, TrackTracersOnGrid=true,
                       SaveOutput_steps=100000, CreateFig_steps=100000,
                       plot_tracers=false, advect_polygon=false, USE_GPU=USE_GPU)
    Sill_off2 = SillParams(
                sill=EllipticalIntrusion(Center=Point2(5.0e3, -5.0e3) * m, W=2e3 * m, H=600.0 * m),
                InjectionInterval_year=100, nTr_dike=100, SillPhase=1, BackgroundPhase=1)
    FS_off = FreeSurfaceParams()                          # free_surface=false by default
    MTK_GMG_2D.MTK_GeoParams_2D(Mat_fs, Num_off, Sill_off2; FS=FS_off)
    @test FS_off.free_surface == false
    @test FS_off.z_surf === nothing                       # never allocated when disabled

    rm("FreeSurf2D", recursive=true, force=true)
    rm("FreeSurf2D_off", recursive=true, force=true)
end

# remove directory created by this test
rm("ZASSy_Geneva_9_1e_6", recursive=true, force=true)
rm("Unzen1", recursive=true, force=true)

end

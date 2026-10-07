using Test, Random
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end
using InjectSills

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 3)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
else
    environment!(:cpu, Float64, 3)      # initialize parallel stencil in 2D
end
import MagmaThermoKinematics.Diffusion3D

# Allow overwriting user routines
import MagmaThermoKinematics.MTK_GMG
import MagmaThermoKinematics.MTK_GMG_3D

using Random, GeoParams, GeophysicalModelGenerator

const rng = Random.seed!(1234);     # same seed such that we can reproduce results

function _build_injectsill_3d(Dikes)
    center = Point3(Dikes.Center[1], Dikes.Center[2], Dikes.Center[3]) * m
    angle = Vec2(Dikes.Angle[1], Dikes.Angle[end]) * NoUnits

    if Dikes.Type == "CylindricalDike_TopAccretion"
        return CylindricalDikeTopAccretion(Center=center, Angle=angle, W=Dikes.W_in * m, H=Dikes.H_in * m)
    elseif Dikes.Type == "EllipticalIntrusion" || Dikes.Type == "ElasticDike"
        return EllipticalIntrusion(Center=center, Angle=angle, W=Dikes.W_in * m, H=Dikes.H_in * m)
    elseif Dikes.Type == "InjectSills"
        isnothing(Dikes.sill) && error("Dikes.Type='InjectSills' requires Dikes.sill to be set")
        return Dikes.sill
    else
        error("Unsupported Dikes.Type for InjectSills callback in test_MTK_GMG_3D: $(Dikes.Type)")
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
            sill = IS.CylindricalDikeTopAccretion(Center=IS.Point3(Dikes.Center[1], Dikes.Center[2], Dikes.Center[3]) * m_unit,
                                                 Angle=IS.Vec2(Dikes.Angle[1], Dikes.Angle[end]) * no_unit,
                                                 W=Dikes.W_in * m_unit,
                                                 H=Dikes.H_in * m_unit)
        elseif Dikes.Type == "EllipticalIntrusion" || Dikes.Type == "ElasticDike"
            sill = IS.EllipticalIntrusion(Center=IS.Point3(Dikes.Center[1], Dikes.Center[2], Dikes.Center[3]) * m_unit,
                                         Angle=IS.Vec2(Dikes.Angle[1], Dikes.Angle[end]) * no_unit,
                                         W=Dikes.W_in * m_unit,
                                         H=Dikes.H_in * m_unit)
        else
            error("Unsupported Dikes.Type for InjectSills callback in test_MTK_GMG_3D: $(Dikes.Type)")
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
            PhasesFromTracers!(Array(Arrays.Phases), Grid, Tracers, BackgroundPhase=Dikes.BackgroundPhase, InterpolationMethod="Constant")

            if Num.keep_init_RockPhases == true
                Phases = Array(Arrays.Phases)
                Phases_init = Array(Arrays.Phases_init)
                for i in eachindex(Phases)
                    if Phases[i] != intrusion_phase
                        Phases[i] = Phases_init[i]
                    end
                end
                Arrays.Phases .= DataArray(Phases)
            end
        end
    end

    return Tracers
end
end


@testset "MTK_GMG_3D" begin

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    if mod(Num.it,10) == 0
        println("$(Num.it), $(Num.time/SecYear/1e3) kyrs; max(T)=$(maximum(Arrays.Tnew))")
    end
    return nothing
end

# Test setup
println("===============================================")
println("Testing MTK - GMG integration in 3D")
println("===============================================")

# Perform simulations @ a lower resolution to speed up GitHub CI tests (on limited memory machines)
Num         = NumParam( #Nx=269*1, Nz=269*1,
                        Nx=31*1, Ny=31*1, Nz=31*1,
                        SimName="Test1",
                        W=20e3, H=20e3, L=20e3,
                        #maxTime_Myrs=1.5,
                        maxTime_Myrs=0.001,
                        fac_dt=0.2, ω=0.5, verbose=false,
                        flux_bottom_BC=false, flux_bottom=0, deactivate_La_at_depth=false,
                        Geotherm=30/1e3, TrackTracersOnGrid=true,
                        SaveOutput_steps=10, CreateFig_steps=100000, plot_tracers=false, advect_polygon=true,
                        FigTitle="Geneva Models, Geotherm 30/km",
                        USE_GPU=USE_GPU,
                        AddRandomSills = false, RandomSills_timestep=5
                        );

Sill_params = SillParams(
            sill=EllipticalIntrusion(Center=Point3(0.0, 0.0, -7000.0) * m, Angle=Vec2(0.0, 0.0) * NoUnits, W=5e3 * m, H=200.0*4 * m),
            InjectionInterval_year = 1000,
            Dip_ran = 20.0, Strike_ran = 0.0,
            W_ran = 10e3, H_ran = 10e3, L_ran=10e3,
            nTr_dike=300*1,
            SillsAbove = -10e3,
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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GMG_3D.MTK_GeoParams_3D(MatParam, Num, Sill_params); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 299.981239425671  rtol= 1e-2
@test sum(time_props.MeltFraction)  ≈ 0.0  rtol= 1e-5
# -----------------------------


Topo_cart = load_GMG(normpath(joinpath(@__DIR__, "..", "examples", "Topo_cart")))       # Note: Laacher seee is around [10,20]

# Create 3D grid of the region
Nx,Ny,Nz = 100,100,100
X,Y,Z       =   xyz_grid(range(-23,23, length=Nx),range(-19,19, length=Ny),range(-20,5, length=Nz))
Data_3D     =   CartData(X,Y,Z,(Phases=zeros(Int64,size(X)),Temp=zeros(size(X))));       # 3D dataset

# Intersect with topography
Below = below_surface(Data_3D, Topo_cart)
Data_3D.fields.Phases[Below] .= 1

# Set Moho
ind = findall(Data_3D.z.val .< -30.0)
Data_3D.fields.Phases[ind] .= 2

# Set T:
gradient = 30
Data_3D.fields.Temp .= -Data_3D.z.val*gradient
ind = findall(Data_3D.fields.Temp .< 10.0)
Data_3D.fields.Temp[ind] .= 10.0

# Set thermal anomaly
x_c, y_c, z_c, r = -10, -10, -15, 2.5
Volume  = 4/3*pi*r^3 # equivalent 3D volume of the anomaly [km^3]
ind = findall((Data_3D.x.val .- x_c).^2 .+ (Data_3D.y.val .- y_c).^2 .+ (Data_3D.z.val .- z_c).^2 .< r^2)
Data_3D.fields.Temp[ind] .= 800.0


# Define numerical parameters
Num         = NumParam( SimName="Unzen2", axisymmetric=false,
                        maxTime_Myrs=0.001,
                        fac_dt=0.2,
                        SaveOutput_steps=20, CreateFig_steps=1000, plot_tracers=false, advect_polygon=false,
                        USE_GPU=USE_GPU,
                        AddRandomSills = false, RandomSills_timestep=5);

# dike parameters
Sill_params = SillParams(
            sill=EllipticalIntrusion(Center=Point3(0.0, 0.0, -7000.0) * m, Angle=Vec2(0.0, 0.0) * NoUnits, W=5e3 * m, H=250*4 * m),
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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GMG_3D.MTK_GeoParams_3D(MatParam, Num, Sill_params, CartData_input=Data_3D); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 244.14916470514495  rtol= 1e-2
@test sum(time_props.MeltFraction)  ≈ 0.8377621121586017 rtol= 1e-5

rm("Test1", recursive=true, force=true) # remove directory created by this test
rm("Unzen2", recursive=true, force=true) # remove directory created by this test

end

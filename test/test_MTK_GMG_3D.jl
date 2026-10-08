using Test, Random
using InjectSills

using MagmaThermoKinematics

# Allow overwriting user routines
import MagmaThermoKinematics.MTK_GMG

using Random, GeoParams, GeophysicalModelGenerator

const rng = Random.seed!(1234);     # same seed such that we can reproduce results

@eval MTK_GMG begin
function MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructVector)
    if floor(Num.time / Dikes.InjectionInterval) > Dikes.sill_inj
        Dikes.sill_inj = floor(Num.time / Dikes.InjectionInterval)

        T_bottom = copy(selectdim(Arrays.T, Num.dim, 1))

        sill = _active_sill(Dikes)
        if Num.advect_polygon == true && isempty(Dikes.sill_poly)
            Dikes.sill_poly = InjectSills.dike_polygon(sill)
        end

        Tracers, _, Vol, _, _ = inject_sills(Tracers, Arrays.T, Grid.coord1D, sill, Dikes.T_in_Celsius, Dikes.SillPhase, Dikes.nTr_dike)

        if Num.flux_bottom_BC == false
            selectdim(Arrays.T, Num.dim, 1) .= T_bottom
        end

        Dikes.InjectVol += Vol
        Qrate = Dikes.InjectVol / Num.time
        Dikes.Qrate_km3_yr = Qrate * SecYear / km³
        println("  Added new dike; time=$(Num.time / kyr) kyrs, total injected magma volume = $(Dikes.InjectVol / km³) km³; rate Q= $(Dikes.Qrate_km3_yr) km³yr⁻¹")

        if length(Mat_tup) > 1
            Phases = Array(Arrays.Phases)
            PhasesFromTracers!(Phases, Grid, Tracers, BackgroundPhase=Dikes.BackgroundPhase, InterpolationMethod="Constant")

            if Num.keep_init_RockPhases == true
                Phases_init = Array(Arrays.Phases_init)
                for i in eachindex(Phases, Phases_init)
                    if Phases[i] != Dikes.SillPhase
                        Phases[i] = Phases_init[i]
                    end
                end
            end
            copyto!(Arrays.Phases, Phases)
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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params); # start the main code

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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params, CartData_input=Data_3D); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 244.14916470514495  rtol= 1e-2
@test sum(time_props.MeltFraction)  ≈ 0.00837762112158602 rtol= 1e-5

rm("Test1", recursive=true, force=true) # remove directory created by this test
rm("Unzen2", recursive=true, force=true) # remove directory created by this test

end

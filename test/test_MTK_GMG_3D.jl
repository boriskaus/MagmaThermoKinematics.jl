using Test, Random
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end
using ParallelStencil, ParallelStencil.FiniteDifferences3D

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 3)      # initialize parallel stencil in 2D
    CUDA.device!(1)                     # select the GPU you use (starts @ zero)
    @init_parallel_stencil(CUDA, Float64, 3)
else
    environment!(:cpu, Float64, 3)      # initialize parallel stencil in 2D
    @init_parallel_stencil(Threads, Float64, 3)
end
using MagmaThermoKinematics.Diffusion3D
using Random, GeoParams, GeophysicalModelGenerator

const rng = Random.seed!(1234);     # same seed such that we can reproduce results

# Allow overwriting user routines
using MagmaThermoKinematics.MTK_GMG

@testset "MTK_GMG_3D" begin

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
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

Dike_params = DikeParam(Type="ElasticDike",
                        InjectionInterval_year = 1000,
                        W_in=5e3, H_in=200.0*4,       # note: H must be numerically resolved
                        Dip_ran = 20.0, Strike_ran = 0.0,
                        W_ran = 10e3; H_ran = 10e3, L_ran=10e3,
                        nTr_dike=300*1,
                        SillsAbove = -10e3,
                        Center=[0.0,0.0, -7000], Angle=[0.0, 0.0],
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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_3D(MatParam, Num, Dike_params); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 299.981239425671  rtol= 1e-2
@test sum(time_props.MeltFraction)  ≈ 0.0  rtol= 1e-5
# -----------------------------


Topo_cart = load_GMG("../examples/Topo_cart")       # Note: Laacher seee is around [10,20]

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
Dike_params = DikeParam(Type="ElasticDike",
                        InjectionInterval_year = 1000,       # flux= 14.9e-6 km3/km2/yr
                        W_in=5e3, H_in=250*4,
                        nTr_dike=300*1,
                        H_ran = 5000, W_ran = 5000,
                        DikePhase=3, BackgroundPhase=1,
                        Center=[0.0,0.0, -7000], Angle=[0.0, 0.0],
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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_3D(MatParam, Num, Dike_params, CartData_input=Data_3D); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 244.14916470514495  rtol= 1e-2
@test sum(time_props.MeltFraction)  ≈ 0.8377621121586017 rtol= 1e-5


end

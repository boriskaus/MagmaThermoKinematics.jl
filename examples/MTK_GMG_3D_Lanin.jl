using  Random
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 3)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
else
    environment!(:cpu, Float64, 3)      # initialize parallel stencil in 2D
end
using MagmaThermoKinematics.Diffusion3D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields3D
using MagmaThermoKinematics.MTK_GMG
using MagmaThermoKinematics.MTK_GMG_3D
using Random, GeoParams, GeophysicalModelGenerator

const rng = Random.seed!(1234);     # same seed such that we can reproduce results


@static if USE_GPU
    function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
        if mod(Num.it,10) == 0
            println("$(Num.it), $(Num.time/SecYear/1e3) kyrs; max(T)=$(maximum(Arrays.Tnew))")
        end
        return nothing
    end
end
# Test setup
println("===============================================")
println("            Performing MTK model               ")
println("===============================================")
# -----------------------------


# Create 3D grid of the region
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
!isfile(joinpath(@__DIR__,"Topo_cart_Lanin3D.vts")) ? write_paraview(Topo_cart,joinpath(@__DIR__,"Topo_cart_Lanin3D")) : nothing
x_range     =   (-20,20)
z_range     =   (-40,5)
Nx          =   128
Ny          =   128
Nz          =   128
X,Y,Z       =   xyz_grid(range(x_range[1],x_range[2], length=Nx),range(x_range[1],x_range[2], length=Ny),range(z_range[1],z_range[2], length=Nz))
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

!isfile(joinpath(@__DIR__,"Initial_Setup_Lanin3D.vts")) ? write_paraview(Data_3D, joinpath(@__DIR__,"Initial_Setup_Lanin3D")) : nothing


# Define numerical parameters
Num         = NumParam( SimName="Lanin3D_$(Nx)^3", axisymmetric=false,
                        maxTime_Myrs=0.025,
                        Nx = Nx, Ny = Ny, Nz = Nz,
                        fac_dt=0.2,
                        SaveOutput_steps=20, CreateFig_steps=1000, plot_tracers=false, advect_polygon=false,
                        USE_GPU=USE_GPU,
                        AddRandomSills = true, RandomSills_timestep=5);

# dike parameters
Dike_params = DikeParam(Type="ElasticDike",
                        InjectionInterval_year = 500,       # flux= 14.9e-6 km3/km2/yr
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
Grid, Arrays, Tracers, Dikes, time_props = MTK_GMG_3D.MTK_GeoParams_3D(MatParam, Num, Dike_params, CartData_input=Data_3D); # start the main code

Data_set3D_out = Data_3D;
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "Temperature[C]",  Array(Arrays.Tnew ));   # in MPa
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "Temp",         Array(Arrays.Tnew));
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "Phases",       Array(Arrays.Phases));
Data_set3D_out = MTK_GMG.add_data_CartData(Data_set3D_out, "MeltFraction", Array(Arrays.ϕ));
save_GMG(joinpath(Num.SimName,"Lanin3D_MTK_final"), Data_set3D_out)
write_paraview(Data_3D, joinpath(Num.SimName,"Lanin3D_MTK_final"))

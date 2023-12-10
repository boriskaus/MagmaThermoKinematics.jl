# This is a second example that shows how to use MTK in combination with the GeophysicalModelGenerator
# Compared to example 1, we show a few more features:
# - How to define a custom structure with temporal values and how to use it in the code
# - How to generate a model setup using GMG

const USE_GPU=true;
using MagmaThermoKinematics
using CUDA
if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    #CUDA.device!(1)                     # select the GPU you use (starts @ zero)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
end
using MagmaThermoKinematics.Diffusion2D

using Plots
using Random

# Import a few routines, so we can overwrite them below
import MagmaThermoKinematics.MTK_GMG_2D.MTK_visualize_output
import MagmaThermoKinematics.MTK_GMG_2D.MTK_print_output
import MagmaThermoKinematics.MTK_GMG_2D.MTK_update_TimeDepProps!
import MagmaThermoKinematics.MTK_GMG_2D.MTK_update_ArraysStructs!
import MagmaThermoKinematics.MTK_GMG_2D.MTK_initialize!
import MagmaThermoKinematics.MTK_GMG_2D.MTK_updateTracers
import MagmaThermoKinematics.MTK_GMG_2D.MTK_save_output

# Test setup
println("Example 2 of the MTK - GMG integration")

println(" --- Generating Setup --- ")
using  GeophysicalModelGenerator

# Topography and project it. 

# NOTE: The first time you do this, please uncomment the following lines, 
# which will download the topography data from the internet and save it in a file
#=
 using GMT, Statistics
 Topo = ImportTopo(lon = [130.0, 130.5], lat=[32.55, 32.90], file="@earth_relief_03s.grd")
 proj =  ProjectionPoint(; Lat=mean(Topo.lat.val), Lon=mean(Topo.lon.val))
 Topo_cart = Convert2CartData(Topo, proj)
 save_GMG("examples/Topo_cart", Topo_cart)
=#
Topo_cart = load_GMG("examples/Topo_cart")       # Note: Laacher seee is around [10,20]

# Create 3D grid of the region
X,Y,Z       =   XYZGrid(-23:.1:23,-19:.1:19,-20:.1:5)
Data_set3D  =   CartData(X,Y,Z,(Phases=zeros(Int64,size(X)),Temp=zeros(size(X))));       # 3D dataset

# Create 2D cross-section
Nx          =   135*2;  # resolution in x
Nz          =   135*2;
Data_2D     =   CrossSection(Data_set3D, Start=(-20,4), End=(20,4), dims=(Nx, Nz))
Data_2D     =   AddField(Data_2D,"FlatCrossSection", FlattenCrossSection(Data_2D))
Data_2D     =   AddField(Data_2D,"Phases", Int64.(Data_2D.fields.Phases))

# Intersect with topography
Below = BelowSurface(Data_2D, Topo_cart)
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


println(" --- Performing MTK models --- ")

# Overwrite some of the default functions
if USE_GPU
    function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
        println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
        return nothing
    end
else
    function MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
        if mod(Num.it,Num.CreateFig_steps)==0
            x_1d =  Grid.coord1D[1]/1e3;
            z_1d =  Grid.coord1D[2]/1e3;
            temp_data = Array(Arrays.Tnew)'
            ϕ_data = Array(Arrays.ϕ)'
            phase_data = Float64.(Array(Arrays.Phases))'
            
            # remove topo on plots
            ind = findall(phase_data .== 0)
            phase_data[ind] .= NaN
            temp_data[ind] .= NaN
            
            t = Num.time/SecYear/1e3;

            p=plot(layout=grid(1,2) )

            Plots.heatmap!(p[1],x_1d, z_1d, temp_data, c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Temperature, t=$(round(t)) kyrs", aspect_ratio=:equal,  ylimits=(minimum(z_1d),2))
            Plots.heatmap!(p[2],x_1d, z_1d, ϕ_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction", clims=(0,1), aspect_ratio=:equal, ylimits=(minimum(z_1d),2))
            #Plots.heatmap!(p[2],x_1d, z_1d, phase_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction", aspect_ratio=:equal, ylimits=(minimum(z_1d),2))

        # p = plot(ps, layout=(1,2))
            display(p)
        end
        return nothing
    end
end

function MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    return nothing
end

function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    println("$(Num.it), Time=$(round(Num.time/Num.SecYear/1e3, digits=3)) kyrs; max(T) = $(round(maximum(Arrays.Tnew)))")
    return nothing
end

"""
Randomly change orientation and location of a dike
"""
function MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters)
    if mod(Num.it,10)==0
        cen       =     (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, 2).*[Dikes.W_ran; Dikes.H_ran];    # Randomly vary center of dike 
        if cen[end]<-15e3;  Angle_rand = rand( 80.0:0.1:100.0)                                              # Orientation: near-vertical @ depth             
        else                Angle_rand = rand(-10.0:0.1:10.0); end                        
        
        Dikes.Center = cen; 
        Dikes.Angle = [Angle_rand];
    end
    return nothing
end

"""
    MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

Update time-dependent properties during a simulation
"""
function MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    push!(time_props.Time_vec,      Num.time);   # time 
    push!(time_props.MeltFraction,  sum( Arrays.ϕ)/(Num.Nx*Num.Nz));    # melt fraction       

    ind = findall(Arrays.T.>700);          
    if ~isempty(ind)
        Tav_magma_Time = sum(Arrays.T[ind])/length(ind)     # average T of part with magma
    else
        Tav_magma_Time = NaN;
    end
    push!(time_props.Tav_magma, Tav_magma_Time);       # average magma T
    push!(time_props.Tmax,      maximum(Arrays.T));   # maximum magma T
    return nothing
end

# Define a new structure with time-dependent properties
@with_kw mutable struct TimeDepProps1 <: TimeDependentProperties
    Time_vec::Vector{Float64}  = [];        # Center of dike 
    MeltFraction::Vector{Float64} = [];     # Melt fraction over time
    Tav_magma::Vector{Float64} = [];        # Average magma 
    Tmax::Vector{Float64} = [];             # Max magma temperature
    Tmax_1::Vector{Float64} = [];           # Another magma temperature vector
end

# Define numerical parameters
Num         = NumParam( SimName="Unzen1", axisymmetric=false,
                        maxTime_Myrs=0.005, 
                        fac_dt=0.2, ω=0.5, verbose=false, 
                        SaveOutput_steps=25, CreateFig_steps=5, plot_tracers=false, advect_polygon=false,
                        USE_GPU=USE_GPU);

# dike parameters
Dike_params = DikeParam(Type="ElasticDike", 
                        InjectionInterval_year = 1000,       # flux= 14.9e-6 km3/km2/yr
                        W_in=5e3, H_in=250,
                        nTr_dike=300*1,
                        H_ran = 5000, W_ran = 5000,
                        DikePhase=3, BackgroundPhase=1,
                )

# Define parameters for the different phases 
MatParam     = (SetMaterialParams(Name="Air", Phase=0, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                                Conductivity = ConstantConductivity(k=3Watt/K/m),          # in case we use constant k
                                HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),          # Marxer & Ulmer melting     

                SetMaterialParams(Name="Crust", Phase=1, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting 

                SetMaterialParams(Name="Mantle", Phase=2, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K)),

                SetMaterialParams(Name="Dikes", Phase=3, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                        #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder()))      # Marxer & Ulmer melting     
                                  
                )


# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Dike_params, CartData_input=Data_2D, time_props=TimeDepProps1()); # start the main code


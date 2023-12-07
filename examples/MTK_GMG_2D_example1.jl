# This is a first example of how to use MagmaThermoKinematics with a real setup which can be customized with user-defined functions,
# for example for plotting or printing output.

const USE_GPU=false;
using MagmaThermoKinematics
if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(1)                     # select the GPU you use (starts @ zero)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
end
using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics

using Plots
using Random, GeoParams

# Import a few routines, so we can overwrite them below
import MagmaThermoKinematics.MTK_GMG_2D.MTK_visualize_output
import MagmaThermoKinematics.MTK_GMG_2D.MTK_print_output
import MagmaThermoKinematics.MTK_GMG_2D.MTK_update_TimeDepProps!
import MagmaThermoKinematics.MTK_GMG_2D.MTK_update_ArraysStructs!
import MagmaThermoKinematics.MTK_GMG_2D.MTK_initialize!
import MagmaThermoKinematics.MTK_GMG_2D.MTK_updateTracers
import MagmaThermoKinematics.MTK_GMG_2D.MTK_save_output


Random.seed!(1234);     # such that we can reproduce results

# Test setup
println("Example 1 of the MTK - GMG integration")

# Overwrite some functions
function MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    if mod(Num.it,Num.CreateFig_steps)==0
        x_1d =  Grid.coord1D[1]/1e3;
        z_1d =  Grid.coord1D[2]/1e3;
        temp_data = Array(Arrays.Tnew)'
        ϕ_data = Array(Arrays.ϕ)'
        phase_data = Array(Arrays.Phases)'
        phase_data = Array(Arrays.Phases_init)'
        
        t = Num.time/SecYear;

        p=plot(layout=grid(1,2) )

        Plots.heatmap!(p[1],x_1d, z_1d, temp_data, c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Temperature, t=$(round(t)) yrs", aspect_ratio=:equal, ylimits=(-20,0))
#        Plots.heatmap!(p[2],x_1d, z_1d, ϕ_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction", clims=(0,1), aspect_ratio=:equal, ylimits=(-20,0))
        Plots.heatmap!(p[2],x_1d, z_1d, phase_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction, t=$(round(t)) yrs", aspect_ratio=:equal, ylimits=(-20,0))

       # p = plot(ps, layout=(1,2))
        display(p)
    end
    return nothing
end


function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
    return nothing
end


"""
Randomly change orientation and location of a dike
"""
function MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters)
    if mod(Num.it,10)==0
        cen       =     (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, 2).*[Dikes.W_ran; Dikes.H_ran];    # Randomly vary center of dike 
        if cen[end]<-12e3;  Angle_rand = rand( 80.0:0.1:100.0)                                              # Orientation: near-vertical @ depth             
        else                Angle_rand = rand(-10.0:0.1:10.0); end                        
        
        Dikes.Center = cen; 
        Dikes.Angle = [Angle_rand];
    end
    return nothing
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArrays, Dikes::DikeParameters)

Initialize temperature and phases of the grid
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)
    # Initalize T
    Arrays.T_init      .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm;                # Initial (linear) temperature profile
    
    # Initialize Phases
    ind =  findall(Arrays.Z .> -2000);
    Arrays.Phases[ind] .= 0;    
    
    Arrays.Phases_init .= Arrays.Phases;    
                                                                # Initialize all as rock

    return nothing
end


# Define numerical parameters
Num         = NumParam( Nx=135*1, Nz=135*1, 
                        SimName="Test1", axisymmetric=false,
                        maxTime_Myrs=0.005, 
                        fac_dt=0.2, ω=0.5, verbose=false, 
                        flux_bottom_BC=false, flux_bottom=0, deactivate_La_at_depth=false, 
                        Geotherm=30/1e3, TrackTracersOnGrid=true,
                        SaveOutput_steps=100000, CreateFig_steps=20, plot_tracers=false, advect_polygon=true,
                        FigTitle="Geneva Models, Geotherm 30/km",
                        USE_GPU=USE_GPU);

Dike_params = DikeParam(Type="ElasticDike", 
                        InjectionInterval_year = 1000,       # flux= 14.9e-6 km3/km2/yr
                        W_in=5e3, H_in=250,
                        nTr_dike=300*1,
                        DikePhase=2, BackgroundPhase=1
                )

MatParam     = (SetMaterialParams(Name="Rock 1", Phase=0, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                                #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting     
                SetMaterialParams(Name="Rock 2", Phase=1, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                        #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                            #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                            HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting     
                SetMaterialParams(Name="Dike", Phase=2, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                        #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                            #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                            HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder()))



                                # Melting = MeltingParam_Caricchi()),                     # Caricchi melting
                # add more parameters here, in case you have >1 phase in the model                                    
                )


# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Dike_params); # start the main code


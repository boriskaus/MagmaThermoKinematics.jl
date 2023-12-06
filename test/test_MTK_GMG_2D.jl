#using Test
using Plots

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

using Random, GeoParams

# Import a few routines, so we can overwrite them below
import MagmaThermoKinematics.MTK_GMG_2D.MTK_visualize_output
import MagmaThermoKinematics.MTK_GMG_2D.MTK_print_output
import MagmaThermoKinematics.MTK_GMG_2D.MTK_update_TimeDepProps!

@testset "MTK_GMG_2D" begin

Random.seed!(1234);     # such that we can reproduce results

# Test setup
println("===============================================")
println("Testing the MTK - GMG integration")
println("===============================================")

# Overwrite some functions
#function MTK_visualize_output(Grid, Num, Arrays, Mat_tup, Dikes)
function MTK_visualize_output(Grid, Num::NumericalParameters, Arrays, Mat_tup, Dikes)    
    if mod(Num.it,10)==0
        x_1d =  Grid.coord1D[1]/1e3;
        z_1d =  Grid.coord1D[2]/1e3;
        temp_data = Array(Arrays.Tnew)'
        ϕ_data = Array(Arrays.ϕ)'
        phase_data = Array(Arrays.Phases)'
        t = Num.time/SecYear;


        p=plot(layout=grid(1,2) )

        Plots.heatmap!(p[1],x_1d, z_1d, temp_data, c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Temperature, t=$(round(t)) yrs", aspect_ratio=:equal, ylimits=(-20,0))
#        Plots.heatmap!(p[2],x_1d, z_1d, ϕ_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction, t=$(round(t)) yrs", clims=(0,1), aspect_ratio=:equal, ylimits=(-20,0))
        Plots.heatmap!(p[2],x_1d, z_1d, phase_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction, t=$(round(t)) yrs", aspect_ratio=:equal, ylimits=(-20,0))

       # p = plot(ps, layout=(1,2))
        display(p)
    end
    return nothing
end


function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    @show "Boris", Num.it, maximum(Arrays.Tnew)
    
    return nothing
end


# These are the final simulations for the ZASSy paper, but done @ a lower resolution
Num         = NumParam( #Nx=269*1, Nz=269*1, 
                        Nx=135*1, Nz=135*1, 
                        SimName="ZASSy_Geneva_9_1e_6", axisymmetric=false,
                        #maxTime_Myrs=1.5, 
                        maxTime_Myrs=0.025, 
                        fac_dt=0.2, ω=0.5, verbose=false, 
                        flux_bottom_BC=false, flux_bottom=0, deactivate_La_at_depth=false, 
                        Geotherm=30/1e3, TrackTracersOnGrid=true,
                        SaveOutput_steps=100000, CreateFig_steps=100000, plot_tracers=false, advect_polygon=true,
                        FigTitle="Geneva Models, Geotherm 30/km",
                        USE_GPU=USE_GPU);

Dike_params = DikeParam(Type="CylindricalDike_TopAccretion", 
                        InjectionInterval_year = 5000,       # flux= 14.9e-6 km3/km2/yr
                        W_in=20e3, H_in=74.6269,
                        nTr_dike=300*1
                )

MatParam     = (SetMaterialParams(Name="Rock & partial melt", Phase=1, 
                                Density    = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
                        #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
                            Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                            #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
                            HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
                                Melting = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting     
                                # Melting = MeltingParam_Caricchi()),                     # Caricchi melting
                # add more parameters here, in case you have >1 phase in the model                                    
                )

# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Dike_params); # start the main code

@test sum(Arrays.Tnew)/prod(size(Arrays.Tnew)) ≈ 315.4638294086378  rtol= 1e-2
@test sum(time_props.MeltFraction)  ≈ 0.32112172814171824  rtol= 1e-5



end

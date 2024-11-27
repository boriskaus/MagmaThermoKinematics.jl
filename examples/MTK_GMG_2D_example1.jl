# This is a first example of how to use MagmaThermoKinematics with a real setup which can be customized with user-defined functions,
# for example for plotting or printing output.

const USE_GPU=false;
using MagmaThermoKinematics
if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
end
using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D
using MagmaThermoKinematics.MTK_GMG     # Allow overwriting user routines
using MagmaThermoKinematics.GeophysicalModelGenerator
using Plots                             # plots
using Random, GeoParams

Random.seed!(1234);     # such that we can reproduce results

# Test setup
println("Example 1 of the MTK - GMG integration")

# Overwrite some functions
function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
    return nothing
end

@static if USE_GPU
    function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
        println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
        return nothing
    end
else
    function MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
        if mod(Num.it,Num.CreateFig_steps)==0
            x_1d        =   Grid.coord1D[1]/1e3;
            z_1d        =   Grid.coord1D[2]/1e3;
            temp_data   =   Array(Arrays.Tnew)'
            ϕ_data      =   Array(Arrays.ϕ)'
            phase_data  =   Array(Arrays.Phases)'

            t   =   Num.time/SecYear;
            p   =   plot(layout=grid(1,2) )

            Plots.heatmap!(p[1],x_1d, z_1d, temp_data, c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Temperature, t=$(round(t)) yrs", aspect_ratio=:equal, ylimits=(-20,0))
    #        Plots.heatmap!(p[2],x_1d, z_1d, ϕ_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction", clims=(0,1), aspect_ratio=:equal, ylimits=(-20,0))
            Plots.heatmap!(p[2],x_1d, z_1d, phase_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Phases", aspect_ratio=:equal, ylimits=(-20,0))

           # p = plot(ps, layout=(1,2))
            display(p)
        end
        return nothing
    end
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArrays, Dikes::DikeParameters)

Initialize temperature and phases of the grid
"""
function MTK_GMG.MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)
    # Initalize T
    Arrays.T_init   .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm;                # Initial (linear) temperature profile

    # Initialize Phases
    @views  Arrays.Phases[Arrays.Z .> -5000] .= 0;
    Arrays.Phases_init .= Arrays.Phases;    # Initialize all as rock

     # open pvd file if requested
     if Num.Output_VTK
        name =  joinpath(Num.SimName,Num.SimName*".pvd")
        Num.pvd = movie_paraview(name=name, Initialize=true);
    end

    return nothing
end


# Define numerical parameters
Num         = NumParam( Nx                      =   135*2,
                        Nz                      =   135*2,
                        SimName                 =   "Test1",
                        maxTime_Myrs            =   0.005,
                        fac_dt                  =   0.2,
                        ω                       =   0.5,
                        CreateFig_steps         =   20,
                        SaveOutput_steps        =   100,
                        USE_GPU                 =   USE_GPU,
                        AddRandomSills          =   true,
                        RandomSills_timestep    =   5);

Dike_params = DikeParam(Type                    =   "ElasticDike",
                        InjectionInterval_year  =   1000,
                        W_in                    =   5e3,
                        H_in                    =   250,
                        nTr_dike                =   300*4,
                        DikePhase               =   2,
                        T_in_Celsius            =   1000,
                        SillsAbove              =   -12e3       # below this we have dikes; above sills
                )

MatParam     = (SetMaterialParams(Name="Host rock 1", Phase=0,
                                Density         = ConstantDensity(ρ=2700kg/m^3),                    # used in the parameterisation of Whittington
                                LatentHeat      = ConstantLatentHeat(Q_L=2.55e5J/kg),
                                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                                Conductivity    = T_Conductivity_Whittington(),                       # T-dependent k
                                HeatCapacity    = T_HeatCapacity_Whittington(),                      # T-dependent cp
                                Melting         = MeltingParam_Assimilation()                              # Quadratic parameterization as in Tierney et al.
                                ),
                SetMaterialParams(Name="Host rock", Phase=1,
                                Density         = ConstantDensity(ρ=2700kg/m^3),                    # used in the parameterisation of Whittington
                                LatentHeat      = ConstantLatentHeat(Q_L=2.55e5J/kg),
                                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                                Conductivity    = T_Conductivity_Whittington(),                       # T-dependent k
                                HeatCapacity    = T_HeatCapacity_Whittington(),                      # T-dependent cp
                                Melting         = MeltingParam_Assimilation()                              # Quadratic parameterization as in Tierney et al.
                            ),
                SetMaterialParams(Name="Intruded rocks", Phase=2,
                                Density         = ConstantDensity(ρ=2700kg/m^3),                     # used in the parameterisation of Whittington
                                LatentHeat      = ConstantLatentHeat(Q_L=2.67e5J/kg),
                                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                                Conductivity    = T_Conductivity_Whittington(),                       # T-dependent k
                                HeatCapacity    = T_HeatCapacity_Whittington(),                       # T-dependent cp
                                Melting         = SmoothMelting(MeltingParam_Quadratic(T_s=(700+273.15)K, T_l=(1100+273.15)K))
                            )
                )

# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Dike_params);

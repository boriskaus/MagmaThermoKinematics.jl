# Unzen setup

using MagmaThermoKinematics
# using CUDA                        # for an NVIDIA GPU, then: backend = CUDABackend()
backend = CPU()
using GeophysicalModelGenerator
using GeoParams, Random
using CairoMakie                        # plots
using MagmaThermoKinematics.MTK_GMG     # Allow overwriting user routines


# Model setup
println(" --- Generating Setup --- ")

# Topography and project it.

# NOTE: The first time you do this, please set this to true, which will download the topography data from the internet and save it in a file
if false
    using GMT, Statistics
    Topo = ImportTopo(lon = [130.0, 130.5], lat = [32.55, 32.9], file = "@earth_relief_03s.grd")
    proj = ProjectionPoint(; Lat = mean(Topo.lat.val), Lon = mean(Topo.lon.val))
    Topo_cart = Convert2CartData(Topo, proj)
    Xt, Yt, Zt = xyz_grid(-23:0.1:23, -19:0.1:19, 0)
    Topo_cart = ProjectCartData(CartData(Xt, Yt, Zt, (Zt = Zt,)), Topo, proj)

    save_GMG(joinpath(@__DIR__, "Topo_cart"), Topo_cart)
end
Topo_cart = load_GMG(joinpath(@__DIR__, "Topo_cart"))

# Create 3D grid of the region
X, Y, Z = xyz_grid(-23:0.1:23, -19:0.1:19, -20:0.1:5)
Data_set3D = CartData(X, Y, Z, (Phases = zeros(Int64, size(X)), Temp = zeros(size(X))));       # 3D dataset

# Create 2D cross-section
Nx = 135 * 6;  # resolution in x
Nz = 135 * 4;
Data_2D = cross_section(Data_set3D, Start = (-20, 4), End = (20, 4), dims = (Nx, Nz))
Data_2D = addfield(Data_2D, "FlatCrossSection", flatten_cross_section(Data_2D))
Data_2D = addfield(Data_2D, "Phases", Int64.(Data_2D.fields.Phases))

# Intersect with topography
Below = below_surface(Data_2D, Topo_cart)
Data_2D.fields.Phases[Below] .= 1

# Set Moho
@views Data_2D.fields.Phases[Data_2D.z.val .< -30.0] .= 2

# Set T:
gradient = 30
Data_2D.fields.Temp .= -Data_2D.z.val * gradient
@views Data_2D.fields.Temp[Data_2D.fields.Temp .< 10.0] .= 10

# Set thermal anomaly
x_c, z_c, r = -10, -15, 2.5
Volume = 4 / 3 * pi * r^3 # equivalent 3D volume of the anomaly [km^3]
@views Data_2D.fields.Temp[(Data_2D.x.val .- x_c) .^ 2 .+ (Data_2D.z.val .- z_c) .^ 2 .< r^2] .= 800.0

println(" --- Performing MTK models --- ")

# Overwrite some of the default functions
if !(backend isa CPU)
    function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
        println("$(Num.it), Time=$(round(Num.time / Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
        return nothing
    end
else
    function MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
        if mod(Num.it, Num.CreateFig_steps) == 0
            x_1d = Grid.coord1D[1] / 1.0e3
            z_1d = Grid.coord1D[2] / 1.0e3
            temp_data = Array(Arrays.Tnew)
            ϕ_data = Array(Arrays.ϕ)
            phase_data = Float64.(Array(Arrays.Phases))

            # remove topo on plots
            ind = findall(phase_data .== 0)
            phase_data[ind] .= NaN
            temp_data[ind] .= NaN

            t = Num.time / SecYear / 1.0e3

            fig = Figure(size = (1000, 450))

            ax1 = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "z [km]", title = "Temperature, t=$(round(t)) kyrs", aspect = DataAspect(), limits = (nothing, (minimum(z_1d), 2)))
            Colorbar(fig[1, 2], heatmap!(ax1, x_1d, z_1d, temp_data, colormap = :viridis))
            ax2 = Axis(fig[1, 3], xlabel = "x [km]", ylabel = "z [km]", title = "Melt fraction", aspect = DataAspect(), limits = (nothing, (minimum(z_1d), 2)))
            Colorbar(fig[1, 4], heatmap!(ax2, x_1d, z_1d, ϕ_data, colormap = :viridis, colorrange = (0, 1)))
            #Colorbar(fig[1,4], heatmap!(ax2, x_1d, z_1d, phase_data, colormap=:viridis))

            save("MTK_GMG_2D_example2_$(Num.it).png", fig)
        end
        return nothing
    end
end

function MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    return nothing
end

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    println("$(Num.it), Time=$(round(Num.time / Num.SecYear / 1.0e3, digits = 3)) kyrs; max(T) = $(round(maximum(Arrays.Tnew)))")
    return nothing
end

"""
    MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

Update time-dependent properties during a simulation
"""
function MTK_GMG.MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    push!(time_props.Time_vec, Num.time)    # time
    push!(time_props.MeltFraction, sum(Arrays.ϕ) / (Num.Nx * Num.Nz))     # melt fraction

    ind = findall(Arrays.T .> 700)
    if ~isempty(ind)
        Tav_magma_Time = sum(Arrays.T[ind]) / length(ind)     # average T of part with magma
    else
        Tav_magma_Time = NaN
    end
    push!(time_props.Tav_magma, Tav_magma_Time)         # average magma T
    push!(time_props.Tmax, maximum(Arrays.T))      # maximum magma T
    return nothing
end

# Define a new structure with time-dependent properties
@with_kw mutable struct TimeDepProps1 <: TimeDependentProperties
    Time_vec::Vector{Float64} = []            # Center of dike
    MeltFraction::Vector{Float64} = []            # Melt fraction over time
    Tav_magma::Vector{Float64} = []            # Average magma
    Tmax::Vector{Float64} = []            # Max magma temperature
    Tmax_1::Vector{Float64} = []            # Another magma temperature vector
end

# Define numerical parameters
Num = NumParam(
    SimName = "Unzen1",
    dim = 2,
    maxTime_Myrs = 0.005,
    SaveOutput_steps = 25,
    CreateFig_steps = 5,
    backend = backend,
    ω = 0.5,
    AddRandomSills = true,
    RandomSills_timestep = 5
);

# Default setup: ElasticDike equivalent via PennyShapedSill.
sill = PennyShapedSill(Center = Point2(0.0, -7.0e3) * m, R = 2.5e3 * m, H = 250 * m, E = 1.5e10 * Pa, ν = 0.3 * NoUnits)

# Alternative sill definitions (currently unused):
# sill = CylindricalDikeTopAccretion(Center=Point2(0.0, -7.0e3) * m, W=5e3 * m, H=250 * m)
# sill = EllipticalIntrusion(Center=Point2(0.0, -7.0e3) * m, W=5e3 * m, H=250 * m)

Sill_params = SillParams(
    sill = sill,
    InjectionInterval_year = 1000,
    nTr_dike = 2000,
    H_ran = 5000,
    W_ran = 5000,
    Dip_ran = 45,
    SillPhase = 3,
    BackgroundPhase = 1,
    SillsAbove = -10.0e3,
)

# Keep random sill relocation and actual injection in sync.
if Num.AddRandomSills
    Sill_params.InjectionInterval = Num.dt * Num.RandomSills_timestep
    Sill_params.InjectionInterval_year = Sill_params.InjectionInterval / SecYear
end

# Define parameters for the different phases
MatParam = (
    SetMaterialParams(
        Name = "Air", Phase = 0,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 0.0J / kg),
        Conductivity = ConstantConductivity(k = 3Watt / K / m),          # in case we use constant k
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())
    ),          # Marxer & Ulmer melting
    SetMaterialParams(
        Name = "Crust", Phase = 1,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
        Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())
    ),      # Marxer & Ulmer melting
    SetMaterialParams(
        Name = "Mantle", Phase = 2,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
        Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K)
    ),
    SetMaterialParams(
        Name = "Dikes", Phase = 3,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
        Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())
    ),           # Marxer & Ulmer melting
)

# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params, CartData_input = Data_2D, time_props = TimeDepProps1()); # start the main code

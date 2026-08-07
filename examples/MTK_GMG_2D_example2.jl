# Unzen setup
const USE_GPU=false;
if USE_GPU
    using CUDA      # needs to be loaded before loading Parallkel=
end
using ParallelStencil, ParallelStencil.FiniteDifferences2D

using MagmaThermoKinematics
@static if USE_GPU
    environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
    CUDA.device!(0)                     # select the GPU you use (starts @ zero)
    @init_parallel_stencil(CUDA, Float64, 2)
else
    environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
    @init_parallel_stencil(Threads, Float64, 2)
end
using MagmaThermoKinematics.Diffusion2D # to load AFTER calling environment!()
using MagmaThermoKinematics.Fields2D
using MagmaThermoKinematics.MTK_GMG_2D
using GeophysicalModelGenerator
using GeoParams, Random
using Plots                             # plots
using MagmaThermoKinematics.MTK_GMG     # Allow overwriting user routines
using InjectSills

# Model setup
println(" --- Generating Setup --- ")

# Topography and project it.

# NOTE: The first time you do this, please set this to true, which will download the topography data from the internet and save it in a file
if false
    using GMT, Statistics
    Topo       =   ImportTopo(lon = [130.0, 130.5], lat=[32.55, 32.90], file="@earth_relief_03s.grd")
    proj       =   ProjectionPoint(; Lat=mean(Topo.lat.val), Lon=mean(Topo.lon.val))
    Topo_cart  =   Convert2CartData(Topo, proj)
    Xt,Yt,Zt   =   xyz_grid(-23:.1:23,-19:.1:19,0)
    Topo_cart  =   ProjectCartData(CartData(Xt,Yt,Zt,(Zt=Zt,)), Topo, proj)

    save_GMG("Topo_cart", Topo_cart)
end
# Topo_cart = load_GMG("Topo_cart")
Topo_cart = load_GMG("examples/Topo_cart")

# Create 3D grid of the region
X,Y,Z       =   xyz_grid(-23:.1:23,-19:.1:19,-20:.1:5)
Data_set3D  =   CartData(X,Y,Z,(Phases=zeros(Int64,size(X)),Temp=zeros(size(X))));       # 3D dataset

# Create 2D cross-section
Nx      =   135*6;  # resolution in x
Nz      =   135*4;
Data_2D =   cross_section(Data_set3D, Start=(-20,4), End=(20,4), dims=(Nx, Nz))
Data_2D =   addfield(Data_2D,"FlatCrossSection", flatten_cross_section(Data_2D))
Data_2D =   addfield(Data_2D,"Phases", Int64.(Data_2D.fields.Phases))

# Intersect with topography
Below   =   below_surface(Data_2D, Topo_cart)
Data_2D.fields.Phases[Below] .= 1

# Set Moho
@views Data_2D.fields.Phases[Data_2D.z.val .< -30.0] .= 2

# Set T:
gradient = 30
Data_2D.fields.Temp .= -Data_2D.z.val*gradient
@views Data_2D.fields.Temp[Data_2D.fields.Temp .< 10.0] .= 10

# Set thermal anomaly
x_c, z_c, r = -10, -15, 2.5
Volume      = 4/3*pi*r^3 # equivalent 3D volume of the anomaly [km^3]
@views Data_2D.fields.Temp[(Data_2D.x.val .- x_c).^2 .+ (Data_2D.z.val .- z_c).^2 .< r^2] .= 800.0

println(" --- Performing MTK models --- ")

# Overwrite some of the default functions
@static if USE_GPU
    function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
        println("$(Num.it), Time=$(round(Num.time/Num.SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
        return nothing
    end
else
    function MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
        if mod(Num.it,Num.CreateFig_steps)==0
            x_1d        =   Grid.coord1D[1]/1e3;
            z_1d        =   Grid.coord1D[2]/1e3;
            temp_data   =   Array(Arrays.Tnew)'
            ϕ_data      =   Array(Arrays.ϕ)'
            phase_data  =   Float64.(Array(Arrays.Phases))'

            # remove topo on plots
            ind             = findall(phase_data .== 0)
            phase_data[ind] .= NaN
            temp_data[ind]  .= NaN

            t = Num.time/SecYear/1e3;

            p=plot(layout=grid(1,2) )

            Plots.heatmap!(p[1],x_1d, z_1d, temp_data, c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Temperature, t=$(round(t)) kyrs", aspect_ratio=:equal,  ylimits=(minimum(z_1d),2))
            Plots.heatmap!(p[2],x_1d, z_1d, ϕ_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction", clims=(0,1), aspect_ratio=:equal, ylimits=(minimum(z_1d),2))
            #Plots.heatmap!(p[2],x_1d, z_1d, phase_data,    c=:viridis, xlabel="x [km]", ylabel="z [km]", title="Melt fraction", aspect_ratio=:equal, ylimits=(minimum(z_1d),2))

            display(p)
        end
        return nothing
    end
end

function MTK_GMG.MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    return nothing
end

function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    println("$(Num.it), Time=$(round(Num.time/Num.SecYear/1e3, digits=3)) kyrs; max(T) = $(round(maximum(Arrays.Tnew)))")
    return nothing
end

"""
    MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

Update time-dependent properties during a simulation
"""
function MTK_GMG.MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    push!(time_props.Time_vec,      Num.time);   # time
    push!(time_props.MeltFraction,  sum( Arrays.ϕ)/(Num.Nx*Num.Nz));    # melt fraction

    ind = findall(Arrays.T.>700);
    if ~isempty(ind)
        Tav_magma_Time = sum(Arrays.T[ind])/length(ind)     # average T of part with magma
    else
        Tav_magma_Time = NaN;
    end
    push!(time_props.Tav_magma, Tav_magma_Time);        # average magma T
    push!(time_props.Tmax,      maximum(Arrays.T));     # maximum magma T
    return nothing
end

# Define a new structure with time-dependent properties
@with_kw mutable struct TimeDepProps1 <: TimeDependentProperties
    Time_vec::Vector{Float64}       = [];           # Center of dike
    MeltFraction::Vector{Float64}   = [];           # Melt fraction over time
    Tav_magma::Vector{Float64}      = [];           # Average magma
    Tmax::Vector{Float64}           = [];           # Max magma temperature
    Tmax_1::Vector{Float64}         = [];           # Another magma temperature vector
end

# Define numerical parameters
Num         = NumParam( SimName             =   "Unzen1",
                        # Nx                  =   64,
                        Nz                  =   64,
                        dim                 =   2,
                        maxTime_Myrs        =   0.005,
                        SaveOutput_steps    =   25,
                        CreateFig_steps     =   5,
                        USE_GPU             =   USE_GPU,
                        ω                   =   0.5,
                        AddRandomSills      =   true,
                        RandomSills_timestep=   5,
                        deform_hostrock=   false);

# Default setup: ElasticDike equivalent via PennyShapedSill.
sill = PennyShapedSill(Center=Point2(0.0, -7.0e3) * m, W=2.5e3 * m, H=250 * m, E=1.5e10 * Pa, ν=0.3 * NoUnits)

# Alternative sill definitions (currently unused):
# sill = CylindricalDikeTopAccretion(Center=Point2(0.0, -7.0e3) * m, W=5e3 * m, H=250 * m)
# sill = EllipticalIntrusion(Center=Point2(0.0, -7.0e3) * m, W=5e3 * m, H=250 * m)

Sill_params = SillParams(
    sill                    = sill,
    InjectionInterval_year  = 1000,
    nTr_dike                = 2000,
    H_ran                   = 5000,
    W_ran                   = 5000,
    Dip_ran                 = 45,
    SillPhase               = 3,
    BackgroundPhase         = 1,
    SillsAbove              = -10e3,
)

Erupt = EruptionParams(
    erupt            = true,
    ϕ_erupt          = 0.5,        # mobile-melt threshold
    V_crit_km3       = 10e0,       # critical eruptible volume [km³, 3D]
    erupt_efficiency = 0.5,        # fraction of mobile melt withdrawn per event
    deflate          = true,       # also subside the chamber (deflation_model defaults to :local_mogi)
    out_of_plane_3D  = true,       # 2D eruptible volume → 3D via Gaussian out-of-plane (km³)
    T_min            = 0.0,        # floor for the thermal-extraction cooling
)

# --- topography along the cross-section, as a *function* f(x) [m, z up] ---------------
# Built at Data_2D's own resolution (size(Ph)), NOT at Num.Nx/Num.Nz: at this point
# Num.Nx/Nz are still the NumParam defaults — Setup_Model_CartData only sets them to the
# real model size *inside* MTK_GeoParams_2D. Passing a function (instead of a length-Nx
# array) keeps this Nx/Ny-agnostic: init_free_surface evaluates f at the actual model
# grid coordinates, so the same topography works at 810×540, 201×201 or 64×64 without
# allocating a resolution-sized array or having to match lengths.
Ph        = Data_2D.fields.Phases
Nxd, Nzd  = size(Ph)
xprof     = reshape(Data_2D.x.val, Nxd, Nzd)[:, 1] .* 1e3        # section x [m] (same frame as the model grid: CreateGrid uses extrema(x))
zcol      = reshape(Data_2D.z.val, Nxd, Nzd)[1, :] .* 1e3        # z down a column [m], increasing up
Bmat      = reshape(Below,         Nxd, Nzd)                     # below-surface mask, same (Nx,Nz) grid
zprof     = map(1:Nxd) do i
    k = findlast(@view Bmat[i, :])                              # topmost below-surface cell
    isnothing(k) ? zcol[1] : zcol[k]                            # fallback: domain floor
end

# linear-interpolation closure over the (small, 1D) surface profile
topography_fun(x) = let xs = xprof, zs = zprof
    x <= xs[1]   ? zs[1]   :
    x >= xs[end] ? zs[end] :
    let j = searchsortedlast(xs, x), t = (x - xs[j]) / (xs[j+1] - xs[j])
        zs[j] * (1 - t) + zs[j+1] * t
    end
end

FS = FreeSurfaceParams(
    free_surface = true,
    air_phase    = 0,
    Tair         = 0.0,
    topography   = topography_fun,                           # f(x) [m, z up]; evaluated at the real model grid
)
# Keep random sill relocation and actual injection in sync.
if Num.AddRandomSills
    Sill_params.InjectionInterval = Num.dt * Num.RandomSills_timestep
    Sill_params.InjectionInterval_year = Sill_params.InjectionInterval / SecYear
end

# Define parameters for the different phases
MatParam     = (SetMaterialParams(Name="Air", Phase=0,
                                Density      = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat   = ConstantLatentHeat(Q_L=0.0J/kg),
                                Conductivity = ConstantConductivity(k=3Watt/K/m),          # in case we use constant k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting      = SmoothMelting(MeltingParam_4thOrder())),          # Marxer & Ulmer melting
                SetMaterialParams(Name="Crust", Phase=1,
                                Density      = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat   = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting      = SmoothMelting(MeltingParam_4thOrder())),      # Marxer & Ulmer melting
                SetMaterialParams(Name="Mantle", Phase=2,
                                Density      = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat   = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K)),
                SetMaterialParams(Name="Dikes", Phase=3,
                                Density      = ConstantDensity(ρ=2700kg/m^3),
                                LatentHeat   = ConstantLatentHeat(Q_L=3.13e5J/kg),
                                Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
                                HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                                Melting      = SmoothMelting(MeltingParam_4thOrder()))           # Marxer & Ulmer melting
                )

# Call the main code with the specified material parameters
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Sill_params, CartData_input=Data_2D, time_props=TimeDepProps1();  Erupt=Erupt, FS=FS); # start the main code

println("Done: $(Erupt.n_eruptions) eruptions, total erupted melt = " *
        "$(round(Erupt.erupted_volume/1e9, digits=3)) km³; " *
        "final surface min/max = $(round(minimum(FS.z_surf)/1e3,digits=3)) / " *
        "$(round(maximum(FS.z_surf)/1e3,digits=3)) km")


using ZirconGrowth
cargo = Tracers[Tracers.erupted];
age_years, zircon_radius_um = simulate_zircon_growth_from_tracers(cargo)


# age_years_full, zircon_radius_um_full, results_full = simulate_zircon_growth_from_tracers(
    # cargo; return_results = true)

# ---------------------------------------------------------------------------
# Summary figure: erupted volume, eruption times, and the erupted-zircon age
# distribution.
#   • Erupt.eruption_times   — eruption time of each event [s]
#   • Erupt.eruption_volumes — erupted melt volume of each event [m³]
#   • age_years              — volume-averaged crystallisation age (years before
#                              the END of the run) of each successfully grown
#                              cargo zircon. `simulate_zircon_growth_from_tracers`
#                              skips tracers with <2 Tt-points, so age_years can
#                              be shorter than `cargo`; we rebuild the matching
#                              per-tracer eruption time with the SAME rule so the
#                              two arrays stay aligned.
# ---------------------------------------------------------------------------
if Erupt.n_eruptions == 0
    println("No eruptions occurred — nothing to plot.")
else
    t_erupt_kyr = Erupt.eruption_times   ./ SecYear ./ 1e3      # eruption time [kyr]
    V_erupt_km3 = Erupt.eruption_volumes ./ 1e9                 # per-event erupted V [km³]
    V_cum_km3   = cumsum(V_erupt_km3)                           # cumulative erupted V [km³]

    cargo_ok = [tr for tr in cargo if length(tr.time_vec) >= 2] # same skip rule as the ext
    t_zr_kyr = Float64[tr.time_vec[end]*1e3 for tr in cargo_ok] # eruption time of each zircon [kyr]
    age_kyr  = age_years ./ 1e3                                 # zircon age [kyr]

    psum = plot(layout=(3,1), size=(820,920), left_margin=8Plots.mm)

    # (1) erupted volume over time — per-event sticks + cumulative on a twin axis
    plot!(psum[1], t_erupt_kyr, V_erupt_km3, seriestype=:sticks, lw=2, c=:steelblue,
          marker=:circle, ms=5, label="per eruption", legend=:topleft,
          xlabel="time [kyr]", ylabel="erupted V [km³]", title="Erupted volume over time")
    pcum = twinx(psum[1])
    plot!(pcum, t_erupt_kyr, V_cum_km3, lw=2, c=:firebrick, marker=:diamond, ms=4,
          ylabel="cumulative V [km³]", label="cumulative", legend=:bottomright)

    # (2) erupted-zircon age distribution
    if isempty(age_kyr)
        plot!(psum[2], framestyle=:none, title="No grown zircons (cargo Tt-paths too short)")
    else
        histogram!(psum[2], age_kyr, bins=25, c=:darkorange, alpha=0.85, label="",
                   xlabel="zircon age [kyr] (before end of run)", ylabel="count",
                   title="Erupted-zircon age distribution (n=$(length(age_kyr)))")
        vline!(psum[2], [sum(age_kyr)/length(age_kyr)], lw=2, c=:black, ls=:dash, label="mean")
    end

    # (3) zircon age vs the time it erupted
    if !isempty(age_kyr)
        scatter!(psum[3], t_zr_kyr, age_kyr, ms=3, alpha=0.4, c=:purple, label="",
                 xlabel="eruption time [kyr]", ylabel="zircon age [kyr]",
                 title="Zircon age vs eruption time")
    end

    figpath = joinpath(Num.SimName, "eruption_zircon_summary.png")
    savefig(psum, figpath)
    display(psum)
    println("Saved summary figure to: $figpath  " *
            "($(Erupt.n_eruptions) eruptions, $(length(age_kyr)) zircons grown)")
end

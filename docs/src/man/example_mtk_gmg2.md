# MTK_GMG Example 2 (Unzen)

This page documents the coding style used in [examples/MTK_GMG_2D_example2.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/MTK_GMG_2D_example2.jl).

## GeophysicalModelGenerator-Driven Setup

```julia
# NOTE: topography preparation is typically done once and loaded from file
Topo_cart = load_GMG("Topo_cart")

X,Y,Z       = xyz_grid(-23:.1:23,-19:.1:19,-20:.1:5)
Data_set3D  = CartData(X,Y,Z,(Phases=zeros(Int64,size(X)),Temp=zeros(size(X))))

Nx = 135*6
Nz = 135*4
Data_2D = cross_section(Data_set3D, Start=(-20,4), End=(20,4), dims=(Nx, Nz))
Data_2D = addfield(Data_2D, "FlatCrossSection", flatten_cross_section(Data_2D))
Data_2D = addfield(Data_2D, "Phases", Int64.(Data_2D.fields.Phases))

Below = below_surface(Data_2D, Topo_cart)
Data_2D.fields.Phases[Below] .= 1
@views Data_2D.fields.Phases[Data_2D.z.val .< -30.0] .= 2
```

## Time-Dependent Diagnostics Hook

```julia
@with_kw mutable struct TimeDepProps1 <: TimeDependentProperties
    Time_vec::Vector{Float64}       = []
    MeltFraction::Vector{Float64}   = []
    Tav_magma::Vector{Float64}      = []
    Tmax::Vector{Float64}           = []
    Tmax_1::Vector{Float64}         = []
end

function MTK_GMG.MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    push!(time_props.Time_vec, Num.time)
    push!(time_props.MeltFraction, sum(Arrays.ϕ)/(Num.Nx*Num.Nz))
    push!(time_props.Tmax, maximum(Arrays.T))
    return nothing
end
```

## Parameter Setup and Run

```julia
Num = NumParam(SimName="Unzen1", dim=2, maxTime_Myrs=0.005, USE_GPU=USE_GPU, AddRandomSills=true)
Dike_params = DikeParam(Type="ElasticDike", InjectionInterval_year=1000, W_in=5e3, H_in=250, DikePhase=3)

# MatParam tuple omitted for brevity; define all phases as in the example file.
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(
    MatParam,
    Num,
    Dike_params,
    CartData_input=Data_2D,
    time_props=TimeDepProps1(),
)
```

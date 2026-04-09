"""
Module MagmaThermoKinematics

Enables Earth Scientists to simulate the thermal evolution of magmatic systems.

"""
module MagmaThermoKinematics

# list required modules
using Reexport
using Random                                    # random numbers
using StructArrays                              # for tracers and dike polygon
using Parameters                                # More flexible definition of parameters
using Interpolations                            # Fast interpolations
using StaticArrays
using JLD2                                      # Load/save data to disk
@reexport using GeoParams                                 # Material parameters calculations
@reexport using ParallelStencil

abstract type NumericalParameters end
abstract type DikeParameters end
abstract type TimeDependentProperties end

include("Units.jl")                             # various useful units

# Few useful parameters
const SecYear     = 3600*24*365.25
const kyr         = 1000*SecYear
const Myr         = 1e6*SecYear
const km³         = 1000^3
export SecYear, kyr, Myr, km³

export NumericalParameters, DikeParameters, TimeDependentProperties

struct EnvironmentConfig
    model_device::Symbol
    precision::DataType
    dimension::Int
end

const _environment_config = Ref{Union{Nothing, EnvironmentConfig}}(nothing)

@inline function _normalize_environment_config(model_device, precision, dimension)
    model_device in (:cpu, :gpu, :metal) || throw(ArgumentError("Unsupported model_device=$model_device. Use :cpu, :gpu or :metal."))
    precision isa DataType || throw(ArgumentError("precision must be a type (e.g. Float64), got $(typeof(precision))."))
    dimension isa Integer || throw(ArgumentError("dimension must be an integer, got $(typeof(dimension))."))

    dim = Int(dimension)
    dim in (2, 3) || throw(ArgumentError("Unsupported dimension=$dim. Use 2 or 3."))

    if model_device == :metal && precision != Float32
        throw(ArgumentError("Metal backend currently supports only Float32 precision. Got $precision."))
    end

    return EnvironmentConfig(model_device, precision, dim)
end

@inline _compute_kernel_names() = (:meltfraction, :dϕdT, :density, :heatcapacity, :conductivity, :radioactive_heat, :latent_heat)

@inline _compute_symbol(name::Symbol) = Symbol(:compute_, name)
@inline _compute_symbol_ps(name::Symbol) = Symbol(:compute_, name, :_ps!)
@inline _compute_symbol_ps_3D(name::Symbol) = Symbol(:compute_, name, :_ps_3D!)

for kernel_name in _compute_kernel_names()
    fn = _compute_symbol_ps(kernel_name)
    fn_3D = _compute_symbol_ps_3D(kernel_name)
    @eval begin
        function $(fn) end
        function $(fn_3D) end
        export $(fn), $(fn_3D)
    end
end

function environment!(model_device, precision, dimension)
    config = _normalize_environment_config(model_device, precision, dimension)
    if _environment_config[] == config
        return nothing
    end
    if !isnothing(_environment_config[]) && isinteractive()
        @warn "Reinitializing ParallelStencil from $(_environment_config[]) to $config."
    end

    # call appropriate FD module
    finite_differences_module = config.dimension == 2 ? :FiniteDifferences2D : :FiniteDifferences3D
    Base.eval(@__MODULE__, Expr(:using, Expr(:., :ParallelStencil, finite_differences_module)))

    # start ParallelStencil
    if config.model_device == :gpu
        println("Using GPU for ParallelStencil")
        Base.eval(@__MODULE__, :(using CUDA))
        @eval begin
             ParallelStencil.@reset_parallel_stencil()
             @init_parallel_stencil(CUDA, $(config.precision), $(config.dimension))
        end
    elseif config.model_device == :metal
        println("Using Metal for ParallelStencil")
        Base.eval(@__MODULE__, :(using Metal))
        @eval begin
             ParallelStencil.@reset_parallel_stencil()
             @init_parallel_stencil(Metal, $(config.precision), $(config.dimension))
        end
    else
        println("Using CPU for ParallelStencil")
        @eval begin
             ParallelStencil.@reset_parallel_stencil()
             @init_parallel_stencil(Threads, $(config.precision), $(config.dimension))
        end
    end

        # GeoParams routines we want to work on GPU:
        for kernel_name in _compute_kernel_names()
            fn = _compute_symbol_ps(kernel_name)
            _fn = _compute_symbol(kernel_name)
            fn_3D = _compute_symbol_ps_3D(kernel_name)
      @eval begin
        # 2D version
        @parallel_indices (i, j) function $(fn)(A,MatParam, Phases, args)
              k = keys(args)
              v = getindex.(values(args), i, j)
              argsi = (; zip(k, v)...)
              A[i, j] = $(_fn)(MatParam, Phases[i,j], argsi)
              return
          end

        # Special version for multiple phaes
        @parallel_indices (i,j) function $(fn)(
            rho::AbstractArray,
            MatParam::Tuple,
            Phases::AbstractArray,
            args,
        )
            k = keys(args)
            v = getindex.(values(args), i,j)
            argsi = (; zip(k, v)...)
            rho[i,j] = compute_param($(_fn), MatParam, Phases[i,j], argsi)
            return
        end

        # 3D version
        @parallel_indices (i, j, k) function $(fn_3D)(A,MatParam, Phases, args)
            k_3D = keys(args)
            v_3D= getindex.(values(args), i, j, k)
            argsi = (; zip(k_3D, v_3D)...)
            A[i, j, k] = $(_fn)(MatParam[Phases[i,j,k]], argsi)
            return
        end

        # Special version for multiple phaes
        @parallel_indices (i,j,k) function $(fn_3D)(
            A::AbstractArray,
            MatParam::Tuple,
            Phases::AbstractArray,
            args,
        )
            k_3D = keys(args)
            v_3D= getindex.(values(args), i, j, k)
            argsi = (; zip(k_3D, v_3D)...)
            A[i,j,k] = compute_param($(_fn), MatParam, Phases[i,j,k], argsi)
            return
        end

      end
    end

    # conditional submodule load
    if config.model_device == :gpu
        Base.@eval begin
            include(joinpath(@__DIR__, "CUDA/DiffusionCUDA.jl"))
        end
    elseif config.model_device == :metal
        Base.@eval begin
            include(joinpath(@__DIR__, "Metal/DiffusionMetal.jl"))
        end
    else
        Base.@eval begin
            include(joinpath(@__DIR__, "Threads/Diffusion.jl"))
        end
    end


    # Create arrays (depends on PS, so should be loaded after)
    if config.model_device == :gpu
        Base.@eval begin
            include(joinpath(@__DIR__, "CUDA/FieldsCUDA.jl"))
        end
    elseif config.model_device == :metal
        Base.@eval begin
            include(joinpath(@__DIR__, "Metal/FieldsMetal.jl"))
        end
    else
        Base.@eval begin
            include(joinpath(@__DIR__, "Threads/FieldsThreads.jl"))
        end
    end

    # GMG integration
    if config.model_device == :gpu
        Base.@eval begin
            include(joinpath(@__DIR__, "CUDA/MTK_GMG_2D_CUDA.jl"))

            include(joinpath(@__DIR__, "CUDA/MTK_GMG_3D_CUDA.jl"))
        end
    elseif config.model_device == :metal
        Base.@eval begin
            include(joinpath(@__DIR__, "Metal/MTK_GMG_2D_Metal.jl"))

            include(joinpath(@__DIR__, "Metal/MTK_GMG_3D_Metal.jl"))
        end
    else
        Base.@eval begin
            include(joinpath(@__DIR__, "Threads/MTK_GMG_2D.jl"))

            include(joinpath(@__DIR__, "Threads/MTK_GMG_3D.jl"))
        end
    end

    _environment_config[] = config

    return nothing



end

export environment!

include("Grid.jl")
using .Grid
export GridData, CreateGrid

# Routines that deal with tracers
include("Tracers.jl")
export UpdateTracers, AdvectTracers!, InitializeTracers,PhaseRatioFromTracers, CorrectTracersForTopography!
export RockAssemblage, update_Tvec!
export PhaseRatioFromTracers!, PhasesFromTracers!, UpdateTracers_T_ϕ!, UpdateTracers_Field! # new routines


include("MeltingRelationships.jl")
export SolidFraction, ComputeLithostaticPressure, LoadPhaseDiagrams, PhaseDiagramData, ComputeDensityAndPressure
export PhaseRatioAverage!, ComputeSeismicVelocities, SolidFraction_Parameterized!

# Export functions that will be available outside this module
export StructArray, LazyRow # useful
export Tracer

include("Dikes.jl")
export Dike, DikePoly
export Tracer, AddDike, HostRockVelocityFromDike, CreateDikePolygon, advect_dike_polygon!,
       volume_dike, InjectDike, TracersToGrid!

# routines related to advection & interpolation
include("Advection.jl")
export AdvectTemperature, Interpolate!, CorrectBounds, evaluate_interp_2D, evaluate_interp_3D

# Shared utility routines and MTK/GMG structs are loaded at module initialization
# to avoid defining new global bindings inside environment!.
include("Utils.jl")
export Process_ZirconAges, copy_arrays_GPU2CPU!, copy_arrays_CPU2GPU!

include("MTK_GMG_structs.jl")
export NumParam, DikeParam, TimeDepProps

include("MTK_GMG.jl")

# Routines related to Parameters.jl, which come in handy in the main routine
export @unpack, @with_kw



end # module

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
SecYear     = 3600*24*365.25
kyr         = 1000*SecYear
Myr         = 1e6*SecYear  
km³         = 1000^3
export SecYear, kyr, Myr, km³

export NumericalParameters, DikeParameters, TimeDependentProperties

include("Grid.jl")
using .Grid
export GridData, CreateGrid

# Routines that deal with tracers
include("Tracers.jl")
export UpdateTracers, AdvectTracers!, InitializeTracers,PhaseRatioFromTracers, CorrectTracersForTopography!
export RockAssemblage, update_Tvec!
export PhaseRatioFromTracers!, PhasesFromTracers!, UpdateTracers_T_ϕ!, UpdateTracers_Field! # new routines

function environment!(model_device, precision, dimension)
    gpu = model_device == :gpu ? true : false

    # environment variable for XPU
    @eval begin
        const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : $gpu
    end

    # call appropriate FD module
    Base.eval(@__MODULE__, Meta.parse("using ParallelStencil.FiniteDifferences$(dimension)D"))
    eval(Meta.parse("using ParallelStencil.FiniteDifferences$(dimension)D"))

    # start ParallelStencil
    if model_device == :gpu
        # eval(:(@init_parallel_stencil(CUDA, $(precision), $(dimension))))
        Base.eval(@__MODULE__, :(@init_parallel_stencil(CUDA, $(precision), $(dimension))))
        Base.eval(Main, Meta.parse("using CUDA"))
    else
        @eval begin
            ParallelStencil.@reset_parallel_stencil()
            @init_parallel_stencil(Threads, $(precision), $(dimension))
        end
    end

    # GeoParams routines we want to work on GPU:
    for fni in ("meltfraction","dϕdT","density","heatcapacity","conductivity","radioactive_heat","latent_heat")
      fn = Symbol(string("compute_$(fni)_ps!"))
      _fn = Symbol(string("compute_$(fni)"))
      fn_3D = Symbol(string("compute_$(fni)_ps_3D!"))
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

        export $fn
        export $fn_3D
      end
    end

    # conditional submodule load
    module_names = Symbol("Diffusion$(dimension)D")
    Base.@eval begin
        include(joinpath(@__DIR__, "Diffusion.jl"))
        @reexport import .$module_names
        export Data
    end

    # Create arrays (depends on PS, so should be loaded after)
    include(joinpath(@__DIR__, "Fields.jl"))
    Base.@eval begin
        using .Fields
        export CreateArrays
    end

    # Various helpful routines
    Base.@eval begin
        include(joinpath(@__DIR__, "Utils.jl"))
        export Process_ZirconAges, copy_arrays_GPU2CPU!, copy_arrays_CPU2GPU!
    end

    # GMG integration
    Base.@eval begin
        include(joinpath(@__DIR__, "MTK_GMG_structs.jl"))
        export NumParam, DikeParam, TimeDepProps

        include(joinpath(@__DIR__, "MTK_GMG.jl"))
        
        include(joinpath(@__DIR__, "MTK_GMG_2D.jl"))
        using .MTK_GMG_2D
        export MTK_GeoParams_2D

        include(joinpath(@__DIR__, "MTK_GMG_3D.jl"))
        using .MTK_GMG_3D
        export MTK_GeoParams_3D

    end

end

export environment!


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

# Routines related to Parameters.jl, which come in handy in the main routine
export @unpack, @with_kw



end # module

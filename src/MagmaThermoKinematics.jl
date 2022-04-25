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

include("Units.jl")                             # various useful units

function environment!(model_device, precission, dimension)
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
        # eval(:(@init_parallel_stencil(CUDA, $(precission), $(dimension))))
        Base.eval(@__MODULE__, :(@init_parallel_stencil(CUDA, $(precission), $(dimension))))
        Base.eval(Main, Meta.parse("using CUDA"))
    else
        @eval begin
            @init_parallel_stencil(Threads, $(precission), $(dimension))
        end
    end

    # conditional submodule load
    module_names = Symbol("Diffusion$(dimension)D")
    Base.@eval begin
        include(joinpath(@__DIR__, "Diffusion.jl"))
        @reexport import .$module_names
        export Data
    end

    for fni in ("meltfraction","dϕdT","density","heatcapacity","conductivity","radioactive_heat")
        fn = Symbol(string("compute_$(fni)_ps!"))
        _fn = Symbol(string("compute_$(fni)"))
        @eval begin
            @parallel_indices (i, j) function $(fn)(A,MatParam, Phases, args)
                k = keys(args)
                v = getindex.(values(args), i, j)
                argsi = (; zip(k, v)...)
                A[i, j] = $(Symbol(_fn))(MatParam[Phases[i,j]], argsi)
                return
            end
            export $fn
        end
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

# Routines that deal with tracers
include("Tracers.jl")
export UpdateTracers, AdvectTracers!, InitializeTracers,PhaseRatioFromTracers, CorrectTracersForTopography!
export RockAssemblage, update_Tvec!  

# Post-processing routines
include("Utils.jl")
export Process_ZirconAges

# Routines related to Parameters.jl, which come in handy in the main routine
export @unpack



end # module

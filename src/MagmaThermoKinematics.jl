"""
Module MagmaThermoKinematics

Enables Earth Scientists to simulate the thermal evolution of magmatic systems.

"""
module MagmaThermoKinematics

# list required modules
using Random                                    # random numbers
using StructArrays                              # for tracers and dike polygon
using Parameters                                # More flexible definition of parameters
using Interpolations                            # Fast interpolations
using StaticArrays                      
using CSV                                       # for reading phase diagrams

include("Units.jl")                             # various useful units

## Alphabetical include of computation-submodules (must be at end as needs to import from ParallelStencil).
include("Diffusion.jl")
export Diffusion2D, Diffusion3D  #

include("MeltingRelationships.jl")
export SolidFraction, ComputeLithostaticPressure, LoadPhaseDiagrams, PhaseDiagramData, ComputeDensityAndPressure
export PhaseRatioAverage!, ComputeSeismicVelocities, SolidFraction_Parameterized!

# Export functions that will be available outside this module
export StructArray, LazyRow # useful 
export Tracer 

include("Dikes.jl")
export Dike, DikePoly
export Tracer, AddDike, HostRockVelocityFromDike, CreatDikePolygon, volume_dike, InjectDike, TracersToGrid!

# routines related to advection & interpolation
include("Advection.jl")
export AdvectTemperature, Interpolate!, CorrectBounds, evaluate_interp_2D, evaluate_interp_3D    

# Routines that deal with tracers
include("Tracers.jl")
export UpdateTracers, AdvectTracers!, InitializeTracers,PhaseRatioFromTracers, CorrectTracersForTopography!
export RockAssemblage  

# Routines related to Parameters.jl, which come in handy in the main routine
export @unpack

end # module

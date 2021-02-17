"""
Module ZirconThermoKinematics

Enables Earth Scientists to simulate the thermal evolution of magmatic systems.

"""
module ZirconThermoKinematics


# list required modules
using Random                                    # random numbers
using StructArrays                              # for tracers and dike polygon
using Parameters                                # More flexible definition of parameters


## Alphabetical include of computation-submodules (must be at end as needs to import from ParallelStencil, .e.g. INDICES).
include("Diffusion.jl")
export Diffusion2D, Diffusion3D

include("MeltingRelationships.jl")
export SolidFraction!

# Export functions that will be available outside this module
export StructArray
export Tracer 
export Interpolate_Linear,  AdvectTemperature, AdvectTracers

const SecYear =   3600*24*365.25;                      # seconds/year
export SecYear


# routines related to advection 
include("Advection.jl")
export Interpolate_Linear, AdvectTracers, AdvectTemperature

include("Dikes.jl")
export Dike, DikePoly
export Tracer, AddDike, HostRockVelocityFromDike, CreatDikePolygon, volume_dike, InjectDike


# Routines that deal with tracers
include("Tracers.jl")
export UpdateTracers


# Routines related to Parameters.jl, which come in handy in the main routine
export @unpack

end # module

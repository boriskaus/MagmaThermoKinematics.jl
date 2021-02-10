"""
Module ZirconThermoKinematics

Enables Earth Scientists to simulate the thermal evolution of magmatic systems.

"""
module ZirconThermoKinematics


# list required modules
using Random                                    # random numbers
using StructArrays                              # for tracers and dike polygon



## Alphabetical include of computation-submodules (must be at end as needs to import from ParallelStencil, .e.g. INDICES).
include("Diffusion.jl")
export Diffusion2D, Diffusion3D

include("MeltingRelationships.jl")
using .MeltingRelationships
export SolidFraction

# Export functions that will be available outside this module
export StructArray
export Tracer 
export Interpolate_Linear,  AdvectTemperature, AdvectTracers

const SecYear =   3600*24*365.25;                      # seconds/year
export SecYear


include("Dikes.jl")
using .Dikes
export Dike, DikePoly, Tracer
export AddDike, HostRockVelocityFromDike, CreatDikePolygon

# routines related to advection 
include("Advection.jl")
using .Advection
export Interpolate_Linear, AdvectTracers, AdvectTemperature





end # module

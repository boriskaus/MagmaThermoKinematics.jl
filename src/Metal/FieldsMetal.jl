module Fields2D
using MagmaThermoKinematics
#environment!(:cpu, Float64, 2)

using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(Metal, Float32, 2)
using Metal

# Some helping routines that simplifies creating fields and work arrays
export CreateArrays

include(joinpath(@__DIR__, "..", "Fields.jl"))

end

module Fields3D
using MagmaThermoKinematics
#environment!(:cpu, Float64, 2)

using ParallelStencil
using ParallelStencil.FiniteDifferences3D

@init_parallel_stencil(Metal, Float32, 3)

using Metal

# Some helping routines that simplifies creating fields and work arrays
export CreateArrays

include(joinpath(@__DIR__, "..", "Fields.jl"))

end

module Fields2D
using MagmaThermoKinematics

using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(Threads, Float64, 2)

# Some helping routines that simplifies creating fields and work arrays
export CreateArrays

include(joinpath(@__DIR__, "..", "Fields.jl"))

end

module Fields3D
using MagmaThermoKinematics

using ParallelStencil
using ParallelStencil.FiniteDifferences3D

@init_parallel_stencil(Threads, Float64, 3)

# Some helping routines that simplifies creating fields and work arrays
export CreateArrays

include(joinpath(@__DIR__, "..", "Fields.jl"))

end

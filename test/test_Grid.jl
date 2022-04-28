using Test
using MagmaThermoKinematics

environment!(:cpu, Float64, 2) 

@testset "Grid" begin


# Create 2D grid
Grid = CreateGrid(size=(10,20),x=(0.,10), z=(2.,10))

@test Grid.L == (10.0, 8.0)
@test Grid.Δ[1] ≈ 1.1111111111111112
@test Grid.Δ[2] ≈ 0.42105263157894735



end

using Test
using MagmaThermoKinematics
environment!(:cpu, Float64, 2) 
using MagmaThermoKinematics.Diffusion2D

@testset "Grid" begin


# Create 2D grid
Grid = CreateGrid(size=(10,20),x=(0.,10), z=(2.,10))

@test Grid.L == (10.0, 8.0)
@test Grid.Δ[1] ≈ 1.1111111111111112
@test Grid.Δ[2] ≈ 0.42105263157894735


X = @zeros(Grid.N...)
Z = @zeros(Grid.N...)
@parallel (1:Grid.N[1], 1:Grid.N[2]) GridArray!(X,Z,Grid.coord1D[1], Grid.coord1D[2])

@test sum(X) ≈ 1000
@test minimum(Z) ==2.0


end

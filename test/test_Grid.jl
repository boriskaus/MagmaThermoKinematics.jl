using Test
using MagmaThermoKinematics
using GeophysicalModelGenerator: CartData, xyz_grid

@testset "Grid" begin


    # Create 2D grid
    Grid = CreateGrid(size = (10, 20), x = (0.0, 10), z = (2.0, 10))

    @test Grid.L == (10.0, 8.0)
    @test Grid.Δ[1] ≈ 1.1111111111111112
    @test Grid.Δ[2] ≈ 0.42105263157894735


    X = zeros(Grid.N...)
    Z = zeros(Grid.N...)
    GridArray!(X, Z, Grid)

    @test sum(X) ≈ 1000
    @test minimum(Z) == 2.0

    # 1D grid from scalar size and extent
    Grid1 = CreateGrid(size = 11, extent = 5.0)
    @test Grid1.N == (11,)
    @test Grid1.Δ == (0.5,)
    @test Grid1.coord1D[1] == 0:0.5:5

    # printing: compare without whitespace, so layout changes do not break the test
    nows(g) = filter(!isspace, sprint(show, g))
    @test occursin("domain:x∈[0.0,5.0]", nows(Grid1))
    @test occursin("domain:x∈[0.0,10.0],z∈[2.0,10.0]", nows(Grid))
    Grid3 = CreateGrid(size = (3, 4, 5), x = (0.0, 1.0), y = (2.0, 3.0), z = (-4.0, 0.0))
    @test occursin("Grid{Float64,3}", nows(Grid3))
    @test occursin("domain:x∈[0.0,1.0],y∈[2.0,3.0],z∈[-4.0,0.0]", nows(Grid3))

    # grid from a CartData set, with coordinates in km or m
    Xc, Yc, Zc = xyz_grid(0.0:4.0, 0.0:2.0, -3.0:0.0)
    d = CartData(Xc, Yc, Zc, (Z = Zc,))
    @test CreateGrid(d).max == (4.0e3, 2.0e3, 0.0)
    @test CreateGrid(d; m_to_km = false).min == (0.0, 0.0, -3.0)
    # 2D cross section: size (Nx, Nz, 1), z varying along the second dimension
    X2 = [x for x in 0.0:4.0, z in -3.0:0.0, _ in 1:1]
    Z2 = [z for x in 0.0:4.0, z in -3.0:0.0, _ in 1:1]
    d2 = CartData(X2, zero(X2), Z2, (Z = Z2,))
    @test CreateGrid(d2).N == (5, 4)
    @test CreateGrid(d2).min == (0.0, -3.0e3)
end

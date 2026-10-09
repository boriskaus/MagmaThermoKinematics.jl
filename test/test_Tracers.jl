# Tests for grid -> tracer interpolation and tracer -> grid phase ratios
using MagmaThermoKinematics
using Test

# Exact multilinear interpolation of `F` (sampled on the 1D coordinates `xs`) at point `p`
function multilinear_reference(xs, F, p)
    p = Tuple(p)
    i = map((x, q) -> clamp(searchsortedlast(x, q), 1, length(x) - 1), xs, p)
    t = map((x, q, j) -> (q - x[j]) / (x[j + 1] - x[j]), xs, p, i)
    val = 0.0
    for off in Iterators.product(ntuple(_ -> 0:1, length(xs))...)
        w = prod(o == 1 ? tk : 1 - tk for (o, tk) in zip(off, t))
        val += w * F[CartesianIndex(i .+ off)]
    end
    return val
end

@testset "Tracers" begin
    cases = (
        (CreateGrid(size = (7, 9), x = (0.0, 1.0), z = (-2.0, 0.5)), (x, z) -> sin(3x) * cos(2z)),
        (CreateGrid(size = (7, 6, 8), x = (0.0, 1.0), y = (1.0, 2.0), z = (-2.0, 0.5)), (x, y, z) -> sin(3x) * cos(2y) * exp(z)),
    )
    for (Grid, f) in cases
        dim = length(Grid.N)
        xs = Grid.coord1D
        F = [f(ntuple(d -> xs[d][I[d]], dim)...) for I in CartesianIndices(Grid.N)]
        G = 2 .* F .+ 1
        # first cell, last cell, on min and max boundary, interior
        pts = [
            [Grid.min[d] + 0.5 * Grid.Δ[d] for d in 1:dim],
            [Grid.max[d] - 0.5 * Grid.Δ[d] for d in 1:dim],
            collect(Grid.min),
            collect(Grid.max),
            [Grid.min[d] + 0.37 * Grid.L[d] for d in 1:dim],
        ]
        for p in pts
            Tr = StructArray([Tracer{Float64}(coord = copy(p))])
            ref = multilinear_reference(xs, F, p)
            UpdateTracers_T_ϕ!(Tr, xs, F, G)
            @test Tr.T[1] ≈ ref atol = 1.0e-12
            @test Tr.Phi[1] ≈ multilinear_reference(xs, G, p) atol = 1.0e-12
            Tr = StructArray([Tracer{Float64}(coord = copy(p))])
            UpdateTracers_Field!(Tr, Grid, F, :T)
            @test Tr.T[1] ≈ ref atol = 1.0e-12
            UpdateTracers_Field!(Tr, Grid, G, :Phi)
            @test Tr.Phi[1] ≈ multilinear_reference(xs, G, p) atol = 1.0e-12
        end
    end

    @testset "PhaseRatioFromTracers! DistanceWeighted, tracers on a node" begin
        Grid = CreateGrid(size = (5, 5), x = (0.0, 1.0), z = (0.0, 1.0))
        node = [Grid.coord1D[1][3], Grid.coord1D[2][3]]
        Tr = StructArray([Tracer{Float64}(coord = copy(node), Phase = 1), Tracer{Float64}(coord = copy(node), Phase = 2)])
        PhaseRatio = zeros(Grid.N..., 2)
        PhaseRatioFromTracers!(PhaseRatio, Grid, Tr; InterpolationMethod = "DistanceWeighted")
        @test PhaseRatio[3, 3, :] ≈ [0.5, 0.5]
    end
end

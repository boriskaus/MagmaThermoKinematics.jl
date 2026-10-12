# Tests for grid -> tracer interpolation and tracer -> grid phase ratios
using MagmaThermoKinematics
using MagmaThermoKinematics: JLD2
using ZirconGrowth
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

    @testset "PhaseRatioFromTracers! 1D, background phase, out-of-grid tracer" begin
        Grid = CreateGrid(size = 5, extent = 1.0)
        Tr = StructArray([Tracer{Float64}(coord = [0.5], Phase = 2), Tracer{Float64}(coord = [1.5], Phase = 2)])
        PhaseRatio = zeros(5, 2)
        PhaseRatioFromTracers!(PhaseRatio, Grid, Tr; BackgroundPhase = 1)
        @test PhaseRatio[:, 2] == [0, 0, 1, 0, 1]        # the tracer past x = 1 counts at the last node
        @test PhaseRatio[:, 1] == [1, 1, 0, 1, 0]

        @test_throws "Size of PhaseRatio array inconsistent with input grid" PhaseRatioFromTracers!(zeros(4, 2), Grid, Tr)
        @test_throws "Size of last dimension of PhaseRatio is too small" PhaseRatioFromTracers!(zeros(5, 1), Grid, Tr)
        G = Grid
        Grid_var = typeof(G)(false, G.N, G.Δ, G.L, G.min, G.max, G.coord1D, G.coord1D_cen)
        @test_throws "only works for constant spacing" PhaseRatioFromTracers!(zeros(5, 2), Grid_var, Tr)
    end

    @testset "zircon ages and growth from Tt-paths" begin
        t = collect(0.0:0.002:0.1)              # Myr
        cooling(Tstart) = Tracer{Float64}(coord = [0.0, 0.0], time_vec = copy(t), T_vec = collect(range(Tstart, 750.0, length(t))))
        Tr = StructArray([cooling(1000.0), cooling(950.0), cooling(900.0)])
        dir = mktempdir()
        JLD2.jldsave(joinpath(dir, "Tracers_SimParams.jld2"); Tracers = Tr, Tav_magma_Time = [1.0], Time_vec = t)

        redirect_stdout(devnull) do
            Process_ZirconAges(dir)
        end
        d = JLD2.load(joinpath(dir, "ZirconAges.jld2"))
        @test issubset(["Age_Ma", "cum_PDF", "norm_PDF", "T_av_time", "T_average_magma_time", "number_zircons"], keys(d))
        @test issorted(d["cum_PDF"], rev = true)
        @test extrema(d["cum_PDF"]) == (0.0, 1.0)

        # a single-step tracer and one that never cools below zircon saturation are skipped
        hot = Tracer{Float64}(coord = [0.0, 0.0], time_vec = copy(t), T_vec = fill(1100.0, length(t)))
        short = Tracer{Float64}(coord = [0.0, 0.0], time_vec = [0.0], T_vec = [1000.0])
        r = redirect_stdout(devnull) do
            simulate_zircon_growth_from_tracers(StructArray([Tr..., hot, short]); nx = 20, return_results = true)
        end
        @test length(r.age_years) == length(r.zircon_radius_um) == length(r.results) == 3
        @test all(0 .< r.age_years .< 1.0e5)               # within the 0.1 Myr Tt-path
        @test issorted(r.zircon_radius_um, rev = true)     # longer above the solidus, larger crystal
        @test volume_averaged_age(r.results) == r.age_years

        r1, rdir = redirect_stdout(devnull) do
            simulate_zircon_growth_from_tracers(Tr[1]; nx = 20), simulate_zircon_growth_from_tracers(dir; nx = 20)
        end
        @test r1.age_years == r.age_years[1:1]
        @test rdir.age_years == r.age_years
        @test JLD2.load(joinpath(dir, "ZirconGrowth.jld2"), "age_years") == r.age_years
    end
end

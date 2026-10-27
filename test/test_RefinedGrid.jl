# Refined (non-uniform) grids: coordinates, the diffusion, advection and tracer routines, the eruption model and
# MTK_GeoParams on a grid whose spacing varies along every direction (a tensor product of 1D coordinate vectors).
using Test, Random
using MagmaThermoKinematics
using MagmaThermoKinematics: chamber_geometry, erupt_sources, revolved_sources, EruptionParams, EruptionState, StructArray, Tracer
using GeophysicalModelGenerator: CartData, xyz_grid
using JLArrays
JLArrays.allowscalar(false)
MagmaThermoKinematics.KernelAbstractions.synchronize(::JLArrays.JLBackend) = nothing

const κ_RG = 3.0 / (2700 * 1000)
const Mat_RG = (SetMaterialParams(Phase = 1, Density = ConstantDensity(ρ = 2700), HeatCapacity = ConstantHeatCapacity(Cp = 1000), Conductivity = ConstantConductivity(k = 3), LatentHeat = ConstantLatentHeat(Q_L = 0.0), Melting = MeltingParam_Caricchi()),)

# the same coordinates as a "refined" grid (vectors), to compare the refined code paths with the constant-spacing ones
as_vectors(G) = GridData{Float64, length(G.N), Vector{Float64}, Vector{Float64}}(false, G.N, G.Δ, G.L, G.min, G.max, map(collect, G.coord1D), map(collect, G.coord1D_cen))

function rg_arrays(dims; backend = CPU())
    names = (
        T = 0, T_K = 0, Tnew = 0, T_it_old = 0, Tupdate = 0, Kc = 1, Rho = 1, Cp = 1, Hr = 0, Hq = 0, Hl = 0, ϕ = 0, dϕdT = 0, R = 0, Y = 0, Z = 0, P = 0,
        pt_C = 0, pt_b = 0, pt_phiold = 0, pt_res = 0, pt_Tk = 0, pt_Tkm1 = 0, pt_Tn = 0, Qrelax = 0, Qenth = 0,
    )
    return CreateArrays(Dict(dims => names); backend)
end
function rg_coordinates!(A, Grid)
    return length(Grid.N) == 2 ? GridArray!(A.R, A.Z, Grid) : GridArray!(A.R, A.Y, A.Z, Grid)
end
rg_num(solver = :explicit; axisymmetric = false) = (; (k => getfield(Numeric_params(axisymmetric = axisymmetric, convergence = 1.0e-10), k) for k in fieldnames(Numeric_params))..., solver, apt_tol = 1.0e-10, apt_max_iter = 200_000, apt_check = 10)

# Gaussian anomaly (100 K, width s) diffusing along `d` directions, t after the start
rg_gauss(r2, s, t, d) = 100 * (s^2 / (s^2 + 2κ_RG * t))^(d / 2) * exp(-r2 / (2 * (s^2 + 2κ_RG * t)))

# Decay of a Gaussian anomaly centred at `c`: maximum error against the analytic solution and the relative change of the heat content
function rg_decay(Grid; axisymmetric = false, solver = :explicit, fac = 0.4, tend = 20.0e3 * SecYear, s = 1.0e3, c)
    A = rg_arrays(Grid.N); rg_coordinates!(A, Grid)
    r2 = (A.R .- c[1]) .^ 2 .+ (A.Z .- c[2]) .^ 2
    d = axisymmetric ? 3 : 2
    A.T .= 400 .+ rg_gauss.(r2, s, 0.0, d)
    w = map(c -> (cw = cell_widths(c); cw isa Number ? fill(cw, length(c)) : cw), Grid.coord1D)
    W = w[1] .* w[2]' .* (axisymmetric ? A.R : 1.0)
    E0 = sum((A.T .- 400) .* W)
    dt = fac * minimum(Grid.Δ)^2 / κ_RG / 4
    nt = ceil(Int, tend / dt); dt = tend / nt
    Phases = ones(Int, Grid.N)
    for _ in 1:nt
        Nonlinear_Diffusion_step!(A, Mat_RG, Phases, Grid, dt, rg_num(solver; axisymmetric))
        A.T .= A.Tnew
    end
    return maximum(abs, A.T .- 400 .- rg_gauss.(r2, s, tend, d)), sum((A.T .- 400) .* W) / E0 - 1
end

@testset "Refined grids" begin
    @testset "coordinates" begin
        x = refined_coordinates(0.0, 20.0e3, 201; zone = (5.0e3, 10.0e3), ratio = 4, max_growth = 1.1)
        h = diff(x)
        @test length(x) == 201 && x[1] == 0 && x[end] == 20.0e3 && all(>(0), h)
        inz = [5.0e3 < x[i] && x[i + 1] < 10.0e3 for i in eachindex(h)]
        @test maximum(h[inz]) ≈ minimum(h[inz]) rtol = 1.0e-6                     # constant spacing in the zone
        @test minimum(h) < 0.7 * 20.0e3 / 200                                     # finer than the constant spacing with the same number of points
        @test maximum(h) / minimum(h) ≈ 4 rtol = 1.0e-3
        @test maximum(max.(h[2:end] ./ h[1:(end - 1)], h[1:(end - 1)] ./ h[2:end])) < 1.11      # smooth transition
        xΔ = refined_coordinates(0.0, 20.0e3, 201; zone = (5.0e3, 10.0e3), Δ = 70.0, max_growth = 1.1)
        @test minimum(diff(xΔ)) ≈ 70.0 rtol = 1.0e-3
        xt = refined_coordinates(0.0, 20.0e3, 201; zone = (5.0e3, 10.0e3), ratio = 4, transition = 2.0e3)
        @test maximum(diff(xt)) / minimum(diff(xt)) ≈ 4 rtol = 1.0e-3
        @test refined_coordinates(0.0, 1.0, 11; zone = (0.2, 0.4), ratio = 1) ≈ collect(0:0.1:1)
        # abrupt change of the spacing: the zone edges are grid points, constant spacing inside and outside
        xa = refined_coordinates(0.0, 20.0e3, 61; zone = (7.0e3, 13.0e3), ratio = 3, transition = 0)
        ha = diff(xa)
        @test length(xa) == 61 && xa[1] == 0 && xa[end] == 20.0e3 && 7.0e3 in xa && 13.0e3 in xa
        @test all(≈(ha[findfirst(==(7.0e3), xa)]), ha[(7.0e3 .<= xa[1:(end - 1)]) .& (xa[2:end] .<= 13.0e3)])
        @test maximum(ha) / minimum(ha) ≈ 3 rtol = 0.1
        @test minimum(ha) < minimum(diff(refined_coordinates(0.0, 20.0e3, 61; zone = (7.0e3, 13.0e3), ratio = 3)))     # no points spent on a transition
        @test unique(round.(diff(refined_coordinates(0.0, 20.0e3, 61; zone = (7.0e3, 13.0e3), Δ = 150.0, transition = 0)); digits = 6)) == [700.0, 150.0]
        xe = refined_coordinates(0.0, 10.0, 21; zone = (0.0, 4.0), ratio = 2, transition = 0)       # zone at the boundary
        @test xe[1] == 0 && 4.0 in xe && length(xe) == 21
        @test_throws "cannot be reached" refined_coordinates(0.0, 20.0e3, 61; zone = (7.0e3, 13.0e3), Δ = 50.0, transition = 0)
        @test_throws ArgumentError refined_coordinates(0.0, 1.0, 11; zone = (0.5, 2.0))
        @test_throws "cannot be reached" refined_coordinates(0.0, 20.0e3, 201; zone = (5.0e3, 10.0e3), Δ = 5.0)

        G = CreateGrid(size = (61, 41), extent = (20.0e3, 10.0e3), refine = (x = (zone = (0.0, 4.0e3), ratio = 3, max_growth = 1.1),))
        @test !G.ConstantΔ && G.coord1D[1] isa Vector{Float64} && G.coord1D[2] isa Vector{Float64}
        @test G.N == (61, 41) && G.min == (0.0, -10.0e3) && G.max == (20.0e3, 0.0)
        @test G.Δ[1] ≈ minimum(diff(G.coord1D[1])) && G.Δ[2] ≈ 250.0
        @test G.coord1D_cen[1] ≈ (G.coord1D[1][1:(end - 1)] .+ G.coord1D[1][2:end]) ./ 2
        @test occursin("refined grid", sprint(show, G))
        @test_throws ArgumentError CreateGrid(size = (11, 11), extent = (1.0, 1.0), refine = (y = (zone = (0.0, 0.5),),))
        @test CreateGrid(coords = (collect(0.0:0.5:5.0), collect(-2.0:0.25:0.0))).ConstantΔ        # uniform vectors give a grid with constant spacing
        G3 = CreateGrid(size = (9, 9, 21), x = (-1.0e3, 1.0e3), y = (-1.0e3, 1.0e3), z = (-4.0e3, 0.0), refine = (z = (zone = (-2.5e3, -1.5e3), ratio = 2, max_growth = 1.2),))
        @test !G3.ConstantΔ && G3.coord1D[1] ≈ collect(range(-1.0e3, 1.0e3, length = 9))
        # from a CartData set on refined coordinates (km)
        xr, zr = refined_coordinates(0.0, 10.0, 21; zone = (2.0, 4.0), ratio = 3, max_growth = 1.3), refined_coordinates(-5.0, 0.0, 11; zone = (-3.0, -2.0), ratio = 2, max_growth = 1.3)
        X, Y, Z = xyz_grid(xr, [0.0, 1.0], zr)
        Gc = CreateGrid(CartData(X, Y, Z, (Z = Z,)))
        @test !Gc.ConstantΔ && Gc.coord1D[1] ≈ 1.0e3 .* xr && Gc.coord1D[3] ≈ 1.0e3 .* zr

        @test node_spacing(0.0:0.5:2.0) == 0.5 && node_spacing([0.0, 1.0, 3.0]) == [1.0, 2.0]
        @test cell_widths(0.0:0.5:2.0) == 0.5 && cell_widths([0.0, 1.0, 3.0, 4.0]) == [1.0, 1.5, 1.5, 1.0]
        @test min_spacing([0.0, 1.0, 3.0]) == 1.0
    end

    @testset "diffusion: the refined stencils with constant spacing" begin
        for (N, axisymmetric) in (((17, 13), false), ((17, 13), true), ((7, 8, 9), false)), solver in (:explicit, :implicit_enthalpy)
            G = CreateGrid(size = N, extent = ntuple(i -> 5.0e3 * i, length(N)))
            T0 = 400 .+ 300 .* rand(N...)
            out = map((G, as_vectors(G))) do Grid
                A = rg_arrays(N); rg_coordinates!(A, Grid); A.T .= T0
                Nonlinear_Diffusion_step!(A, Mat_RG, ones(Int, N), Grid, 2.0e11, rg_num(solver; axisymmetric))
                copy(A.Tnew)
            end
            @test out[2] ≈ out[1] rtol = 1.0e-12
        end
    end

    @testset "diffusion: exact for quadratic temperature fields" begin
        # the three-point flux stencil gives ∂²(x²)/∂x² = 2 for any spacing
        for N in ((15, 12), (9, 8, 10))
            G = CreateGrid(coords = ntuple(d -> cumsum(vcat(0.0, 100.0 .* (1 .+ 0.5 .* rand(N[d] - 1)))), length(N)))
            A = rg_arrays(N); rg_coordinates!(A, G)
            T = length(N) == 2 ? A.R .^ 2 .+ A.Z .^ 2 : A.R .^ 2 .+ A.Y .^ 2 .+ A.Z .^ 2
            Tnew = similar(T); K = 3.0; ρcp = 2.7e6; dt = 1.0e8
            diffusion_step!(Tnew, T, fill(K, N), fill(2700.0, N), fill(1000.0, N), zeros(N), zeros(N), dt, MagmaThermoKinematics._kernel_spacing(G, T), zeros(N))
            inner = CartesianIndices(ntuple(d -> 2:(N[d] - 1), length(N)))
            @test (Tnew .- T)[inner] ≈ fill(dt * K * 2 * length(N) / ρcp, size(inner)) rtol = 1.0e-9
        end
    end

    @testset "diffusion: implicit step agrees with a direct solve" begin
        x = cumsum(vcat(0.0, 100.0 .* (1 .+ rand(11)))); z = cumsum(vcat(0.0, 150.0 .* (1 .+ rand(13))))
        G = CreateGrid(coords = (x, z)); N = G.N
        A = rg_arrays(N); rg_coordinates!(A, G)
        A.T .= [300.0 + 40.0 * sin(i / 3) + 25.0 * j for i in 1:N[1], j in 1:N[2]]; A.T_it_old .= A.T
        A.Kc .= [1.5 + 3.0 * (j > 6) for i in 1:N[1], j in 1:N[2]]; A.Rho .= 2700.0; A.Cp .= 1000.0; A.Hr .= 1.0e-6
        dt = 1.0e12; Δ = MagmaThermoKinematics._kernel_spacing(G, A.T)
        apt_diffusion_solve!(A, dt, Δ, nothing, NumParam(Nx = N[1], Nz = N[2], solver = :implicit_enthalpy, apt_tol = 1.0e-11, apt_check = 5))
        # dense backward-Euler system with the same discretisation: fixed T top and bottom, zero flux on the sides
        li = LinearIndices(N); M = zeros(prod(N), prod(N)); b = zeros(prod(N)); h = (diff(x), diff(z))
        for I in CartesianIndices(N)
            i, j = Tuple(I); r = li[I]
            if j == 1 || j == N[2]
                M[r, r] = 1; b[r] = A.T[I]
            elseif i == 1 || i == N[1]
                M[r, r] = 1; M[r, li[i == 1 ? 2 : N[1] - 1, j]] = -1
            else
                M[r, r] = A.Rho[I] * A.Cp[I] / dt; b[r] = A.Rho[I] * A.Cp[I] * A.T[I] / dt + A.Hr[I]
                for d in 1:2, s in (-1, 1)
                    J = I + s * CartesianIndex(ntuple(k -> Int(k == d), 2)); k = I[d]
                    c = (A.Kc[I] + A.Kc[J]) / 2 / (s == 1 ? h[d][k] : h[d][k - 1]) / ((h[d][k - 1] + h[d][k]) / 2)
                    M[r, r] += c; M[r, li[J]] -= c
                end
            end
        end
        @test A.Tnew ≈ reshape(M \ b, N) rtol = 1.0e-8
    end

    @testset "diffusion: decay of a Gaussian anomaly" begin
        for (axisymmetric, xzone) in ((false, (7.0e3, 13.0e3)), (true, (0.0, 3.0e3)))
            c = (axisymmetric ? 0.0 : 10.0e3, -10.0e3)
            Gu = CreateGrid(size = (61, 61), x = (0.0, 20.0e3), z = (-20.0e3, 0.0))
            Gr = CreateGrid(size = (61, 61), x = (0.0, 20.0e3), z = (-20.0e3, 0.0), refine = (x = (zone = xzone, ratio = 3, max_growth = 1.1), z = (zone = (-13.0e3, -7.0e3), ratio = 3, max_growth = 1.1)))
            eu, _ = rg_decay(Gu; axisymmetric, c)
            er, Er = rg_decay(Gr; axisymmetric, c)
            @test er < 0.5 * eu                                    # the same number of points resolves the anomaly better
            @test abs(Er) < 1.0e-4                                 # the scheme is conservative
            eiu, _ = rg_decay(Gu; axisymmetric, c, solver = :implicit_enthalpy, fac = 2.0)
            ei, Ei = rg_decay(Gr; axisymmetric, c, solver = :implicit_enthalpy, fac = 2.0)
            @test ei < eiu
            @test abs(Ei) < 1.0e-4
            Ga = CreateGrid(size = (61, 61), x = (0.0, 20.0e3), z = (-20.0e3, 0.0), refine = (x = (zone = xzone, ratio = 3, transition = 0), z = (zone = (-13.0e3, -7.0e3), ratio = 3, transition = 0)))
            ea, Ea = rg_decay(Ga; axisymmetric, c)                 # abrupt change of the spacing: conservative across the jump
            @test abs(Ea) < 1.0e-4
            axisymmetric && @test ea < 0.5 * eu
        end
    end

    @testset "lithostatic pressure and bottom flux" begin
        z = refined_coordinates(-10.0e3, 0.0, 41; zone = (-6.0e3, -3.0e3), ratio = 3, max_growth = 1.2)
        Rho = fill(2700.0, 5, 41); P = zero(Rho)
        lithostatic_pressure!(P, Rho, 9.81, diff(z))
        @test P ≈ repeat((2700 * 9.81 .* (0.0 .- z))', 5) rtol = 1.0e-12
        Pjl = JLArray(zero(Rho)); lithostatic_pressure!(Pjl, JLArray(Rho), 9.81, diff(z))
        @test Array(Pjl) ≈ P rtol = 1.0e-14
        @test_throws DimensionMismatch lithostatic_pressure!(P, Rho, 9.81, diff(z)[2:end])
    end

    @testset "advection" begin
        x = refined_coordinates(0.0, 10.0e3, 51; zone = (3.0e3, 6.0e3), ratio = 3, max_growth = 1.2)
        z = refined_coordinates(-8.0e3, 0.0, 41; zone = (-5.0e3, -3.0e3), ratio = 2, max_growth = 1.2)
        T = [0.3 * xx - 0.2 * zz for xx in x, zz in z]
        U = (fill(120.0, size(T)), fill(-80.0, size(T)))
        Tn = similar(T)
        AdvectTemperature!(Tn, T, (x, z), U, 1.0)
        inner = [i > 3 && i < length(x) - 3 && j > 3 && j < length(z) - 3 for i in eachindex(x), j in eachindex(z)]
        @test Tn[inner] ≈ (T .- 0.3 * 120.0 .- 0.2 * 80.0)[inner] rtol = 1.0e-12       # a linear field is advected exactly
        Tjl = JLArray(zero(T)); AdvectTemperature!(Tjl, JLArray(T), (x, z), map(JLArray, U), 1.0)
        @test Array(Tjl) ≈ Tn rtol = 1.0e-14
        # refined code path with constant spacing = the constant-spacing path
        xu, zu = range(0.0, 10.0e3, length = 51), range(-8.0e3, 0.0, length = 41)
        Tu = [sin(xx / 2.0e3) * cos(zz / 1.5e3) for xx in xu, zz in zu]; Uu = ([300 * sin(zz / 1.0e3) for xx in xu, zz in zu], [-200 * cos(xx / 3.0e3) for xx in xu, zz in zu])
        a, b = similar(Tu), similar(Tu)
        AdvectTemperature!(a, Tu, (xu, zu), Uu, 1.0); AdvectTemperature!(b, Tu, (collect(xu), collect(zu)), Uu, 1.0)
        @test b ≈ a rtol = 1.0e-12
        # tracers: linear velocity interpolation on the coordinate vectors
        Tr = StructArray([Tracer{Float64}(coord = [4.0e3, -4.0e3]), Tracer{Float64}(coord = [5.5e3, -3.5e3])])
        AdvectTracers!(Tr, (x, z), U, 1.0)
        @test Tr.coord[1] ≈ [4.12e3, -4.08e3] && Tr.coord[2] ≈ [5.62e3, -3.58e3]
        @test_throws "needs a grid with constant spacing" Interpolate!((zeros(1),), (x, z), (T,), ([1.0], [-1.0]), "Cubic")
    end

    @testset "tracers" begin
        G = CreateGrid(size = (31, 21), x = (0.0, 6.0e3), z = (-4.0e3, 0.0), refine = (x = (zone = (2.0e3, 3.0e3), ratio = 3, max_growth = 1.3), z = (zone = (-3.0e3, -2.0e3), ratio = 2, max_growth = 1.3)))
        F = [2.0 * xx - 3.0 * zz + 1.0e-4 * xx * zz for xx in G.coord1D[1], zz in G.coord1D[2]]
        pts = [[1234.5, -2345.6], [2500.0, -2500.0], [5999.0, -10.0], [7000.0, 100.0]]
        Tr = StructArray([Tracer{Float64}(coord = copy(p)) for p in pts])
        UpdateTracers_Field!(Tr, G, F, :T)
        f(p) = 2.0 * p[1] - 3.0 * p[2] + 1.0e-4 * p[1] * p[2]
        @test Tr.T[1:3] ≈ f.(pts[1:3]) rtol = 1.0e-12                        # bilinear fields are reproduced exactly
        @test Tr.T[4] ≈ f([6.0e3, 0.0]) && Tr.coord[4] == [6.0e3, 0.0]        # clamped to the grid
        Tr2 = StructArray([Tracer{Float64}(coord = copy(p)) for p in pts])
        UpdateTracers_T_ϕ!(Tr2, G.coord1D, F, 2 .* F)
        @test Tr2.T ≈ Tr.T && Tr2.Phi ≈ 2 .* Tr.T
        # phases at the nearest point (physical distance) and the distance weights
        xg, zg = G.coord1D
        i, j = 12, 9; xm = (xg[i] + xg[i + 1]) / 2
        Tp = StructArray([Tracer{Float64}(coord = [xm - 1.0e-3 * (xg[i + 1] - xg[i]), zg[j]], Phase = 2), Tracer{Float64}(coord = [xg[i + 1] + 0.25 * (xg[i + 2] - xg[i + 1]), zg[j]], Phase = 3)])
        PR = zeros(G.N..., 3)
        PhaseRatioFromTracers!(PR, G, Tp; InterpolationMethod = "Constant")
        @test PR[i, j, 2] == 1 && PR[i + 1, j, 3] == 1
        PRw = zeros(G.N..., 3)
        PhaseRatioFromTracers!(PRw, G, Tp; InterpolationMethod = "DistanceWeighted", BackgroundPhase = 1)
        @test PRw[i + 1, j, 3] == 1 && PRw[i, j, 2] == 1
        Ph = zeros(Int, G.N); PhasesFromTracers!(Ph, G, Tp; BackgroundPhase = 1)
        @test Ph[i, j] == 2 && Ph[i + 1, j] == 3 && Ph[1, 1] == 1
    end

    @testset "eruptions" begin
        # a 50 m zone around the chamber, coarser outside: the chamber cells have the same positions and sizes as on a uniform 50 m grid
        dx = 50.0
        xf = collect(-1500.0:dx:1500.0); zf = collect(-5500.0:dx:-3500.0)
        x = vcat(collect(-6000.0:200.0:-1700.0), -1600.0, xf, 1600.0, collect(1700.0:200.0:6000.0))
        z = vcat(collect(-9000.0:200.0:-5700.0), -5600.0, zf, -3400.0, collect(-3300.0:100.0:500.0))
        xu = -6000.0:dx:6000.0; zu = -9000.0:dx:500.0
        incham(xx, zz) = abs(xx) <= 10 * dx && -5000.0 <= zz <= -4000.0
        ep = EruptionParams(m_w = 0.06, ΔP_crit = 3.0e6, σ_barrier = 0.0, w_dike = 20.0, melt_viscosity = LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.9606K, η0 = 1Pas), solubility = Liu2005_Solubility())
        res = map(((x, z), (collect(xu), collect(zu)), (xu, zu))) do (x, z)
            ϕ = [incham(xx, zz) ? 0.8 : 0.0 for xx in x, zz in z]; T = [incham(xx, zz) ? 900.0 : 300.0 for xx in x, zz in z]
            st = EruptionState()
            chamber_step!(st, ep, ϕ, T, (x, z); recharge = 0.0, Δt = 3.15576e7, time = 0.0, T_background = d -> 20.0 + 0.02d)
            cells = findall(>(0), ϕ)
            geo = chamber_geometry(cells, map(cell_widths, (x, z)), (x, z)); event = (; cells, f = 0.1, geo)
            _, ΔV, _ = erupt_sources(event, (x, z)); _, ΔVr, xc = revolved_sources(event, (x, z))
            Tr = StructArray([Tracer{Float64}(num = 1, coord = [0.0, -4500.0], Phase = 1)])
            U, _ = erupt!(copy(T), similar(T), Tr, (x, z), event; revolve = true, rng = MersenneTwister(1), softening = dx)
            js = findfirst(==(0.0), collect(z)); is = [findfirst(==(xx), collect(x)) for xx in (-1000.0, 0.0, 400.0)]
            (; geo, P_lith = st.P_lith, V = sum(ΔV), Vr = sum(ΔVr), xc, Us = U[2][is, js])
        end
        ru = res[3]
        for r in res[1:2]                                                         # refined grid, and vectors with constant spacing
            for k in (:V, :W, :h, :R, :zc, :z_lo, :z_hi)
                @test getfield(r.geo, k) ≈ getfield(ru.geo, k) rtol = 1.0e-12
            end
            @test r.P_lith ≈ ru.P_lith rtol = 1.0e-12
            @test r.V ≈ ru.V rtol = 1.0e-12
            @test r.Vr ≈ ru.Vr rtol = 1.0e-12
            @test abs(r.xc - ru.xc) < 1.0e-9
            @test r.Us ≈ ru.Us rtol = 1.0e-10                                     # surface displacement of the same sources
        end
        # free surface: initialised half way to the point below the lowest air point
        Ph = [zz > 0 ? 0 : 1 for xx in x, zz in z]; Zs = fill(NaN, length(x))
        move_free_surface!(Ph, copy(Ph), zeros(size(Ph)), Zs, (x, z), (zeros(size(Ph)), fill(30.0, size(Ph))))
        @test all(Zs .≈ 50.0 + 30.0)
    end

    @testset "eruptions: 3D chamber, coarsened sources, surface" begin
        using MagmaThermoKinematics: cell_sources, _surface_above
        # 3D: refined coordinates that contain the uniform 100 m points around the chamber
        xf = collect(-1000.0:100.0:1000.0); zf = collect(-3000.0:100.0:-1000.0)
        x = vcat(-3000.0, -2000.0, -1500.0, xf, 1500.0, 2000.0, 3000.0); z = vcat(-6000.0, -4500.0, -3500.0, zf, -500.0, 0.0)
        xu = -3000.0:100.0:3000.0; zu = -6000.0:100.0:0.0
        cham(xx, yy, zz) = xx^2 + yy^2 <= 600.0^2 && -2500.0 <= zz <= -1500.0
        geos = map(((x, x, z), (xu, xu, zu))) do c
            cells = [I for I in CartesianIndices(map(length, c)) if cham(c[1][I[1]], c[2][I[2]], c[3][I[3]])]
            chamber_geometry(cells, map(cell_widths, c), c)
        end
        for k in (:V, :W, :h, :R, :zc)
            @test getfield(geos[1], k) ≈ getfield(geos[2], k) rtol = 1.0e-12
        end
        # coarsened sources with the cell volumes: the same total volume and centroid as the uniform sources
        c2 = (x, z); cu = (xu, zu)
        cells2(c) = [I for I in CartesianIndices(map(length, c)) if abs(c[1][I[1]]) <= 600.0 && -2500.0 <= c[2][I[2]] <= -1500.0]
        for coarsen in (1, 3)
            cen, ΔV = cell_sources(cells2(c2), I -> prod(ntuple(k -> cell_widths(c2[k])[I[k]], 2)), -0.1, c2; coarsen)
            cenu, ΔVu = cell_sources(cells2(cu), 100.0^2, -0.1, cu; coarsen)
            @test sum(ΔV) ≈ sum(ΔVu) rtol = 1.0e-12
            @test sum(cen .* ΔV; dims = 1) ≈ sum(cenu .* ΔVu; dims = 1) rtol = 1.0e-12
        end
        ev = (; cells = cells2(c2), f = 0.1, geo = chamber_geometry(cells2(c2), map(cell_widths, c2), c2))
        evu = (; cells = cells2(cu), f = 0.1, geo = chamber_geometry(cells2(cu), map(cell_widths, cu), cu))
        @test sum(erupt_sources(ev, c2; coarsen = 3)[2]) ≈ sum(erupt_sources(evu, cu; coarsen = 3)[2]) rtol = 1.0e-12
        @test _surface_above(fill(NaN, length(x)), cells2(c2), z) ≈ z[end] + (z[end] - z[end - 1]) / 2
    end

    @testset "out-of-plane loss and convection on a refined grid" begin
        using MagmaThermoKinematics: _thickness, _cw
        # the width of a Gaussian anomaly from the moments weighted with the cell widths
        x = refined_coordinates(-20.0e3, 20.0e3, 201; zone = (-3.0e3, 3.0e3), ratio = 4, transition = 0); z = collect(range(-10.0e3, 0.0, length = 11))
        s = 2.0e3; θ = [exp(-xx^2 / (2s^2)) for xx in x, zz in z]; X = [xx for xx in x, zz in z]
        L2w = out_of_plane_width2(θ, X; smooth = 0, Lmin = 1.0, dx = cell_widths(x), dz = fill(1.0e3, length(z)))
        L2u = out_of_plane_width2(θ, X; smooth = 0, Lmin = 1.0)
        @test maximum(abs, L2w ./ s^2 .- 1) < 1.0e-2                                         # the weighted moments give the variance
        @test maximum(abs, L2u ./ s^2 .- 1) > 0.05                                            # unweighted moments are biased by the concentrated points
        Hr = zeros(size(θ)); L2h = add_out_of_plane_loss!(Hr, 400 .+ 100 .* θ, fill(400.0, size(θ)), fill(3.0, size(θ)), X, ones(Int, size(θ)); smooth = 0, Lmin = 1.0, dx = cell_widths(x), dz = fill(1.0e3, length(z)))
        @test L2h ≈ L2w && minimum(Hr) < 0
        # thickness of a convecting layer: the sum of the control-volume widths
        h = diff(collect(z) .+ [0.0, 50.0, -30.0, 20.0, 0.0, 10.0, -10.0, 40.0, 0.0, -20.0, 0.0])
        @test _thickness(h, 3, 6, 11) ≈ sum(_cw(h, k, 11) for k in 3:6) && _thickness(1.0e3, 3, 6, 11) == 4.0e3
        @test _cw(h, 1, 11) == h[1] && _cw(h, 11, 11) == h[end] && _cw(h, 4, 11) == (h[3] + h[4]) / 2
        # Nusselt numbers on a refined grid with constant spacing = those of the constant-spacing grid
        N = (6, 40); zc = collect(range(-12.0e3, 0.0, length = 40)); Z = repeat(zc', 6)
        T = [zz < -6.0e3 && zz > -10.0e3 ? 1250.0 + 0.05 * (-6.0e3 - zz) : 900.0 for _ in 1:6, zz in zc]; ϕ = [t > 1200 ? 0.8 : 0.0 for t in T]     # K
        Kc, Rho, Cp = fill(3.0, N), fill(2700.0, N), fill(1000.0, N); Ph = ones(Int, N)
        Nu1, Nu2 = ones(N), ones(N)
        compute_convective_nusselt!(Nu1, Kc, T, ϕ, Rho, Cp, Ph, Z, step(range(-12.0e3, 0.0, length = 40)))
        compute_convective_nusselt!(Nu2, Kc, T, ϕ, Rho, Cp, Ph, Z, diff(zc))
        @test maximum(Nu1) > 1
        @test Nu2 ≈ Nu1 rtol = 1.0e-10
    end

    @testset "MTK_GeoParams on a refined grid" begin
        function run_model(backend, dim; solver = :explicit, refine = nothing)
            if dim == 2
                refine === nothing && (refine = (x = (zone = (0.0, 6.0e3), ratio = 2, max_growth = 1.2), z = (zone = (-10.0e3, -5.0e3), ratio = 2, max_growth = 1.2)))
                Num = NumParam(; Nx = 33, Nz = 33, SimName = mktempdir(), maxTime_Myrs = 0.003, fac_dt = solver === :explicit ? 0.2 : 2.0, ω = 0.5, Geotherm = 30 / 1.0e3, Output_VTK = false, backend, solver, refine)
                sill = CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 6.0e3 * m, H = 300.0 * m)
            else
                refine === nothing && (refine = (z = (zone = (-9.0e3, -5.0e3), ratio = 2, max_growth = 1.3),))
                Num = NumParam(; Nx = 13, Ny = 13, Nz = 17, W = 20.0e3, L = 20.0e3, H = 20.0e3, SimName = mktempdir(), maxTime_Myrs = 0.02, fac_dt = 0.2, ω = 0.5, Geotherm = 30 / 1.0e3, Output_VTK = false, backend, refine)
                sill = EllipticalIntrusion(Center = Point3(0.0, 0.0, -7000.0) * m, Angle = Vec2(0.0, 0.0) * NoUnits, W = 5.0e3 * m, H = 800.0 * m)
            end
            Dikes = SillParams(sill = sill, InjectionInterval_year = dim == 2 ? 500 : 5000, nTr_dike = 30)
            Random.seed!(1234)
            Grid, Arrays, Tracers, Dikes, time_props = redirect_stdout(devnull) do
                MTK_GeoParams(Mat_RG, Num, Dikes)
            end
            return (; Grid, Num, T = Array(Arrays.T), ϕ = Array(Arrays.ϕ), Phases = Array(Arrays.Phases), InjectVol = Dikes.InjectVol, Tracers, time_props)
        end
        for dim in (2, 3)
            cpu = run_model(CPU(), dim)
            @test !cpu.Grid.ConstantΔ
            @test cpu.Num.Δmin ≈ minimum(cpu.Grid.Δ) && cpu.Num.dt ≈ 0.2 * cpu.Num.Δmin^2 / cpu.Num.κ_time / 4      # time step from the smallest spacing
            @test cpu.InjectVol > 0 && all(isfinite, cpu.T) && maximum(cpu.T) > 500
            # volume-weighted mean melt fraction and magma temperature
            V = .*(ntuple(d -> reshape(cell_widths(cpu.Grid.coord1D[d]), ntuple(k -> k == d ? cpu.Grid.N[d] : 1, dim)), dim)...)
            @test cpu.time_props.MeltFraction[end] ≈ sum(cpu.ϕ .* V) / sum(V) rtol = 1.0e-12
            @test cpu.time_props.Tav_magma[end] ≈ sum(cpu.T .* (cpu.T .> 700) .* V) / sum((cpu.T .> 700) .* V) rtol = 1.0e-12
            jl = run_model(JLBackend(), dim)
            @test jl.T ≈ cpu.T rtol = 1.0e-12
            @test jl.Phases == cpu.Phases
            @test jl.time_props.MeltFraction ≈ cpu.time_props.MeltFraction rtol = 1.0e-12
            @test isapprox(jl.time_props.Tav_magma, cpu.time_props.Tav_magma; rtol = 1.0e-12, nans = true)          # NaN before the first injection
        end
        imp = run_model(CPU(), 2; solver = :implicit_enthalpy)
        @test all(isfinite, imp.T) && imp.InjectVol > 0
        @test_throws "cannot be combined with CartData" begin
            Num = NumParam(; Nx = 9, Nz = 9, refine = (x = (zone = (0.0, 1.0), ratio = 2),), Output_VTK = false)
            X, Y, Z = xyz_grid(0.0:1.0:8.0, 0.0, -8.0:1.0:0.0)
            MTK_GeoParams(Mat_RG, Num, SillParams(sill = CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 6.0e3 * m, H = 300.0 * m)); CartData_input = CartData(X, Y, Z, (Z = Z,)))
        end
    end
end

using Test, MagmaThermoKinematics
using MagmaThermoKinematics: EruptionParams, EruptionState, step_overpressure!, mixture_density, chamber_cells, chamber_geometry,
    mogi_displacement!, cell_sources, _lithostatic, dike_ascends, max_ascent_length

@testset "Eruptions" begin
    @testset "Mogi sum" begin
        # a single source: U = (1-ν) ΔV/π · d/|d|³ ; two opposite sources of opposite sign cancel nowhere but add linearly
        X = repeat(collect(-1000.0:100:1000), 1, 21); Z = repeat(collect(-2000.0:100:0)', 21, 1)
        U1 = (zero(X), zero(X)); U2 = (zero(X), zero(X)); U12 = (zero(X), zero(X))
        mogi_displacement!(U1, (X, Z), [0.0 -1000.0], [1.0e6]; softening = 1.0e-6)
        mogi_displacement!(U2, (X, Z), [300.0 -500.0], [-4.0e5]; softening = 1.0e-6)
        mogi_displacement!(U12, (X, Z), [0.0 -1000.0; 300.0 -500.0], [1.0e6, -4.0e5]; softening = 1.0e-6)
        @test U12[1] ≈ U1[1] .+ U2[1]
        @test U12[2] ≈ U1[2] .+ U2[2]
        i, j = 16, 11   # x = 500, z = -1000
        d = 500.0
        @test U1[1][i, j] ≈ 0.75 * 1.0e6 / π / d^2 rtol = 1.0e-10
        @test U1[2][i, j] ≈ 0.0 atol = 1.0e-12
    end

    @testset "Chamber detection" begin
        ϕ = zeros(40, 30); ϕ[5:15, 10:14] .= 0.7; ϕ[30:33, 3:4] .= 0.9
        cells = chamber_cells(ϕ, 0.5)
        @test length(cells) == 55
        coords = (range(0.0, 3.9e3, length = 40), range(-2.9e3, 0.0, length = 30))
        geo = chamber_geometry(cells, step.(coords), coords)
        @test geo.W ≈ 11 * step(coords[1])
        @test geo.h ≈ 5 * step(coords[2])
        @test geo.R ≈ geo.W / 2
        @test isempty(chamber_cells(zeros(5, 5), 0.5))
    end

    @testset "Overpressure and drain" begin
        ep = EruptionParams(m_w = 0.06, ΔP_crit = 3.0e6, σ_barrier = 0.0, w_dike = 20.0,
            melt_viscosity = LinearMeltViscosity(A = -8.1590, B = 2.4050e4K, T0 = -430.9606K, η0 = 1Pas), solubility = Liu2005_Solubility())
        depth = 4000.0
        st = EruptionState(); st.P_lith = st.P = _lithostatic(ep, depth, d -> 800.0)
        yr = 3.15576e7; n = 0; hsum = 0.0
        for _ in 1:400
            step_overpressure!(st, ep, 1073.15, 0.6, 1500.0, 4.0e3, 0.03 / yr, 100yr; depth, η_r = 1.0e19)
            n += st.h_erupt > 0; hsum += st.h_erupt
            @test st.P - st.P_lith <= ep.ΔP_crit + 1.0e-3 * ep.ΔP_crit + 0.1 * ep.ΔP_crit
        end
        @test n > 0
        @test hsum > 0
    end
end

@testset "Line-integrated Mogi source" begin
    # a long line of sources (L ≫ r) tends to the plane-strain kernel 2 d/|d|²; a short one to the point source times 2L
    X = repeat(collect(-500.0:50:500), 1, 11); Z = repeat(collect(-1500.0:50:-1000)', 21, 1)
    Ul = (zero(X), zero(X)); Up = (zero(X), zero(X))
    mogi_displacement!(Ul, (X, Z), [0.0 -2000.0], [1.0e3]; L = [1.0e9], softening = 1.0e-6)
    mogi_displacement!(Up, (X, Z), [0.0 -2000.0], [1.0e3]; softening = 1.0e-6)
    d = 1000.0  # node at (0, -1000)
    @test Ul[2][11, 11] ≈ 0.75 * 1.0e3 / π * 2 / d rtol = 1.0e-6
    Us = (zero(X), zero(X)); mogi_displacement!(Us, (X, Z), [0.0 -2000.0], [1.0e3]; L = [1.0], softening = 1.0e-6)
    @test Us[2][11, 11] ≈ 2 * Up[2][11, 11] rtol = 1.0e-5
end

@testset "Free surface" begin
    using MagmaThermoKinematics: move_free_surface!
    z = range(-900.0, 100.0, length = 11); x = range(0.0, 400.0, length = 5)          # surface at z = 0 (cells 1:9 rock, 10:11 air)
    Ph = zeros(Int, 5, 11); Ph[:, 1:9] .= 1; Pi = copy(Ph); T = fill(10.0, 5, 11); T[:, 1:9] .= 500.0
    U = (zeros(5, 11), zeros(5, 11)); U[2][3, :] .= 120.0; U[2][1, :] .= -250.0            # uplift one column by 120 m, drop another by 250 m
    Zs = fill(NaN, 5)
    move_free_surface!(Ph, Pi, T, Zs, (x, z), U; air = 0, fill = 1)
    @test Zs[3] == 70.0 && Zs[1] == -300.0 && Zs[2] == -50.0     # surface starts half a cell above the top rock cell (z = −50)
    # sub-cell displacements accumulate: three uplifts of 40 m raise another column by one more cell
    for _ in 1:3; U[2] .= 0; U[2][5, :] .= 40.0; move_free_surface!(Ph, Pi, T, Zs, (x, z), U; air = 0, fill = 1); end
    @test Zs[5] == 70.0 && Ph[5, 10] == 1 && Ph[5, 11] == 0
    @test Ph[3, 10] == 1 && Ph[3, 11] == 0                      # cell centred at z = 0 is now below the surface (70 m)
    @test Ph[1, 9] == 0 && Ph[1, 8] == 0 && Ph[1, 7] == 1       # cells at z = −100, −200 are above the new surface (−300 m)
    @test Pi == Ph
    @test T[1, 9] == 10.0 && T[1, 8] == 10.0 && T[3, 9] == 500.0    # new air takes the air temperature; the (already advected) rock T is untouched
end

@testset "Chamber labelling" begin
    using MagmaThermoKinematics: chamber_labels!, largest_label_cells
    using Random
    rng = MersenneTwister(5)
    for sz in ((60, 40), (20, 18, 16))
        ϕ = rand(rng, sz...) .* 0.9
        ϕ[ntuple(k -> 3:7, length(sz))...] .= 1.0                    # one dense block
        lab = chamber_labels!(similar(ϕ, Int32), ϕ, 0.5)
        @test sort(largest_label_cells(lab)) == sort(chamber_cells(ϕ, 0.5))
        # every region carries the smallest linear index of its cells
        @test all(lab[I] == 0 || lab[I] <= LinearIndices(ϕ)[I] for I in CartesianIndices(ϕ))
    end
end

@testset "Volume-consistent sources" begin
    # flux of U through a box around a source: ΔV for :image (far from the surface), 4(1−ν) ΔV for :mogi
    X = repeat(collect(-3000.0:20:3000), 1, 301); Z = repeat(collect(-9000.0:20:-3000)', 301, 1)
    for (hs, fac) in ((:image, 1.0), (:mogi, 4 * (1 - 0.25)))
        U = (zero(X), zero(X)); mogi_displacement!(U, (X, Z), [0.0 -6000.0], [1.0e6]; L = [1.0e9], softening = 1.0e-6, halfspace = hs, z_surface = 0.0)
        # 2-D line source (L → ∞): kernel 2 d/|d|², divergence 4π × strength per unit length
        i1, i2 = 101, 201; j1, j2 = 101, 201      # x = ±1000, z = -6000 ± 1000
        dx = 20.0
        flux = sum(U[1][i2, j1:j2]) * dx - sum(U[1][i1, j1:j2]) * dx + sum(U[2][i1:i2, j2]) * dx - sum(U[2][i1:i2, j1]) * dx
        @test flux ≈ fac * 1.0e6 rtol = 0.05
    end
end

@testset "Image source surface displacement" begin
    # at the free surface the image doubles the vertical and cancels the horizontal displacement of the full-space source
    X = repeat(collect(-4000.0:100:4000), 1, 41); Z = repeat(collect(-4000.0:100:0)', 81, 1)
    U = (zero(X), zero(X)); mogi_displacement!(U, (X, Z), [0.0 -2000.0], [1.0e6]; softening = 1.0e-6, halfspace = :image, z_surface = 0.0)
    i = 61; j = 41                     # x = 2000, z = 0
    d = sqrt(2000.0^2 + 2000.0^2); uz_full = 1.0e6 / (4π) * 2000.0 / d^3
    @test U[2][i, j] ≈ 2 * uz_full rtol = 1.0e-6
    @test abs(U[1][i, j]) < 1.0e-9 * uz_full
end

@testset "3-D" begin
    using MagmaThermoKinematics: chamber_labels!, largest_label_cells, chamber_geometry, move_free_surface!, erupt!, mogi_displacement!, chamber_step!, EruptionParams, EruptionState
    using MagmaThermoKinematics: StructArray, Tracer
    # Mogi point source and image source in 3-D: value against the formula, flux through a box = ΔV (image) / 4(1−ν)ΔV (mogi)
    g = -1500.0:100:1500; X = [x for x in g, y in g, z in -3000.0:100:0]; Y = [y for x in g, y in g, z in -3000.0:100:0]; Z = [z for x in g, y in g, z in -3000.0:100:0]
    for (hs, fac) in ((:image, 1.0), (:mogi, 4 * (1 - 0.25)))
        U = (zero(X), zero(X), zero(X)); mogi_displacement!(U, (X, Y, Z), [0.0 0.0 -1500.0], [1.0e6]; softening = 1.0e-6, halfspace = hs, z_surface = 0.0)
        i = 16; j = 16; k = 16                                                   # cell (0, 0, −1500) = source; box of half-width 500 m around it: faces at index 11 and 21
        dA = 100.0^2; flux = 0.0
        for (d, Ud) in enumerate(U)
            lo = ntuple(n -> n == d ? (6:6) : (6:26), 3); hi = ntuple(n -> n == d ? (26:26) : (6:26), 3)      # box of half-width 1000 m
            flux += (sum(Ud[hi...]) - sum(Ud[lo...])) * dA
        end
        @test flux ≈ fac * 1.0e6 rtol = 0.1
        hs === :mogi && @test U[3][16, 16, 21] ≈ 0.75 * 1.0e6 / π * 500.0 / 500.0^3 rtol = 1.0e-8     # (1−ν)ΔV/π · d/|d|³ straight above the source
    end
    # chamber labelling / geometry / step on a 3-D melt field; free surface and eruption on a 3-D grid
    nx, ny, nz = 24, 20, 30; coords = (range(0.0, 2.3e3, length = nx), range(0.0, 1.9e3, length = ny), range(-2.9e3, 0.0, length = nz))
    ϕ = zeros(nx, ny, nz); ϕ[6:15, 5:14, 8:13] .= 0.8; ϕ[20:22, 2:3, 2:3] .= 0.9
    lab = chamber_labels!(similar(ϕ, Int32), ϕ, 0.5); cells = largest_label_cells(lab)
    @test length(cells) == 10 * 10 * 6
    geo = chamber_geometry(cells, step.(coords), coords)
    @test geo.W ≈ 100 * step(coords[1]) * step(coords[2]) && geo.h ≈ 6 * step(coords[3]) && geo.R ≈ sqrt(geo.W / π)
    T = fill(20.0, nx, ny, nz); T[ϕ .> 0] .= 900.0
    ep = EruptionParams(ΔP_crit = 1.0e5, σ_barrier = 0.0, w_dike = 50.0, h_melt_min = 10.0, melt_viscosity = LinearMeltViscosity(A = -8.1590, B = 2.4050e4K, T0 = -430.9606K, η0 = 1Pas))
    st = EruptionState(); ev = nothing
    for it in 1:400
        ev = chamber_step!(st, ep, ϕ, T, coords; recharge = 2.0e5 / 3.15576e7, Δt = 10 * 3.15576e7, time = it * 10 * 3.15576e7, T_background = d -> 20.0 + 0.02d, margin = 0.0, resolve = 0.1)
        ev === nothing || break
    end
    @test ev !== nothing && ev.f > 0 && length(ev.cells) == 600
    Tr = StructArray([Tracer{Float32}(num = i, coord = [coords[1][10], coords[2][10], coords[3][10]], Phase = 1) for i in 1:50])
    Tb = similar(T); U, cargo = erupt!(T, Tb, Tr, coords, ev; softening = 100.0, halfspace = :image, volume_factor = 0.5)
    @test length(U) == 3 && all(isfinite, U[3]) && length(cargo) + length(Tr) == 50
    Ph = ones(Int, nx, ny, nz); Ph[:, :, nz] .= 0; Pi = copy(Ph); Zs = fill(NaN, nx, ny); Tf = copy(T)
    Uf = (zeros(nx, ny, nz), zeros(nx, ny, nz), zeros(nx, ny, nz)); Uf[3][3, 4, :] .= 150.0
    move_free_surface!(Ph, Pi, Tf, Zs, coords, Uf; air = 0, fill = 1)
    @test Zs[3, 4] ≈ 150.0 - step(coords[3]) / 2 && Ph[3, 4, nz] == 1 && Zs[1, 1] ≈ -step(coords[3]) / 2
end

@testset "Mindlin half-space sources (3-D)" begin
    using MagmaThermoKinematics: mogi_displacement!
    g = -3000.0:100:3000; X = [x for x in g, y in g, z in -8000.0:100:0]; Y = [y for x in g, y in g, z in -8000.0:100:0]; Z = [z for x in g, y in g, z in -8000.0:100:0]
    ΔV = 1.0e6; c = 5000.0; ν = 0.25
    U = (zero(X), zero(X), zero(X)); mogi_displacement!(U, (X, Y, Z), [0.0 0.0 -c], [ΔV]; softening = 1.0e-6, halfspace = :mindlin, z_surface = 0.0, ν = ν)
    # surface: the Mogi formula
    i = 51; j = 31; k = 81                    # x = 2000, y = 0, z = 0
    R = sqrt(2000.0^2 + c^2)
    @test U[1][i, j, k] ≈ (1 - ν) * ΔV / π * 2000.0 / R^3 rtol = 1.0e-10
    @test U[3][i, j, k] ≈ (1 - ν) * ΔV / π * c / R^3 rtol = 1.0e-10
    # near the source: full space; box flux = ΔV
    @test U[1][32, 31, 31] ≈ ΔV / (4π) / 100.0^2 rtol = 0.02    # 100 m beside the source
    flux = 0.0; dA = 100.0^2
    for (d, Ud) in enumerate(U)
        lo = ntuple(n -> n == d ? (21:21) : (21:41), 3); hi = ntuple(n -> n == d ? (41:41) : (21:41), 3)
        flux += (sum(Ud[hi...]) - sum(Ud[lo...])) * dA
    end
    @test flux ≈ ΔV rtol = 0.1
    @test_throws ArgumentError mogi_displacement!((zero(X[:, 1, :]), zero(X[:, 1, :])), (X[:, 1, :], Z[:, 1, :]), [0.0 -c], [ΔV]; softening = 1.0, halfspace = :mindlin)
end

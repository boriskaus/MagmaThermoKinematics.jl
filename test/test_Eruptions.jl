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
    dz = move_free_surface!(Ph, Pi, T, (x, z), U; air = 0, fill = 1)
    @test dz[3] == 120.0 && dz[1] == -250.0 && dz[2] == 0.0
    @test Ph[3, 10] == 1 && Ph[3, 11] == 0                      # cell centred at z = 0 is now below the surface (+70 m)
    @test Ph[1, 9] == 0 && Ph[1, 8] == 0 && Ph[1, 7] == 1       # cells at z = −100, −200 are above the new surface (−300 m)
    @test Pi == Ph
    @test T[1, 9] == 10.0 && T[1, 8] == 10.0 && T[3, 9] == 500.0    # new air takes the air temperature; the (already advected) rock T is untouched
end

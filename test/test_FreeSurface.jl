# =====================================================================
#  Free-surface tests  (issue 4 — moving free surface)
# =====================================================================
#
#  Test-first specification of the kinematic sticky-air free surface. These
#  tests are the contract; the implementation in src/FreeSurface.jl must satisfy
#  them.
#
#  MODEL (kinematic sticky-air free surface)
#  -----------------------------------------
#  * The surface is a topography `z_surf` on the fixed grid: one elevation per
#    surface column (2D: length-Nx vector; 3D: Nx×Ny matrix). z increases
#    upward; the domain spans z ∈ [-H, 0].
#  * A grid cell is "air" when it lies strictly above the topography
#    (z_cell > z_surf[col]). Air cells are stamped: T = Tair, ϕ = 0,
#    Phases = air_phase. Cells at or below the surface are left untouched.
#  * The surface moves with the host rock: it is advected vertically by the
#    displacement field `Dz` (same shape as T), sampled at the surface elevation
#    of each column. Injection inflation raises it; eruption deflation lowers it.
#  * The surface stays within the grid's vertical extent (clamped to [z₁, zₙ]).
#
#  PROPERTIES UNDER TEST
#  ---------------------
#  F1  init_free_surface: correct shape (2D/3D) and flat elevation z0
#  F2  apply_free_surface!: stamps air above a flat topography; below untouched
#  F3  strict boundary: a cell exactly at z_surf is rock (not air)
#  F4  per-column (sloped) topography: air region follows the topography
#  F5  air coupling: ϕ=0 in air ⇒ air is never eruptible
#  F6  advect_surface!: uniform uplift / subsidence move z_surf by ±Dz
#  F7  advect_surface! samples Dz at the surface elevation (depth-varying Dz)
#  F8  advect_surface! clamps the surface to the grid's vertical extent
#  F9  round trip: subside then re-apply ⇒ newly exposed cells become air
#  F10 3D works (init + apply + advect, finite)
#
using Test
using MagmaThermoKinematics
using MagmaThermoKinematics: init_free_surface, apply_free_surface!, advect_surface!,
                             advect_phases!, eruptible_volume, mass_budget, CreateGrid

@testset "FreeSurface" begin

    # ---- F11: advect_phases! — integer-safe host-rock advection -----------
    # The phase field must move *with* the host rock (so the free surface, host
    # rock and sills stay consistent) while keeping valid integer phase indices.
    @testset "F11 advect_phases!" begin
        # 2D: a uniform upward displacement of exactly +2 cells shifts the
        # column up by 2; bottom 2 rows replicate the (clamped) bottom phase.
        G  = CreateGrid(size=(5,11), extent=(4e3,10e3))      # dz = 1000 m
        dz = G.Δ[2]
        Ph = repeat(reshape(collect(1:11), 1, 11), 5, 1)     # Phases[i,j] = j
        Vel = (zeros(5,11), fill(2*dz, 5, 11))               # +2 cells of uplift
        advect_phases!(Ph, Vel, G)
        @test eltype(Ph) <: Integer
        @test all(Ph[:, 3:11] .== reshape(1:9, 1, 9))        # j ← j-2
        @test all(Ph[:, 1:2] .== 1)                          # clamped to bottom phase
        @test all(in(1:11), Ph)                              # only valid phase indices

        # subsidence (downward) shifts the column down and clamps at the top
        Ph2 = repeat(reshape(collect(1:11), 1, 11), 5, 1)
        advect_phases!(Ph2, (zeros(5,11), fill(-3*dz, 5, 11)), G)
        @test all(Ph2[:, 1:8] .== reshape(4:11, 1, 8))       # j ← j+3
        @test all(Ph2[:, 9:11] .== 11)                       # clamped to top phase

        # non-finite displacement is treated as zero (field unchanged, finite)
        Ph3 = copy(Ph2); before = copy(Ph3)
        advect_phases!(Ph3, (fill(NaN,5,11), fill(Inf,5,11)), G)
        @test Ph3 == before

        # 3D: a +1-cell uplift in z shifts the column up by one
        G3  = CreateGrid(size=(3,3,6), extent=(2e3,2e3,5e3)) # dz = 1000 m
        dz3 = G3.Δ[3]
        Ph4 = [k for i in 1:3, j in 1:3, k in 1:6]
        advect_phases!(Ph4, (zeros(3,3,6), zeros(3,3,6), fill(dz3,3,3,6)), G3)
        @test all(Ph4[:,:,2:6] .== reshape(1:5, 1,1,5))
        @test all(Ph4[:,:,1] .== 1)
    end

    # ---- F1: init_free_surface shape + flat elevation -------------------
    @testset "F1 init" begin
        G2 = CreateGrid(size=(11,9),  extent=(10e3,8e3))
        z2 = init_free_surface(G2; z0=-1000.0)
        @test z2 isa AbstractVector && length(z2) == 11
        @test all(z2 .== -1000.0)

        G3 = CreateGrid(size=(7,5,9), extent=(10e3,6e3,8e3))
        z3 = init_free_surface(G3; z0=0.0)
        @test size(z3) == (7,5)
        @test all(z3 .== 0.0)
    end

    # ---- F2: apply stamps air above a flat surface; below untouched -----
    @testset "F2 apply flat" begin
        G = CreateGrid(size=(21,21), extent=(10e3,10e3))      # x∈[0,1e4], z∈[-1e4,0]
        z = G.coord1D[2]
        T = fill(700.0, 21, 21); ϕ = fill(0.6, 21, 21); Ph = ones(Int, 21, 21)
        z_surf = fill(-4000.0, 21)
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0)

        air  = [z[j] > z_surf[i] for i in 1:21, j in 1:21]
        @test all(T[air]  .== 0.0)
        @test all(ϕ[air]  .== 0.0)
        @test all(Ph[air] .== 0)
        # cells at/below the surface are untouched
        @test all(T[.!air]  .== 700.0)
        @test all(ϕ[.!air]  .== 0.6)
        @test all(Ph[.!air] .== 1)
        @test any(air) && !all(air)                          # a genuine partial split
    end

    # ---- F2b: fill_phase back-fills air below a risen surface -----------
    # When the continuous z_surf rises past cells that the (rounded) phase
    # advection left as air, `fill_phase` must reclassify those below-surface
    # air cells as rock so the interface follows the surface exactly.
    @testset "F2b backfill below surface" begin
        G = CreateGrid(size=(6,21), extent=(5e3,10e3))
        z = G.coord1D[2]
        T = fill(700.0, 6, 21); ϕ = fill(0.6, 6, 21)
        # a stale flat air band down to -4000 m (as left by sub-cell advection)
        Ph = ones(Int, 6, 21); Ph[:, z .> -4000.0] .= 0
        z_surf = fill(-1000.0, 6)                              # surface risen to -1000
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0; fill_phase=1)
        air = [z[j] > z_surf[i] for i in 1:6, j in 1:21]
        @test all(Ph[air]  .== 0)                             # air above surface
        @test all(Ph[.!air] .== 1)                            # NO air left below surface
        @test !any((Ph .== 0) .& .!air)                       # interface == z_surf
        # without fill_phase the stale band is preserved (back-compat)
        Ph2 = ones(Int, 6, 21); Ph2[:, z .> -4000.0] .= 0
        apply_free_surface!(fill(700.0,6,21), fill(0.6,6,21), Ph2, z_surf, G, 0.0, 0)
        @test any((Ph2 .== 0) .& .!air)                       # air below surface remains
    end

    # ---- F3: strict boundary (cell exactly at z_surf is rock) -----------
    @testset "F3 strict boundary" begin
        G = CreateGrid(size=(5,11), extent=(4e3,10e3))       # z nodes at -1e4:1e3:0
        z = G.coord1D[2]
        T = fill(700.0, 5, 11); ϕ = fill(0.6, 5, 11); Ph = ones(Int, 5, 11)
        z_surf = fill(z[6], 5)                                # surface ON a node
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0)
        @test all(T[:,6] .== 700.0)                          # the on-surface node is rock
        @test all(T[:,7] .== 0.0)                            # the node above is air
    end

    # ---- F4: per-column (sloped) topography -----------------------------
    @testset "F4 sloped topography" begin
        G = CreateGrid(size=(6,21), extent=(5e3,10e3))
        z = G.coord1D[2]
        T = fill(700.0, 6, 21); ϕ = fill(0.6, 6, 21); Ph = ones(Int, 6, 21)
        z_surf = collect(range(-6000.0, -1000.0, length=6))  # rising topography in x
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0)
        for i in 1:6
            n_air = count(z .> z_surf[i])
            @test count(==(0.0), T[i,:]) == n_air            # air count tracks topography
        end
    end

    # ---- F5: air is never eruptible (ϕ=0 coupling) ----------------------
    @testset "F5 air not eruptible" begin
        G = CreateGrid(size=(21,21), extent=(10e3,10e3))
        T = fill(900.0, 21, 21); ϕ = fill(0.9, 21, 21); Ph = ones(Int, 21, 21)
        z_surf = fill(-5000.0, 21)
        Ve_before, _ = eruptible_volume(ϕ, G.Δ, 0.5)
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0)
        Ve_after, mask = eruptible_volume(ϕ, G.Δ, 0.5)
        @test Ve_after < Ve_before                           # air removed mobile melt
        air = [G.coord1D[2][j] > z_surf[i] for i in 1:21, j in 1:21]
        @test !any(mask[air])                                # no air cell is eruptible
    end

    # ---- F6: advect by uniform uplift / subsidence ----------------------
    @testset "F6 advect uniform" begin
        G = CreateGrid(size=(8,21), extent=(7e3,10e3))
        # uplift
        z_up = fill(-5000.0, 8)
        advect_surface!(z_up, fill(+500.0, 8, 21), G)
        @test all(z_up .≈ -4500.0)
        # subsidence
        z_dn = fill(-5000.0, 8)
        advect_surface!(z_dn, fill(-500.0, 8, 21), G)
        @test all(z_dn .≈ -5500.0)
    end

    # ---- F7: advect samples Dz at the surface elevation -----------------
    @testset "F7 advect samples at surface" begin
        G = CreateGrid(size=(4,21), extent=(3e3,10e3))
        z = G.coord1D[2]
        Dz = [Float64(z[j]) for i in 1:4, j in 1:21]         # Dz equals the elevation
        z_surf = fill(-3000.0, 4)
        # nearest z-node to -3000 (z step = 500): index, value
        kk = clamp(round(Int, (-3000.0 - z[1])/(z[2]-z[1])) + 1, 1, length(z))
        expected = clamp(-3000.0 + z[kk], min(z[1],z[end]), max(z[1],z[end]))
        advect_surface!(z_surf, Dz, G)
        @test all(z_surf .≈ expected)
    end

    # ---- F8: advect clamps the surface to the grid extent ---------------
    @testset "F8 advect clamp" begin
        G = CreateGrid(size=(5,21), extent=(4e3,10e3))
        z = G.coord1D[2]
        z_hi = fill(-1000.0, 5); advect_surface!(z_hi, fill(+1e9, 5, 21), G)
        @test all(z_hi .≈ z[end])                            # clamped to the top
        z_lo = fill(-1000.0, 5); advect_surface!(z_lo, fill(-1e9, 5, 21), G)
        @test all(z_lo .≈ z[1])                              # clamped to the bottom
        @test all(isfinite, z_hi) && all(isfinite, z_lo)
    end

    # ---- F9: round trip — subside then re-apply exposes new air ---------
    @testset "F9 subside then apply" begin
        G = CreateGrid(size=(11,21), extent=(10e3,10e3))
        z = G.coord1D[2]
        T = fill(700.0, 11, 21); ϕ = fill(0.6, 11, 21); Ph = ones(Int, 11, 21)
        z_surf = fill(-4000.0, 11)
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0)
        n_air0 = count(==(0.0), T)
        advect_surface!(z_surf, fill(-1500.0, 11, 21), G)    # subside 1500 m
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0)
        n_air1 = count(==(0.0), T)
        @test all(z_surf .≈ -5500.0)
        @test n_air1 > n_air0                                # subsidence exposed more air
    end

    # ---- F10: 3D works --------------------------------------------------
    @testset "F10 3D" begin
        G = CreateGrid(size=(9,7,21), extent=(8e3,6e3,10e3))
        z = G.coord1D[3]
        T = fill(700.0, 9,7,21); ϕ = fill(0.6, 9,7,21); Ph = ones(Int, 9,7,21)
        z_surf = init_free_surface(G; z0=-3000.0)
        advect_surface!(z_surf, fill(-1000.0, 9,7,21), G)    # subside 1000 m
        @test all(z_surf .≈ -4000.0)
        apply_free_surface!(T, ϕ, Ph, z_surf, G, 0.0, 0)
        air = [z[k] > z_surf[i,j] for i in 1:9, j in 1:7, k in 1:21]
        @test all(T[air] .== 0.0) && all(ϕ[air] .== 0.0) && all(Ph[air] .== 0)
        @test all(T[.!air] .== 700.0)
        @test all(isfinite, T)
    end

    # ---- F12: init_free_surface with a non-flat topography --------------
    # The surface machinery (apply/advect) is already per-column; F12 only
    # exercises the *initialisation* from a topography (array or function).
    @testset "F12 topographic init" begin
        # 2D: flat default unchanged
        G = CreateGrid(size=(11,21), extent=(10e3,10e3))
        x = G.coord1D[1]
        @test init_free_surface(G; z0=-2000.0) == fill(-2000.0, 11)

        # 2D: function of x — a linear ramp; matches an explicit array init
        framp(xc) = -2000.0 + 0.1*xc
        zf = init_free_surface(G; topography=framp)
        @test length(zf) == 11
        @test all(zf .≈ (-2000.0 .+ 0.1 .* x))
        za = init_free_surface(G; topography=collect(zf))
        @test za == zf                                       # array init reproduces it
        @test za !== zf                                      # but is a fresh array (copy)

        # 2D: a non-flat init feeds straight into apply_free_surface! — the
        # air region follows the sloped surface (links F4).
        z2 = G.coord1D[2]
        T = fill(700.0, 11, 21); ϕ = fill(0.6, 11, 21); Ph = ones(Int, 11, 21)
        apply_free_surface!(T, ϕ, Ph, zf, G, 0.0, 0)
        air = [z2[j] > zf[i] for i in 1:11, j in 1:21]
        @test all(T[air] .== 0.0) && all(T[.!air] .== 700.0)

        # 2D: wrong-length array errors
        @test_throws ErrorException init_free_surface(G; topography=zeros(10))

        # 3D: function of (x,y); matches explicit array; wrong shape errors
        G3 = CreateGrid(size=(9,7,21), extent=(8e3,6e3,10e3))
        x3 = G3.coord1D[1]; y3 = G3.coord1D[2]
        fbump(xc,yc) = -3000.0 + 1e-4*(xc^2 - yc^2)
        zf3 = init_free_surface(G3; topography=fbump)
        @test size(zf3) == (9,7)
        @test all(zf3 .≈ [-3000.0 + 1e-4*(x3[i]^2 - y3[j]^2) for i in 1:9, j in 1:7])
        @test init_free_surface(G3; topography=collect(zf3)) == zf3
        @test_throws ErrorException init_free_surface(G3; topography=zeros(9,6))
    end

    # ---- F13: mass_budget conservation diagnostic -----------------------
    # `mass_budget` reports  residual = injected − erupted − Σ(z_surf−z_surf0)·dA,
    # with dA = Δx (2D, per-unit-depth) or Δx·Δy (3D). F13 pins the arithmetic and
    # the 2D/3D area element on a synthetic surface where the answer is exact.
    @testset "F13 mass_budget" begin
        # 3D: dA = Δx·Δy, uniform uplift ⇒ Δsurface = uplift·area
        G   = CreateGrid(size=(5,4,3), extent=(4e3,3e3,2e3))
        dA  = G.Δ[1]*G.Δ[2]
        z0  = init_free_surface(G; z0=-1000.0)          # 5×4 flat
        zs  = z0 .+ 50.0                                # uniform 50 m uplift
        area = length(z0)*dA
        b = mass_budget(1.0e9, 4.0e8, zs, z0, G)
        @test b.Δsurface  ≈ 50.0*area
        @test b.injected  == 1.0e9
        @test b.erupted   == 4.0e8
        @test b.residual  ≈ 1.0e9 - 4.0e8 - 50.0*area
        @test b.rel_residual ≈ b.residual / max(1.0e9, 4.0e8, 50.0*area)
        # scalar z_surf0 reference matches an equivalent array reference
        @test mass_budget(1.0e9, 4.0e8, zs, -1000.0, G).Δsurface ≈ b.Δsurface

        # exact closure: pick erupted so injected−erupted equals Δsurface ⇒ residual 0
        @test mass_budget(50.0*area, 0.0, zs, z0, G).residual ≈ 0.0 atol=1e-3

        # 2D: dA = Δx (per-unit-depth); non-uniform surface integrated correctly
        G2  = CreateGrid(size=(6,11), extent=(5e3,10e3))
        z02 = init_free_surface(G2; z0=-2000.0)         # length-6
        Δz  = collect(range(0.0, 100.0; length=6))      # sloped uplift
        zs2 = z02 .+ Δz
        @test mass_budget(0.0, 0.0, zs2, z02, G2).Δsurface ≈ sum(Δz)*G2.Δ[1]
    end

end

# =====================================================================
#  Eruption tests  (issue 2 — eruption capability)
# =====================================================================
#
#  Test-first specification of the eruption model. These tests are the
#  contract; the implementation in src/InjectSills_utils.jl must satisfy them.
#
#  MODEL (kinematic thermal-extraction proxy)
#  ------------------------------------------
#  * A cell is *eruptible* when its melt fraction ϕ > ϕ_erupt (default 0.5 =
#    mobile magma).
#  * Eruptible volume:  V_e = Σ_{eruptible cells} ϕ · V_cell
#    (the mobile-melt volume; in 2D a per-unit-depth volume = area·1 m).
#  * Trigger: an eruption happens if  Erupt.erupt == true  AND  V_e ≥ V_crit.
#  * Withdrawal: a fraction η = erupt_efficiency of the mobile melt is removed
#    from each eruptible cell, ϕ → (1-η)·ϕ. This is realized thermally by
#    cooling the cell by  ΔT = η·ϕ / (dϕ/dT)  — the local linearized drop that
#    lowers ϕ by η·ϕ.  ⇒ melt actually removed per cell = η·ϕ·V_cell.
#  * Erupted (booked) volume  =  Σ η·ϕ·V_cell  =  η · V_e   (SELF-CONSISTENT
#    with the withdrawal above — booked volume == melt actually removed).
#  * Optional kinematic deflation (deflate=true): a negative-ΔP Mogi source at
#    the melt-weighted centroid advects host rock inward; must stay finite.
#  * Bookkeeping: n_eruptions, cumulative erupted_volume, per-event
#    eruption_times and eruption_volumes.
#
#  PROPERTIES UNDER TEST
#  ---------------------
#  E1  diagnostic  eruptible_volume = Σ_{ϕ>thr} ϕ·V_cell ; strict threshold
#  E2  no trigger below V_crit (returns false; state untouched)
#  E3  trigger ≥ V_crit (returns true; bookkeeping = one event of η·V_e)
#  E4  thermal removal is exact (ΔT = η·ϕ/dϕdT) and local (others untouched)
#  E5  total melt strictly decreases; fully-withdrawn cells leave the eruptible set
#  E6  bookkeeping accumulates over multiple eruptions
#  E7  disabled by default (erupt=false ⇒ no-op even above V_crit)
#  E8  deflation path stays finite (negative-Mogi advection safe)
#  E9  booked volume == η · V_e (exactly)
#  E10 3D works (diagnostic + trigger + removal, finite)
#
using Test, Random
using MagmaThermoKinematics
using MagmaThermoKinematics: erupt_magma!, eruptible_volume, Tracer, CreateGrid, deflate_hostrock!
using InjectSills
using StructArrays

Random.seed!(1234)

# --- helper: a 2D melt setup with a uniform hot disk -----------------
function disk_setup(; Nx=101, Nz=101, W=10e3, H=10e3, ϕ_hot=0.8, R=2.0e3,
                      dϕdT_hot=1e-3, T_hot=950.0, T_bg=700.0)
    Grid = CreateGrid(size=(Nx,Nz), extent=(W,H))
    x, z = Grid.coord1D[1], Grid.coord1D[2]
    cx, cz = (x[1]+x[end])/2, (z[1]+z[end])/2
    ϕ    = zeros(Nx,Nz); T = fill(T_bg,Nx,Nz); dϕdT = zeros(Nx,Nz)
    for j in 1:Nz, i in 1:Nx
        if sqrt((x[i]-cx)^2+(z[j]-cz)^2) < R
            ϕ[i,j] = ϕ_hot; T[i,j] = T_hot; dϕdT[i,j] = dϕdT_hot
        end
    end
    return Grid, ϕ, T, dϕdT
end

@testset "Eruption" begin

    # ---- E1: eruptible-volume diagnostic --------------------------------
    @testset "E1 diagnostic" begin
        Grid, ϕ, _, _ = disk_setup()
        Vcell = prod(Grid.Δ)
        Ve, mask = eruptible_volume(ϕ, Grid.Δ, 0.5)
        ncells = count(>(0.5), ϕ)
        @test mask == (ϕ .> 0.5)
        @test Ve ≈ 0.8 * ncells * Vcell  rtol=1e-12          # uniform ϕ=0.8 disk
        # strict threshold: ϕ exactly at threshold is NOT eruptible
        ϕb = fill(0.5, 4, 4)
        Vb, mb = eruptible_volume(ϕb, Grid.Δ, 0.5)
        @test Vb == 0.0 && !any(mb)
        # all-below threshold ⇒ zero
        Vz, mz = eruptible_volume(fill(0.1, 4, 4), Grid.Δ, 0.5)
        @test Vz == 0.0 && !any(mz)
    end

    # ---- E2 / E7: no trigger below V_crit, and disabled by default ------
    @testset "E2/E7 no trigger" begin
        Grid, ϕ, T, dϕdT = disk_setup()
        Ve, _ = eruptible_volume(ϕ, Grid.Δ, 0.5)
        Tr = StructArray{Tracer{Float32}}(undef,1)

        # below critical volume
        E = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*1.5, deflate=false, out_of_plane_3D=false)
        T2 = copy(T)
        fired = erupt_magma!(T2, copy(ϕ), copy(dϕdT), Tr, Grid, E, 0.0)
        @test fired == false
        @test T2 == T                       # state untouched
        @test E.n_eruptions == 0 && E.erupted_volume == 0.0

        # disabled (erupt=false) even though V_e ≥ V_crit
        Edis = EruptionParams(erupt=false, ϕ_erupt=0.5, V_crit=Ve*0.5, deflate=false, out_of_plane_3D=false)
        @test erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, Edis, 0.0) == false
        @test Edis.n_eruptions == 0
    end

    # ---- E3 / E9: trigger + bookkeeping = one event of η·V_e ------------
    @testset "E3/E9 trigger + bookkeeping" begin
        Grid, ϕ, T, dϕdT = disk_setup()
        Ve, _ = eruptible_volume(ϕ, Grid.Δ, 0.5)
        η  = 0.5
        Tr = StructArray{Tracer{Float32}}(undef,1)
        E  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=η, deflate=false, out_of_plane_3D=false)
        fired = erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, E, 1234.0)
        @test fired == true
        @test E.n_eruptions == 1
        @test E.eruption_times == [1234.0]
        @test length(E.eruption_volumes) == 1
        @test E.eruption_volumes[1] ≈ η*Ve      rtol=1e-12   # E9
        @test E.erupted_volume       ≈ η*Ve      rtol=1e-12
    end

    # ---- E4: thermal removal exact (ΔT = η·ϕ/dϕdT) and local ------------
    @testset "E4 removal exact + local" begin
        Grid, ϕ, T, dϕdT = disk_setup(ϕ_hot=0.8, dϕdT_hot=1e-3, T_hot=950.0)
        Ve, mask = eruptible_volume(ϕ, Grid.Δ, 0.5)
        η  = 0.5
        Tr = StructArray{Tracer{Float32}}(undef,1)
        E  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=η, deflate=false, out_of_plane_3D=false)
        T2 = copy(T)
        erupt_magma!(T2, copy(ϕ), copy(dϕdT), Tr, Grid, E, 0.0)
        ΔT_expected = η*0.8/1e-3                              # = 400 K
        @test all(isfinite, T2)
        # eruptible cells cooled by exactly ΔT_expected
        @test all(abs.((T .- T2)[mask] .- ΔT_expected) .< 1e-6)
        # non-eruptible cells untouched
        @test T2[.!mask] == T[.!mask]
    end

    # ---- E5: total melt strictly decreases; withdrawn cells de-mobilize -
    @testset "E5 melt removed" begin
        Grid, ϕ, T, dϕdT = disk_setup(ϕ_hot=0.8, dϕdT_hot=1e-3)
        Ve, mask = eruptible_volume(ϕ, Grid.Δ, 0.5)
        η  = 0.5
        Tr = StructArray{Tracer{Float32}}(undef,1)
        E  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=η, deflate=false, out_of_plane_3D=false)
        ϕ2 = copy(ϕ)
        # the implementation must also update ϕ to (1-η)ϕ in eruptible cells
        erupt_magma!(copy(T), ϕ2, copy(dϕdT), Tr, Grid, E, 0.0)
        @test sum(ϕ2) < sum(ϕ)                                # total melt decreased
        @test all(ϕ2[mask] .≈ (1-η).*ϕ[mask])                 # ϕ → (1-η)ϕ
        # 0.8 → 0.4 < 0.5 ⇒ those cells leave the eruptible set
        Ve_after, _ = eruptible_volume(ϕ2, Grid.Δ, 0.5)
        @test Ve_after < Ve
    end

    # ---- E6: bookkeeping accumulates over multiple eruptions ------------
    @testset "E6 accumulation" begin
        Grid, ϕ, T, dϕdT = disk_setup()
        Ve, _ = eruptible_volume(ϕ, Grid.Δ, 0.5)
        η  = 0.5
        Tr = StructArray{Tracer{Float32}}(undef,1)
        E  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=η, deflate=false, out_of_plane_3D=false)
        erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, E, 100.0)
        erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, E, 200.0)
        @test E.n_eruptions == 2
        @test E.eruption_times == [100.0, 200.0]
        @test E.erupted_volume ≈ 2*η*Ve   rtol=1e-12
    end

    # ---- E8: deflation path stays finite -------------------------------
    @testset "E8 deflation finite" begin
        Grid, ϕ, T, dϕdT = disk_setup()
        Ve, _ = eruptible_volume(ϕ, Grid.Δ, 0.5)
        Tr = StructArray{Tracer{Float32}}(undef,1)
        E  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=0.5, deflate=true,
                            ΔP=20e6, G=10e9, ν=0.25, out_of_plane_3D=false)
        T2 = copy(T)
        fired = erupt_magma!(T2, copy(ϕ), copy(dϕdT), Tr, Grid, E, 0.0)
        @test fired == true
        @test all(isfinite, T2)
        @test E.n_eruptions == 1
    end

    # ---- E10: 3D works -------------------------------------------------
    @testset "E10 3D" begin
        Nx,Ny,Nz = 41,41,41
        Grid = CreateGrid(size=(Nx,Ny,Nz), extent=(10e3,10e3,10e3))
        x,y,z = Grid.coord1D
        cx,cy,cz = (x[1]+x[end])/2, (y[1]+y[end])/2, (z[1]+z[end])/2
        ϕ = zeros(Nx,Ny,Nz); T = fill(700.0,Nx,Ny,Nz); dϕdT = zeros(Nx,Ny,Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            if sqrt((x[i]-cx)^2+(y[j]-cy)^2+(z[k]-cz)^2) < 2.0e3
                ϕ[i,j,k]=0.8; T[i,j,k]=950.0; dϕdT[i,j,k]=1e-3
            end
        end
        Ve, mask = eruptible_volume(ϕ, Grid.Δ, 0.5)
        @test Ve ≈ 0.8*count(>(0.5),ϕ)*prod(Grid.Δ)  rtol=1e-12
        Tr = StructArray{Tracer{Float32}}(undef,1)
        E  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=0.5, deflate=false)
        T2 = copy(T)
        fired = erupt_magma!(T2, copy(ϕ), copy(dϕdT), Tr, Grid, E, 0.0)
        @test fired == true && all(isfinite, T2)
        @test E.erupted_volume ≈ 0.5*Ve   rtol=1e-12
    end

    # ---- E11: tiny dϕdT (flat melt curve) is clamped to T_min --------------
    # Near melt saturation dϕdT→0, so the linearized ΔT = η·ϕ/dϕdT explodes.
    # The cooling must be floored at Erupt.T_min so T stays finite/physical,
    # while ϕ-withdrawal and booked volume remain exact (η·Ve, dϕdT-independent).
    @testset "E11 cooling floor" begin
        Grid, ϕ, T, dϕdT = disk_setup(ϕ_hot=1.0, dϕdT_hot=1e-9, T_hot=1000.0)
        Ve, mask = eruptible_volume(ϕ, Grid.Δ, 0.5)
        η  = 0.5
        Tr = StructArray{Tracer{Float32}}(undef,1)
        E  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=η, deflate=false, T_min=50.0, out_of_plane_3D=false)
        T2 = copy(T); ϕ2 = copy(ϕ)
        fired = erupt_magma!(T2, ϕ2, copy(dϕdT), Tr, Grid, E, 0.0)
        @test fired == true
        @test all(isfinite, T2)                              # no runaway to -1e5
        @test all(T2[mask] .== 50.0)                         # clamped to the floor
        @test T2[.!mask] == T[.!mask]                        # non-eruptible untouched
        @test all(ϕ2[mask] .≈ (1-η).*ϕ[mask])                # ϕ-withdrawal unaffected
        @test E.erupted_volume ≈ η*Ve   rtol=1e-12           # booked volume still η·Ve
    end

    # ---- E12: out_of_plane_3D lifts the 2D eruptible volume to a 3D volume ----
    # In 2D the eruptible volume is per-unit-depth. With out_of_plane_3D (default)
    # it is multiplied by an effective out-of-plane length √(2π)·max(σ,dx), σ =
    # melt-weighted horizontal std of the eruptible region — so it (and the booked
    # erupted volume) become km³, comparable to V_crit and the injected volume.
    @testset "E12 out-of-plane 3D lift" begin
        Grid, ϕ, T, dϕdT = disk_setup(ϕ_hot=0.8, dϕdT_hot=1e-3, R=2.0e3)
        Ve, mask = eruptible_volume(ϕ, Grid.Δ, 0.5)
        x  = Grid.coord1D[1]
        # melt-weighted horizontal std over the eruptible cells (matches erupt_magma!)
        w = sx = sx2 = 0.0
        for j in axes(ϕ,2), i in axes(ϕ,1)
            if mask[i,j]; w += ϕ[i,j]; sx += ϕ[i,j]*x[i]; sx2 += ϕ[i,j]*x[i]^2; end
        end
        σx = sqrt(max(sx2/w - (sx/w)^2, 0.0))
        Ly_expected = sqrt(2π)*max(σx, Grid.Δ[1])
        η  = 0.5
        Tr = StructArray{Tracer{Float32}}(undef,1)

        # with the lift (default): trigger + booking use the 3D volume
        E3 = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                            erupt_efficiency=η, deflate=false)   # out_of_plane_3D=true (default)
        @test erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, E3, 0.0) == true
        @test E3.erupted_volume ≈ η*Ve*Ly_expected   rtol=1e-9   # booked = η·(3D volume)
        @test Ly_expected > 1.0                                  # genuinely a 3D lift

        # the trigger now uses the larger 3D volume: a V_crit between the 2D and
        # 3D volumes fires with the lift but NOT without it
        Vc = 0.5*(Ve + Ve*Ly_expected)
        E_on  = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Vc, erupt_efficiency=η, deflate=false)
        E_off = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Vc, erupt_efficiency=η, deflate=false, out_of_plane_3D=false)
        @test erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, E_on,  0.0) == true
        @test erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, E_off, 0.0) == false
    end

    # ---- E13: erupted-tracer freeze ("zircon cargo") -----------------------
    # On eruption a random fraction = erupt_efficiency of the melt-rich (eruptible-
    # cell) tracers is FROZEN: flagged erupted, given a final (time,T) history
    # point, and thereafter neither advected nor T/Tt-updated — but retained in the
    # array so their preserved Tt-path is the erupted cargo for zircon analysis.
    @testset "E13 tracer freeze" begin
        Grid, ϕ, T, dϕdT = disk_setup(ϕ_hot=0.8, R=2.0e3)
        x, z = Grid.coord1D[1], Grid.coord1D[2]
        _, mask = eruptible_volume(ϕ, Grid.Δ, 0.5)

        # nearest-cell eligibility, mirroring freeze_erupted_tracers!'s mapping
        cell(tr) = mask[ clamp(round(Int,(tr.coord[1]-Grid.min[1])/Grid.Δ[1])+1, 1, Grid.N[1]),
                         clamp(round(Int,(tr.coord[2]-Grid.min[2])/Grid.Δ[2])+1, 1, Grid.N[2]) ]
        # interior tracer cloud (kept off the domain edges so advection never clamps)
        function seed(n)
            xs = range(x[1]+1e3, x[end]-1e3, length=n)
            zs = range(z[1]+1e3, z[end]-1e3, length=n)
            StructArray(vec([Tracer(num=k, coord=[xi, zj], T=950.0,
                                    time_vec=Float32[0.0], T_vec=Float32[950.0])
                             for (k,(xi,zj)) in enumerate(Iterators.product(xs,zs))]))
        end

        # --- direct unit test: efficiency = 1 ⇒ exactly the eligible tracers freeze
        Random.seed!(42)
        Tr   = seed(40)
        elig = [cell(tr) for tr in Tr]
        @test any(elig) && !all(elig)                         # a genuine subset is eligible
        nfz  = freeze_erupted_tracers!(Tr, Grid, mask, 1.0, 0.123)
        @test nfz == count(elig)
        @test collect(Tr.erupted) == elig                     # froze exactly the eligible set
        for i in findall(elig)                                # each got one final (t,T) point
            @test length(Tr[i].time_vec) == 2
            @test Tr[i].time_vec[end] ≈ 0.123f0
            @test Tr[i].T_vec[end]    ≈ 950.0f0
        end
        @test freeze_erupted_tracers!(Tr, Grid, mask, 1.0, 0.5) == 0   # idempotent

        # --- frozen tracers are not advected; their history no longer grows
        before = [copy(tr.coord) for tr in Tr]
        AdvectTracers!(Tr, Grid.coord1D, (fill(500.0, Grid.N...), fill(500.0, Grid.N...)), 1.0, "Euler")
        for i in eachindex(Tr)
            if Tr.erupted[i]
                @test Tr[i].coord == before[i]                # frozen ⇒ pinned
            else
                @test Tr[i].coord != before[i]                # mobile ⇒ advected
            end
        end
        update_Tvec!(Tr, 0.999)
        for i in findall(collect(Tr.erupted))
            @test length(Tr[i].time_vec) == 2                 # still just the eruption endpoint
            @test Tr[i].time_vec[end] ≈ 0.123f0
        end

        # --- via erupt_magma!: a random fraction ≈ erupt_efficiency of the
        #     eligible tracers freezes, and the endpoint is recorded in Myr.
        Random.seed!(7)
        Tr2   = seed(70)
        nelig = count(cell(tr) for tr in Tr2)
        Ve, _ = eruptible_volume(ϕ, Grid.Δ, 0.5)
        E = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5,
                           erupt_efficiency=0.5, deflate=false, out_of_plane_3D=false)
        erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr2, Grid, E, 1.0*Myr)   # t = 1 Myr (s)
        nfrozen = count(Tr2.erupted)
        @test 0 < nfrozen < nelig                             # proper subset
        @test isapprox(nfrozen/nelig, 0.5, atol=0.1)          # ≈ erupt_efficiency
        @test all(cell(Tr2[i]) for i in findall(collect(Tr2.erupted)))  # only eligible froze
        i = findfirst(collect(Tr2.erupted))
        @test Tr2[i].time_vec[end] ≈ 1.0f0  rtol=1e-3         # endpoint converted s → Myr
    end

    # ---- E14: per-cell (superposed) Mogi deflation -------------------------
    # The default deflation is a superposition of one negative-Mogi source per
    # eruptible cell (irregular, follows the melt distribution), vs the optional
    # single region-scale source.
    @testset "E14 per-cell deflation" begin
        Grid, ϕ, T, dϕdT = disk_setup()
        coord = Grid.coord1D
        x, z  = coord[1], coord[2]
        cz    = (z[1]+z[end])/2

        # (a) superposition is exact: multi-source field = Σ of single-source fields
        #     (each source zeroes only its OWN singular interior). Two unequal
        #     sources at mirror-image x positions ⇒ the summed field is asymmetric.
        s1 = MogiSphere(Center=Point2(x[1]+3e3,  cz)*m, r=400.0*m, ΔP=-20e6*Pa, G=10e9*Pa, ν=0.25*NoUnits)
        s2 = MogiSphere(Center=Point2(x[end]-3e3, cz)*m, r=700.0*m, ΔP=-20e6*Pa, G=10e9*Pa, ν=0.25*NoUnits)
        Tr0 = StructArray{Tracer{Float32}}(undef,1)
        _,_,Vsum = deflate_hostrock!(copy(T), Tr0, coord, [s1, s2])
        _,_,V1   = deflate_hostrock!(copy(T), Tr0, coord, s1)
        _,_,V2   = deflate_hostrock!(copy(T), Tr0, coord, s2)
        @test all(isfinite, Vsum[1]) && all(isfinite, Vsum[2])
        @test Vsum[1] ≈ V1[1] .+ V2[1]
        @test Vsum[2] ≈ V1[2] .+ V2[2]
        @test !isapprox(Vsum[2], reverse(Vsum[2], dims=1); atol=1e-9)   # asymmetric in x

        # (b) integration: the per-cell path is opt-in (deflate_percell=true);
        #     fires, stays finite, keeps tracer coords finite (frozen or advected).
        #     NB: per-cell deflation is O(N_eruptible × N_grid) per eruption and
        #     stalls at production resolution — region-scale is the default (c).
        Ve, _ = eruptible_volume(ϕ, Grid.Δ, 0.5)
        E = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5, erupt_efficiency=0.5,
                           deflate=true, deflate_percell=true, out_of_plane_3D=false)
        Tr = StructArray([Tracer(num=1, coord=[(x[1]+x[end])/2, cz], T=950.0)])
        T2 = copy(T)
        fired = erupt_magma!(T2, copy(ϕ), copy(dϕdT), Tr, Grid, E, 0.0)
        @test fired && all(isfinite, T2) && E.n_eruptions == 1
        @test all(isfinite, Tr[1].coord)

        # (c) region-scale is the DEFAULT (deflate_percell=false): cheap O(N_grid)
        Ereg = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve*0.5, erupt_efficiency=0.5,
                              deflate=true, out_of_plane_3D=false)
        @test Ereg.deflate_percell == false                   # default is region-scale
        Treg = StructArray{Tracer{Float32}}(undef,1)
        T3 = copy(T)
        @test erupt_magma!(T3, copy(ϕ), copy(dϕdT), Treg, Grid, Ereg, 0.0) == true
        @test all(isfinite, T3)
    end

    # ---- E15: depth-capped eruptibility (EruptAbove) -----------------------
    # Deep melt (e.g. host-rock partial melt under a hot geotherm at the base of
    # the domain) must not count as eruptible chamber. `EruptAbove` excludes cells
    # below a floor elevation.
    @testset "E15 depth cap" begin
        Grid, ϕ, T, dϕdT = disk_setup()          # uniform hot disk at the domain centre
        z  = Grid.coord1D[2]
        cz = (z[1]+z[end])/2                      # disk-centre elevation
        Ve0, m0 = eruptible_volume(ϕ, Grid.Δ, 0.5)

        # cap ABOVE the disk ⇒ the whole disk is excluded (deep cells dropped)
        Ve_hi, m_hi = eruptible_volume(ϕ, Grid.Δ, 0.5; zc=z, EruptAbove=cz+3e3)
        @test Ve_hi == 0.0 && !any(m_hi)
        # cap below the domain ⇒ identical to no cap
        Ve_lo, m_lo = eruptible_volume(ϕ, Grid.Δ, 0.5; zc=z, EruptAbove=z[1]-1.0)
        @test Ve_lo ≈ Ve0 && m_lo == m0
        # default (no zc / EruptAbove=-Inf) is unchanged
        @test eruptible_volume(ϕ, Grid.Δ, 0.5)[1] == Ve0

        # erupt_magma! respects the cap: a floor above the disk ⇒ no eruption,
        # even though the uncapped eruptible volume is well above V_crit.
        Tr = StructArray{Tracer{Float32}}(undef,1)
        Ecap = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve0*0.5, erupt_efficiency=0.5,
                              deflate=false, out_of_plane_3D=false, EruptAbove=cz+3e3)
        @test erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, Ecap, 0.0) == false
        @test Ecap.n_eruptions == 0
        # a floor below the disk ⇒ erupts as usual
        Eok = EruptionParams(erupt=true, ϕ_erupt=0.5, V_crit=Ve0*0.5, erupt_efficiency=0.5,
                             deflate=false, out_of_plane_3D=false, EruptAbove=z[1]-1.0)
        @test erupt_magma!(copy(T), copy(ϕ), copy(dϕdT), Tr, Grid, Eok, 0.0) == true
    end

end

# Melt fraction from precomputed phase diagrams (GeoParams' `MAGEMin_Diagram`), used with the implicit solver
using Test, Random, Printf
using MagmaThermoKinematics
using JLArrays
JLArrays.allowscalar(false)
MagmaThermoKinematics.KernelAbstractions.synchronize(::JLArrays.JLBackend) = nothing

# Writes a phase diagram in the MAGEMin format read by `MAGEMin_Diagram` (columns meltRho, meltFrac, rockRho, Vp, Vs, SpecificCp, T, P),
# with melt fraction `ϕfun(T [K], P [Pa])` on a regular T–P grid
function write_magemin_table(fname, ϕfun; T0 = 773.15, T1 = 1573.15, nT = 401, P0 = 1.0e5, P1 = 5.0e8, nP = 51)
    dT = (T1 - T0) / (nT - 1); dP = (P1 - P0) / (nP - 1)
    open(fname, "w") do io
        for _ in 1:48
            println(io)
        end
        println(io, "[meltRho, rho, kg/m^3] [meltFrac, wtPercent, NoUnits] [rockRho, rho, kg/m^3] [Vp, vp, km/s] [Vs, vs, km/s] [SpecificCp, scp, J/kg/K] [Temperature, T, K] [Pressure, P, Pa]")
        foreach(x -> println(io, x), (T0, dT, nT, P0, dP, nP))
        for j in 1:nP, i in 1:nT
            T = T0 + (i - 1) * dT; P = P0 + (j - 1) * dP
            @printf(io, "%.1f %.12f %.1f 6.0 3.5 1050.0 %.6f %.6f\n", 2400.0, ϕfun(T, P), 2700.0, T, P)
        end
    end
    return fname
end

const caricchi = MeltingParam_Caricchi()
ϕ_caricchi(T, P) = compute_meltfraction(caricchi, (; T, P))
# pressure-dependent melting: logistic around a melting temperature that decreases by 0.4 K/MPa
ϕ_pressure(T, P) = 1 / (1 + exp(-(T - (1273.15 - 0.4e-6 * P)) / 25))

const tab_caricchi = MAGEMin_Diagram(write_magemin_table(tempname() * ".in", ϕ_caricchi))
const tab_pressure = MAGEMin_Diagram(write_magemin_table(tempname() * ".in", ϕ_pressure))
# steep table, as computed phase diagrams can be: 30 % melt appear within one table interval (2 K) at a pressure-dependent solidus
function ϕ_steep(T, P)
    Ts = 1123.15 - 0.2e-6 * P
    return T ≤ Ts ? 0.0 : T ≤ Ts + 2 ? 0.3 * (T - Ts) / 2 : min(0.3 + 0.7 * (T - Ts - 2) / 250, 1.0)
end

materials(melting) = (
    SetMaterialParams(
        Name = "Rock", Phase = 1, Density = ConstantDensity(ρ = 2700kg / m^3), HeatCapacity = ConstantHeatCapacity(Cp = 1050J / kg / K),
        Conductivity = ConstantConductivity(k = 2.5Watt / K / m), LatentHeat = ConstantLatentHeat(Q_L = 3.5e5J / kg), Melting = melting
    ),
)

"Cool a 1050 °C sill in a 10 km × 10 km box for `nt` implicit steps; returns host copies of T, ϕ and P."
function cool_sill(melting; backend = CPU(), N = 61, nt = 5, fac = 4.0, ω = 1.0)
    Grid = CreateGrid(size = (N, N), extent = (10.0e3, 10.0e3))
    Num = NumParam(; Nx = N, Nz = N, solver = :implicit_enthalpy, fac_dt = fac, ω, verbose = false)
    names = (:T, :T_K, :T_it_old, :Tupdate, :Rho, :Cp, :Tnew, :Hr, :Hl, :Kc, :P, :X, :Z, :ϕ, :dϕdT, :Qrelax, :Qenth, :pt_C, :pt_b, :pt_phiold, :pt_res, :pt_Tk, :pt_Tkm1, :pt_Tn)
    A = CreateArrays(Dict((N, N) => NamedTuple{names}(ntuple(_ -> 0, length(names)))); backend)
    GridArray!(A.X, A.Z, Grid)
    Phases = similar(A.T, Int64); fill!(Phases, 1)
    A.T .= 20 .- A.Z .* 25.0e-3
    @. A.T += ifelse(abs(A.X - 5.0e3) < 2.0e3 && abs(A.Z + 6.0e3) < 0.5e3, 1050.0 - A.T, 0.0)
    Mat = materials_on_backend(materials(melting), backend)
    dt = fac * minimum(Grid.Δ .^ 2) / (2.5 / (2700 * 1050)) / 4
    for _ in 1:nt
        Nonlinear_Diffusion_step!(A, Mat, Phases, Grid, dt, Num)
        A.T .= A.Tnew
    end
    return Array(A.T), Array(A.ϕ), Array(A.P)
end

@testset "phase-diagram melt fraction" begin
    @testset "table reproduces the melting law" begin
        rng = MersenneTwister(3)
        for (tab, f) in ((tab_caricchi, ϕ_caricchi), (tab_pressure, ϕ_pressure))
            T = 800.0 .+ 750.0 .* rand(rng, 1000); P = 4.9e8 .* rand(rng, 1000)
            @test maximum(abs, [compute_meltfraction(tab, (; T = T[i], P = P[i])) - f(T[i], P[i]) for i in eachindex(T)]) < 2.0e-3
        end
    end

    @testset "implicit step: tabulated and analytic melting law agree" begin
        T_a, ϕ_a, _ = cool_sill(caricchi)
        T_t, ϕ_t, _ = cool_sill(tab_caricchi)
        @test maximum(ϕ_a) > 0.5                                   # the sill is (partly) molten
        @test maximum(abs, T_a - T_t) < 0.05
        @test maximum(abs, ϕ_a - ϕ_t) < 1.0e-3
    end

    @testset "melt fraction uses the model's lithostatic pressure" begin
        T, ϕ, P = cool_sill(tab_pressure)
        @test maximum(P) > 2.5e8                                   # 10 km of rock
        @test maximum(abs, ϕ - ϕ_pressure.(T .+ 273.15, P)) < 2.0e-3    # ϕ of the last iterate
        @test all(0 .≤ ϕ .≤ 1)
    end

    @testset "steep table converges with ω = 0.3" begin
        tab = MAGEMin_Diagram(write_magemin_table(tempname() * ".in", ϕ_steep))
        T, ϕ, P = cool_sill(tab; ω = 0.3)
        @test maximum(ϕ) > 0.3
        @test all(0 .≤ ϕ .≤ 1)
        @test maximum(abs, ϕ - [compute_meltfraction(tab, (; T = T[i] + 273.15, P = P[i])) for i in CartesianIndices(T)]) < 0.01   # ϕ of the last iterate; the table step is 0.15 per K
        @test all(20 .- 1.0 .≤ T .≤ 1050)                         # no over- or undershoots
    end

    @testset "materials_on_backend" begin
        Mat = materials(tab_pressure)
        @test materials_on_backend(Mat, CPU()) === Mat
        Mat_jl = materials_on_backend(Mat, JLBackend())
        @test Mat_jl[1].Melting[1].meltFrac.coefs isa JLArray
        @test Array(Mat_jl[1].Melting[1].meltFrac.coefs) == Mat[1].Melting[1].meltFrac.coefs
        @test materials_on_backend(materials(caricchi), JLBackend())[1].Melting[1] == caricchi    # no arrays: unchanged
    end

    @testset "JLArrays backend" begin
        T_c, ϕ_c, _ = cool_sill(tab_pressure)
        T_j, ϕ_j, _ = cool_sill(tab_pressure; backend = JLBackend())
        @test T_j ≈ T_c rtol = 1.0e-12
        @test ϕ_j ≈ ϕ_c rtol = 1.0e-10 atol = 1.0e-12
    end

    @testset "MTK_GeoParams with a phase diagram" begin
        function run(backend)
            Num = NumParam(;
                Nx = 33, Nz = 33, SimName = mktempdir(), maxTime_Myrs = 0.02, solver = :implicit_enthalpy, fac_dt = 1.0,
                Geotherm = 30 / 1.0e3, Output_VTK = false, backend
            )
            Dikes = SillParams(sill = CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 20.0e3 * m, H = 500.0 * m), InjectionInterval_year = 500, nTr_dike = 30)
            Random.seed!(1234)
            _, Arrays, _, _, _ = redirect_stdout(devnull) do
                MTK_GeoParams(materials(tab_pressure), Num, Dikes)
            end
            return Array(Arrays.T), Array(Arrays.ϕ)
        end
        T_c, ϕ_c = run(CPU())
        T_j, ϕ_j = run(JLBackend())
        @test maximum(ϕ_c) > 0.1                                   # the sill (1000 °C) is partly molten
        @test T_j ≈ T_c rtol = 1.0e-10
        @test ϕ_j ≈ ϕ_c rtol = 1.0e-8 atol = 1.0e-10
    end
end

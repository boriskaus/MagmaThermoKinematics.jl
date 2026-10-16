using Test
using MagmaThermoKinematics, GeoParams, JLArrays
using MagmaThermoKinematics: DEFAULT_CONVECTION_VISCOSITY, convective_viscosity
JLArrays.allowscalar(false)
MagmaThermoKinematics.KernelAbstractions.synchronize(::JLArrays.JLBackend) = nothing

@testset "convective_nusselt" begin
    ρ, κ, η, ΔT, L = 2500.0, 1.0e-6, 1.0e5, 100.0, 1.0e3
    Ra = ρ * 9.81 * 3.0e-5 * ΔT * L^3 / (κ * η)
    x = log10(Ra)
    @test convective_nusselt(ΔT, L, ρ, κ, η; Nu_max = Inf) ≈ 10^(0.0087 * x^2 + 0.1350 * x - 0.2943)
    @test convective_nusselt(ΔT, L, ρ, κ, η; coeffs = (0.0, 1 / 3, log10(0.089)), Nu_max = Inf) ≈ 0.089 * cbrt(Ra)      # other fits, e.g. the Ra^(1/3) law
    @test convective_nusselt(0.0, L, ρ, κ, η) == 1.0                         # no temperature contrast: conduction
    @test convective_nusselt(ΔT, 1.0, ρ, κ, η) == 1.0                        # thin layer: below the onset
    @test convective_nusselt(ΔT, L, ρ, κ, η; Nu_max = 10.0) == 10.0          # bounded
    @test convective_nusselt(ΔT, L, ρ, κ, 10η; Nu_max = Inf) < convective_nusselt(ΔT, L, ρ, κ, η; Nu_max = Inf)
    @test convective_nusselt(Float32(ΔT), Float32(L), Float32(ρ), Float32(κ), Float32(η)) isa Float32
end

@testset "viscosity of the melt-rich region" begin
    law = DEFAULT_CONVECTION_VISCOSITY
    η(T, ϕ) = convective_viscosity(law, T, ϕ, 1.0e-12)
    @test η(1100.0, 1.0) > 0
    @test η(1100.0, 0.6) > η(1100.0, 0.8) > η(1100.0, 1.0)                     # crystals stiffen the magma
    @test η(1300.0, 0.8) < η(1100.0, 0.8)                                      # melt viscosity falls with T
    melt = GiordanoMeltViscosity(oxd_wt = (72.0, 0.3, 14.0, 2.0, 0.4, 1.5, 3.5, 3.5, 6.0))
    @test η(1100.0, 1.0) ≈ compute_τII(melt, 1.0e-12; T = 1100.0) / 2.0e-12 rtol = 1.0e-6   # no crystals: the melt viscosity
    @test convective_viscosity(LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.96K), 1100.0, 0.7, 1.0e-12) ≈ 10^(-8.159 + 2.405e4 / (1100.0 + 430.96)) rtol = 1.0e-6
end

# 2D (or 3D) arrays with two melt-rich runs in the first columns, a mush column, and an air cell
function convection_arrays(dims; FT = Float64)
    nz = dims[end]
    Kc = fill(FT(3), dims); T = fill(FT(900 + 273.15), dims); ϕ = fill(FT(0.2), dims); Rho = fill(FT(2500), dims); Cp = fill(FT(1100), dims)
    Phases = ones(Int, dims)
    col(i) = ntuple(d -> d == length(dims) ? Colon() : i, length(dims))
    for z in 5:14
        ϕ[ntuple(d -> d == length(dims) ? z : 1, length(dims))...] = 0.8                 # first column: one run of 10 cells
    end
    for z in (5:9, 12:20)
        for k in z
            ϕ[ntuple(d -> d == length(dims) ? k : 2, length(dims))...] = 0.8                 # second column: two runs separated by a mush cell
        end
    end
    ϕ[ntuple(d -> d == length(dims) ? 10 : 2, length(dims))...] = 0.2
    ϕ[ntuple(d -> d == length(dims) ? 11 : 2, length(dims))...] = 0.2
    T[ntuple(d -> d == length(dims) ? 14 : 1, length(dims))...] += 80                           # temperature contrast in the first run
    Phases[ntuple(d -> d == length(dims) ? 5 : 1, length(dims))...] = 0                         # an air cell at the bottom of the first run
    Z = zeros(FT, dims); for k in 1:nz; selectdim(Z, length(dims), k) .= FT(-(nz - k) * 50); end        # z is negative downward, index 1 at the bottom
    return (; Kc, T, ϕ, Rho, Cp, Phases, Z)
end

# Nusselt numbers of the arrays of `convection_arrays`, optionally with another viscosity law
function nusselt_of(A, visc = DEFAULT_CONVECTION_VISCOSITY; kw...)
    Nu = similar(A.Kc)
    compute_convective_nusselt!(Nu, A.Kc, A.T, A.ϕ, A.Rho, A.Cp, A.Phases, A.Z, 50.0, visc; kw...)
    return Nu
end

@testset "compute_convective_nusselt!" begin
    A = convection_arrays((6, 30)); Nu = nusselt_of(A)
    @test all(Nu[3:end, :] .== 1.0)                                                            # no melt: conduction
    @test all(Nu[1, 15:end] .== 1.0) && all(Nu[1, 1:4] .== 1.0)                               # outside the run
    @test Nu[1, 5] == 1.0                                                                       # air cell excluded
    @test all(Nu[1, 6:14] .> 1.0) && all(Nu[1, 6:14] .== Nu[1, 6])                              # the run convects as one
    @test all(Nu[2, 5:9] .== Nu[2, 5]) && all(Nu[2, 12:20] .== Nu[2, 12])                       # two separate runs in column 2
    @test Nu[2, 10] == 1.0 && Nu[2, 11] == 1.0
    # no temperature contrast: conduction only
    B = convection_arrays((6, 30)); B.T .= 1173.15
    @test all(nusselt_of(B) .== 1.0)
    # bounded by Nu_max
    @test maximum(nusselt_of(convection_arrays((6, 30)); Nu_max = 7.0)) ≤ 7.0
    # stiffer magma convects less
    @test nusselt_of(A, LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.96K))[1, 10] ≤ Nu[1, 10]
    # the value is the Nusselt number of the run: thickness 9 cells (air cell excluded), contrast 80 K
    η = convective_viscosity(DEFAULT_CONVECTION_VISCOSITY, 1173.15 + 80 / 9, 0.8, 1.0e-12)
    @test Nu[1, 10] ≈ convective_nusselt(80.0, 9 * 50.0, 2500.0, 3.0 / (2500.0 * 1100.0), η) rtol = 1.0e-10
end

@testset "two chambers with their own viscosity laws" begin
    A = convection_arrays((6, 30))
    # column 2 holds two chambers (cells 5:9 and 12:20); the lower one (cell index < 10) and the upper one differ in composition
    rhy = DEFAULT_CONVECTION_VISCOSITY
    mafic = GiordanoMeltViscosity()                                          # basaltic melt, much less viscous
    zsplit = A.Z[2, 10]
    N_rr = nusselt_of(A, rhy); N_mm = nusselt_of(A, mafic)
    N_tuple = nusselt_of(A, (rhy, mafic); z_split = (zsplit,))
    @test all(N_tuple[2, 12:20] .== N_rr[2, 12:20])                           # upper chamber: first law
    @test all(N_tuple[2, 5:9] .== N_mm[2, 5:9])                               # lower chamber: second law
    @test N_mm[2, 5] ≥ N_rr[2, 5]                                             # the less viscous magma convects at least as vigorously
    @test all(N_tuple[3:end, :] .== 1.0)
    @test_throws ArgumentError nusselt_of(A, (rhy, mafic))                    # a split is needed per additional law
    @test_throws ArgumentError nusselt_of(A, (rhy, mafic); z_split = (1.0, 2.0))
    @test_throws ArgumentError nusselt_of(A, (rhy, mafic, rhy); z_split = (-3000.0, -1000.0))     # splits must descend
end

@testset "compute_convective_nusselt! in 3D and on JLArrays" begin
    A = convection_arrays((4, 4, 30)); Nu = nusselt_of(A)
    @test all(Nu[1, 1, 6:14] .> 1.0) && all(Nu[3:end, 3:end, :] .== 1.0)
    a = convection_arrays((6, 30)); Na = nusselt_of(a)
    j = map(x -> JLArray(x), convection_arrays((6, 30))); Nj = similar(j.Kc)
    compute_convective_nusselt!(Nj, j.Kc, j.T, j.ϕ, j.Rho, j.Cp, j.Phases, j.Z, 50.0)
    @test Array(Nj) ≈ Na
    f = convection_arrays((6, 30); FT = Float32); Nf = nusselt_of(f)
    @test eltype(Nf) == Float32
    @test Nf ≈ Na rtol = 1.0e-3
end

const Mat_conv = (
    SetMaterialParams(
        Phase = 1, Density = ConstantDensity(ρ = 2500kg / m^3), LatentHeat = ConstantLatentHeat(Q_L = 2.5e5J / kg),
        Conductivity = ConstantConductivity(k = 3Watt / K / m), HeatCapacity = ConstantHeatCapacity(Cp = 1100J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())
    ),
)

# a hot melt-rich slab in a cold half-space, cooled by `nsteps` implicit steps
function slab_cooling(convection; nsteps = 3, dt = 3.0e10)
    N = (5, 61); Grid = CreateGrid(size = N, extent = (4.0e3, 6.0e3))
    names = (:T, :T_K, :Tnew, :T_init, :T_it_old, :Tupdate, :Kc, :Rho, :Cp, :Hr, :Hq, :Qsrc, :Qrelax, :Qenth, :Nu, :Hl, :ϕ, :dϕdT, :R, :Z, :P, :pt_C, :pt_b, :pt_phiold, :pt_res, :pt_Tk, :pt_Tkm1, :pt_Tn)
    Arrays = CreateArrays(Dict(N => NamedTuple{names}(ntuple(_ -> 0, length(names)))))
    GridArray!(Arrays.R, Arrays.Z, Grid)
    Arrays.T .= 300.0
    Arrays.T[:, 25:36] .= 1000.0                                   # a 1 km slab of melt in the middle
    Arrays.T[:, 36:37] .+= 60.0                                    # hotter layer at its top: a contrast across the slab
    Arrays.Tnew .= Arrays.T
    Num = NumParam(Nx = 5, Nz = 61, solver = :implicit_enthalpy, convection = convection, convergence = 1.0e-6, max_iter = 200)
    Phases = ones(Int64, N)
    Numax = 1.0
    for _ in 1:nsteps
        Nonlinear_Diffusion_step!(Arrays, Mat_conv, Phases, Grid, dt, Num)
        Arrays.T .= Arrays.Tnew
        convection && (Numax = max(Numax, maximum(Arrays.Nu)))
    end
    return Arrays, Numax
end

@testset "convecting slab loses its heat faster" begin
    A0, _ = slab_cooling(false); A1, Numax = slab_cooling(true)
    @test all(isfinite, A1.T)
    @test Numax > 1.5                                               # the slab convects while it is molten
    @test maximum(A1.T) < maximum(A0.T)                              # heat leaves the hot core faster
    # the interior is more homogeneous than the conductive one
    mid = 28:33
    @test maximum(A1.T[3, mid]) - minimum(A1.T[3, mid]) ≤ maximum(A0.T[3, mid]) - minimum(A0.T[3, mid]) + 1.0e-6
end

@testset "convection needs the implicit solver" begin
    @test NumParam().convection === false
    A, _ = slab_cooling(false; nsteps = 0)

    Grid = CreateGrid(size = (5, 61), extent = (4.0e3, 6.0e3)); Phases = ones(Int64, (5, 61))
    Num = NumParam(Nx = 5, Nz = 61, solver = :explicit, convection = true)
    @test_throws ArgumentError Nonlinear_Diffusion_step!(A, Mat_conv, Phases, Grid, 1.0e8, Num)
end

@testset "convection needs the Nu array" begin
    A, _ = slab_cooling(false; nsteps = 0); B = Base.structdiff(A, NamedTuple{(:Nu,)}((A.Nu,)))
    Grid = CreateGrid(size = (5, 61), extent = (4.0e3, 6.0e3)); Phases = ones(Int64, (5, 61))
    Num = NumParam(Nx = 5, Nz = 61, solver = :implicit_enthalpy, convection = true)
    @test_throws ArgumentError Nonlinear_Diffusion_step!(B, Mat_conv, Phases, Grid, 1.0e8, Num)
end

@testset "convecting chamber in a simulation and its VTK output" begin
    dir = mktempdir()
    Num = NumParam(
        Nx = 33, Nz = 33, SimName = "conv", maxTime_Myrs = 0.006, fac_dt = 0.2, ω = 0.5, Geotherm = 30 / 1.0e3, solver = :implicit_enthalpy,
        convection = true, SaveOutput_steps = 10, Output_VTK = true, CreateFig_steps = 100000, verbose = false
    )
    sill = CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 8.0e3 * m, H = 500.0 * m)
    Dikes = SillParams(sill = sill, InjectionInterval_year = 500, nTr_dike = 30)
    Mat = (
        SetMaterialParams(
            Name = "Rock", Phase = 1, Density = ConstantDensity(ρ = 2700kg / m^3), LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
            Conductivity = ConstantConductivity(k = 3Watt / K / m), HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K), Melting = SmoothMelting(MeltingParam_4thOrder())
        ),
    )
    Grid, Arrays, Tracers, Dikes, time_props = cd(dir) do
        redirect_stdout(devnull) do
            MTK_GeoParams(Mat, Num, Dikes)
        end
    end
    @test all(isfinite, Arrays.Tnew)
    vts = filter(endswith(".vts"), readdir(joinpath(dir, Num.SimName); join = true))
    @test !isempty(vts)
    txt = read(last(sort(vts)), String)
    @test occursin("Nusselt", txt) && occursin("Convecting", txt)
end

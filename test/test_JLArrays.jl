# Runs the solver on JLArrays, the GPUArrays reference array type that lives in host
# memory and has its own KernelAbstractions backend. With scalar indexing disallowed,
# any code path that would index a GPU array element by element throws here. Fields
# must equal those of the CPU backend exactly; reductions over them may differ in the
# last bits, because GPUArrays sums in a different order than Base.
using Test, Random
using MagmaThermoKinematics
using JLArrays
JLArrays.allowscalar(false)
# JLBackend runs kernels synchronously but has no `synchronize` method, which InjectSills calls.
MagmaThermoKinematics.KernelAbstractions.synchronize(::JLArrays.JLBackend) = nothing

const Mat_JL = (
    SetMaterialParams(
        Name = "Rock", Phase = 1,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
        Conductivity = T_Conductivity_Whittington_parameterised(),
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())
    ),
    SetMaterialParams(
        Name = "Sill", Phase = 2,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 3.5e5J / kg),
        Conductivity = ConstantConductivity(k = 2.5Watt / K / m),
        HeatCapacity = ConstantHeatCapacity(Cp = 1050J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())
    ),
)

"Run `MTK_GeoParams` for a few time steps on `backend`; returns host copies of the results."
function run_model(backend, dim)
    if dim == 2
        Num = NumParam(;
            Nx = 33, Nz = 33, SimName = mktempdir(), maxTime_Myrs = 0.003, fac_dt = 0.2, ω = 0.5,
            Geotherm = 30 / 1.0e3, Output_VTK = false, backend
        )
        sill = CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 20.0e3 * m, H = 500.0 * m)
    else
        Num = NumParam(;
            Nx = 13, Ny = 13, Nz = 13, W = 20.0e3, L = 20.0e3, H = 20.0e3, SimName = mktempdir(), maxTime_Myrs = 0.02,
            fac_dt = 0.2, ω = 0.5, Geotherm = 30 / 1.0e3, Output_VTK = false, backend
        )
        sill = EllipticalIntrusion(Center = Point3(0.0, 0.0, -7000.0) * m, Angle = Vec2(0.0, 0.0) * NoUnits, W = 5.0e3 * m, H = 800.0 * m)
    end
    Dikes = SillParams(sill = sill, InjectionInterval_year = dim == 2 ? 500 : 5000, nTr_dike = 30)
    Random.seed!(1234)      # new tracers are placed at random positions
    Grid, Arrays, Tracers, Dikes, time_props = redirect_stdout(devnull) do
        MTK_GeoParams(Mat_JL, Num, Dikes)
    end
    return (;
        T = Array(Arrays.T), ϕ = Array(Arrays.ϕ), Phases = Array(Arrays.Phases), on_jl = Arrays.T isa JLArray,
        MeltFraction = time_props.MeltFraction, InjectVol = Dikes.InjectVol,
    )
end

"One nonlinear diffusion step of a random temperature field on `backend`; returns `Tnew` on the host."
function diffusion_step(backend, N, T0; axisymmetric = false)
    Grid = CreateGrid(size = N, extent = ntuple(_ -> 20.0e3, length(N)))
    names = (:T, :T_K, :Tnew, :T_it_old, :Tupdate, :Kc, :Rho, :Cp, :Hr, :Hl, :ϕ, :dϕdT, :R, :Y, :Z, :P)
    Arrays = CreateArrays(Dict(N => NamedTuple{names}(ntuple(_ -> 0, length(names)))); backend)
    length(N) == 2 ? GridArray!(Arrays.R, Arrays.Z, Grid) : GridArray!(Arrays.R, Arrays.Y, Arrays.Z, Grid)
    copyto!(Arrays.T, T0)
    Arrays.Tnew .= Arrays.T
    Phases = similar(Arrays.T, Int64)
    fill!(Phases, 1)
    Nonlinear_Diffusion_step!(Arrays, Mat_JL, Phases, Grid, 1.0e10, Numeric_params(; axisymmetric))
    return Array(Arrays.Tnew)
end

@testset "JLArrays backend" begin
    @testset "diffusion step $(length(N))D$(axisymmetric ? " axisymmetric" : "")" for (N, axisymmetric) in
        (((17, 17), false), ((17, 17), true), ((9, 9, 9), false))
        T0 = 900 .* rand(N...)
        @test diffusion_step(JLBackend(), N, T0; axisymmetric) == diffusion_step(CPU(), N, T0; axisymmetric)
    end

    @testset "lithostatic pressure" begin
        Rho = 2000 .+ 100 .* rand(6, 7, 8)
        P = zero(Rho)
        lithostatic_pressure!(P, Rho, 9.81, 50.0)
        Pjl = JLArray(zero(Rho))
        lithostatic_pressure!(Pjl, JLArray(Rho), 9.81, 50.0)
        @test Array(Pjl) == P
    end

    @testset "host/device copies" begin
        T, ϕ = rand(4, 5), rand(4, 5)
        T_dev, ϕ_dev = JLArray(zeros(Float32, 4, 5)), JLArray(zeros(4, 5))
        copy_arrays_CPU2GPU!(T_dev, ϕ_dev, T, ϕ)
        @test Array(T_dev) == Float32.(T)        # converted to the device eltype
        @test Array(ϕ_dev) == ϕ
        T_host, ϕ_host = zeros(4, 5), zeros(4, 5)           # eltype change for T, none for ϕ
        copy_arrays_GPU2CPU!(T_host, ϕ_host, T_dev, ϕ_dev)
        @test T_host == Float32.(T)
        @test ϕ_host == ϕ
        @test_throws "cannot copy an array of size (4, 5) into one of size (5, 4)" copy_arrays_GPU2CPU!(zeros(5, 4), ϕ_host, T_dev, ϕ_dev)
        @test_throws "cannot copy an array of size (4, 5) into one of size (5, 4)" copy_arrays_CPU2GPU!(JLArray(zeros(5, 4)), ϕ_dev, T, ϕ)
    end

    @testset "MTK_GeoParams $(dim)D" for dim in (2, 3)
        jl, cpu = run_model(JLBackend(), dim), run_model(CPU(), dim)
        @test jl.on_jl
        @test cpu.InjectVol > 0
        @test any(==(2), cpu.Phases)       # the phase update after injection ran
        @test jl.InjectVol == cpu.InjectVol
        @test jl.T == cpu.T
        @test jl.ϕ == cpu.ϕ
        @test jl.Phases == cpu.Phases
        @test jl.MeltFraction ≈ cpu.MeltFraction rtol = 1.0e-12    # GPU reductions sum in a different order
    end

    @testset "chamber step on the device" begin
        using MagmaThermoKinematics: chamber_step!, chamber_cells, EruptionParams, EruptionState
        ep = EruptionParams(
            m_w = 0.06, ΔP_crit = 3.0e6, σ_barrier = 0.0, w_dike = 20.0,
            melt_viscosity = LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.9606K, η0 = 1Pas), solubility = Liu2005_Solubility(), shell_from_T = true
        )
        for coords in (
                (range(-5.0e3, 5.0e3, length = 41), range(-10.0e3, 0.0, length = 41)),
                (refined_coordinates(-5.0e3, 5.0e3, 41; zone = (-2.5e3, 2.5e3), ratio = 2, transition = 0), refined_coordinates(-10.0e3, 0.0, 41; zone = (-7.0e3, -3.0e3), ratio = 2, transition = 0)),
            )
            x, z = coords
            ϕ = [abs(xx) < 2.0e3 && -6.0e3 < zz < -4.0e3 ? 0.8 : 0.0 for xx in x, zz in z]; ϕ[2, 2] = 0.9      # and a one-cell region
            T = [ϕ[i, j] > 0 ? 900.0 + 0.01 * zz : 300.0 for (i, xx) in enumerate(x), (j, zz) in enumerate(z)]
            @test Set(chamber_cells(JLArray(ϕ), 0.5)) == Set(chamber_cells(ϕ, 0.5))          # the same cells (flood fill vs. index order)
            sh, sd = EruptionState(), EruptionState()
            run!(st, ϕ, T) = chamber_step!(st, ep, ϕ, T, coords; recharge = 1.0e-3, Δt = 3.15576e9, time = 0.0, T_background = d -> 20.0 + 0.02d)
            run!(sh, ϕ, T); run!(sd, JLArray(ϕ), JLArray(T))
            @test sd.V ≈ sh.V rtol = 1.0e-12
            @test sd.P ≈ sh.P rtol = 1.0e-12
            @test sd.P_lith ≈ sh.P_lith rtol = 1.0e-12
            @test Set(sd.cells) == Set(sh.cells)
            @test sd.η_r ≈ sh.η_r rtol = 1.0e-12                                           # shell viscosity from the temperature field, gathered on the device
            @test run!(sd, JLArray(zero(ϕ)), JLArray(T)) === nothing && !sd.init          # no chamber left on the device either
        end
    end

    @testset "revolved eruption sink and inflation" begin
        using MagmaThermoKinematics: StructArray, Tracer, chamber_geometry, revolved_inflation_rate
        dx = 100.0; x = range(-3000.0, 3000.0, step = dx); z = range(-6000.0, 500.0, step = dx)
        cells = [CartesianIndex(i, j) for i in eachindex(x), j in eachindex(z) if abs(x[i]) <= 4 * dx && -3500.0 <= z[j] <= -2800.0]
        event = (; cells, f = 0.2, geo = chamber_geometry(cells, (dx, dx), (x, z)))
        T0 = [zz < 0 ? 700.0 + 0.1 * abs(zz) + 50 * exp(-((xx / 800)^2 + ((zz + 3000) / 500)^2)) : 10.0 for xx in x, zz in z]
        tr = StructArray([Tracer{Float32}(num = 1, coord = [x[10], -3000.0], Phase = 1)])
        Tcpu = copy(T0); Ucpu, _ = erupt!(Tcpu, similar(Tcpu), tr, (x, z), event; revolve = true, nθ = 8, rng = MersenneTwister(1))
        Tjl = JLArray(copy(T0)); Ujl, _ = erupt!(Tjl, similar(Tjl), tr, (x, z), event; revolve = true, nθ = 8, rng = MersenneTwister(1))
        @test Ujl[1] isa JLArray
        @test Array(Ujl[1]) ≈ Ucpu[1] rtol = 1.0e-10 atol = 1.0e-12
        @test Array(Ujl[2]) ≈ Ucpu[2] rtol = 1.0e-10
        @test Array(Tjl) ≈ Tcpu rtol = 1.0e-10
        Xc = ([xx for xx in x, zz in z], [zz for xx in x, zz in z]); Xj = map(JLArray, Xc)
        rate_c = revolved_inflation_rate(Xc, [0.0, 0.0], [300.0, 500.0], [-3000.0, -3200.0], [600.0, 800.0], 1.0e9; thickness = 10.0, nθ = 4)
        rate_j = revolved_inflation_rate(Xj, [0.0, 0.0], [300.0, 500.0], [-3000.0, -3200.0], [600.0, 800.0], 1.0e9; thickness = 10.0, nθ = 4)
        @test Array(rate_j[2]) ≈ rate_c[2] rtol = 1.0e-10
        @test Array(rate_j[1]) ≈ rate_c[1] rtol = 1.0e-10 atol = 1.0e-12 * maximum(abs, rate_c[1])
    end
end

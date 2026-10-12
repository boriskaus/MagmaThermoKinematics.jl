using Test, Random
using InjectSills
using MagmaThermoKinematics
using GeophysicalModelGenerator, GeoParams

import MagmaThermoKinematics.MTK_GMG

# Small 2D simulations used to test how a simulation is configured and run
function small_setup(; random_sills = false, simname = joinpath(mktempdir(), "sim"), kwargs...)
    Num = NumParam(
        Nx = 33, Nz = 33, SimName = simname, maxTime_Myrs = 0.03, fac_dt = 0.4, ω = 0.5, verbose = false,
        Geotherm = 30 / 1.0e3, SaveOutput_steps = 10^6, CreateFig_steps = 10^6, plot_tracers = false,
        AddRandomSills = random_sills, RandomSills_timestep = 5; kwargs...
    )
    sill = random_sills ?
        InjectSills.PennyShapedSill(Center = Point2(0.0, -7.0e3) * m, R = 2.0e3 * m, H = 200 * m, E = 1.5e10 * Pa, ν = 0.3 * NoUnits) :
        InjectSills.CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 10.0e3 * m, H = 74.6269 * m)
    Dikes = SillParams(
        sill = sill, InjectionInterval_year = 4000, nTr_dike = 50, H_ran = 3000, W_ran = 3000, Dip_ran = 45, SillPhase = 1, SillsAbove = -10.0e3
    )
    if random_sills
        Dikes.InjectionInterval = Num.dt * Num.RandomSills_timestep; Dikes.InjectionInterval_year = Dikes.InjectionInterval / SecYear
    end
    Mat = (
        SetMaterialParams(
            Name = "Rock", Phase = 1, Density = ConstantDensity(ρ = 2700kg / m^3), LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
            Conductivity = ConstantConductivity(k = 3.0Watt / K / m), HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K), Melting = SmoothMelting(MeltingParam_4thOrder())
        ),
    )
    return Mat, Num, Dikes
end

function captured_stdout(f)
    mktemp() do path, io
        redirect_stdout(f, io); flush(io)
        read(path, String)
    end
end

@testset "Simulation set-up and re-entrancy" begin
    @testset "per-simulation callbacks" begin
        Mat, Num, Dikes = small_setup(print_injections = false)
        calls = Ref(0); times = Float64[]
        cb = (;
            print_output = (Grid, Num, Arrays, Mat_tup, Dikes) -> (calls[] += 1; nothing),
            update_TimeDepProps! = (tp, Grid, Num, Arrays, Mat_tup, Dikes) -> (push!(times, Num.time); nothing),
        )
        Grid, Arrays, Tracers, D, time_props = MTK_GeoParams(Mat, Num, Dikes; callbacks = cb)
        @test calls[] == Num.nt
        @test length(times) == Num.nt && times[end] ≈ Num.time
        @test isempty(time_props.Time_vec)                       # the default recording was replaced for this simulation only
        # a second simulation without callbacks uses the defaults again
        Mat2, Num2, Dikes2 = small_setup(print_injections = false)
        _, _, _, _, tp2 = MTK_GeoParams(Mat2, Num2, Dikes2)
        @test length(tp2.Time_vec) == Num2.nt
        @test_throws ArgumentError MTK_GeoParams(Mat2, copy_params(Num2), copy_params(Dikes2); callbacks = (; not_a_hook = identity))
    end

    @testset "output directory only when output is written" begin
        dir = joinpath(mktempdir(), "no_output")
        Mat, Num, Dikes = small_setup(simname = dir, Output_VTK = false, print_injections = false)
        MTK_GeoParams(Mat, Num, Dikes)
        @test !isdir(dir)
        dir2 = joinpath(mktempdir(), "with_output")
        Mat, Num, Dikes = small_setup(simname = dir2, print_injections = false)       # default: Output_VTK = true
        MTK_GeoParams(Mat, Num, Dikes)
        @test isdir(dir2)
    end

    @testset "injection printing can be switched off" begin
        Mat, Num, Dikes = small_setup(print_injections = true)
        @test occursin("Added new dike", captured_stdout(() -> MTK_GeoParams(Mat, Num, Dikes)))
        Mat, Num, Dikes = small_setup(print_injections = false)
        @test !occursin("Added new dike", captured_stdout(() -> MTK_GeoParams(Mat, Num, Dikes)))
    end

    @testset "random seed" begin
        function runT(seed)
            Mat, Num, Dikes = small_setup(random_sills = true, seed = seed, print_injections = false, Output_VTK = false)
            return copy(MTK_GeoParams(Mat, Num, Dikes)[2].Tnew)
        end
        T1, T2, T3 = runT(11), runT(11), runT(12)
        @test T1 == T2                      # same seed: identical random sills
        @test T1 != T3                      # different seed: different sills
    end

    @testset "copy_params" begin
        Mat, Num, Dikes = small_setup()
        N2 = copy_params(Num; Nx = 65, Nz = 65, fac_dt = 0.2)
        @test N2.Nx == 65 && N2.dx ≈ N2.W / 64 && N2.dz ≈ N2.H / 64
        @test N2.dt ≈ 0.2 * N2.Δmin^2 / N2.κ_time / 4
        @test N2.nt == floor(Int, N2.maxTime / N2.dt)
        @test Num.Nx == 33 && Num.dt ≈ 0.4 * Num.Δmin^2 / Num.κ_time / 4                # the original is untouched
        N3 = copy_params(Num; dt = 1.0e9)
        @test N3.dt == 1.0e9                                                            # explicit values win
        @test_throws ArgumentError copy_params(Num; not_a_field = 1)
        MTK_GeoParams(Mat, Num, Dikes)                                                  # run state accumulates in the parameters ...
        @test Num.it > 0 && Dikes.InjectVol > 0
        N4, D4 = copy_params(Num), copy_params(Dikes; T_in_Celsius = 1100.0, InjectionInterval_year = 2000)
        @test N4.it == 0 && N4.time == 0.0                                              # ... and a copy starts afresh
        @test D4.InjectVol == 0.0 && D4.sill_inj == 0.0 && isempty(D4.sill_poly)
        @test D4.T_in_Celsius == 1100.0 && D4.InjectionInterval ≈ 2000 * SecYear
        @test Dikes.T_in_Celsius == 1000.0
    end
end

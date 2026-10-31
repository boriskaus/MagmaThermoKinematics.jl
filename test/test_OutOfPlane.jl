using Test, Random
using MagmaThermoKinematics
using JLArrays
JLArrays.allowscalar(false)

# a uniform conducting rock, for the diffusion step of the integration test
const Mat_OOP = (
    SetMaterialParams(
        Name = "Rock", Phase = 1, Density = ConstantDensity(ρ = 2700kg / m^3), LatentHeat = ConstantLatentHeat(Q_L = 0.0J / kg),
        Conductivity = ConstantConductivity(k = 3.0Watt / K / m), HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K), Melting = SmoothMelting(MeltingParam_4thOrder())
    ),
)

@testset "out-of-plane heat loss" begin
    Nx, Nz = 241, 7; dx = 250.0
    x = ((1:Nx) .- (Nx + 1) / 2) .* dx; X = repeat(x, 1, Nz)

    @testset "width of a Gaussian anomaly" begin
        s = 3.0e3; θ = [400 * exp(-0.5 * (x[i] / s)^2) for i in 1:Nx, k in 1:Nz]
        L2 = out_of_plane_width2(θ, X)
        @test all(≈(s^2; rtol = 2.0e-3), L2)
        @test all(≈(2.0e3^2; rtol = 1.0e-12), out_of_plane_width2([400 * exp(-0.5 * (x[i] / 500)^2) for i in 1:Nx, k in 1:Nz], X; Lmin = 2.0e3))     # the lower bound
        @test all(==(1.0e14), out_of_plane_width2(zero(θ), X))                                                    # no anomaly, no loss
        θ2 = copy(θ); θ2[:, 4] .*= 3                                                                              # the smoothing leaves the width of a depth-independent shape unchanged
        @test all(≈(s^2; rtol = 2.0e-3), out_of_plane_width2(θ2, X; smooth = 2))
    end

    @testset "sink" begin
        s = 3.0e3; Tbg = fill(100.0, Nx, Nz); T = Tbg .+ [400 * exp(-0.5 * (x[i] / s)^2) for i in 1:Nx, k in 1:Nz]
        Kc = fill(3.0, Nx, Nz); Phases = ones(Int, Nx, Nz); Hr = zeros(Nx, Nz)
        add_out_of_plane_loss!(Hr, T, Tbg, Kc, X, Phases; factor = 1.0)
        @test Hr[(Nx + 1) ÷ 2, 3] ≈ -3.0 * 400 / s^2 rtol = 2.0e-3
        Hr2 = zeros(Nx, Nz); add_out_of_plane_loss!(Hr2, T, Tbg, Kc, X, Phases)
        @test Hr2 ≈ 1.5 .* Hr                                                                                    # the default factor
        Phases[:, 2] .= 0; Hr3 = zeros(Nx, Nz); add_out_of_plane_loss!(Hr3, T, Tbg, Kc, X, Phases; factor = 1.0)
        @test all(iszero, Hr3[:, 2])                                                                              # air cells are excluded
        Hr4 = zeros(Nx, Nz); add_out_of_plane_loss!(Hr4, Tbg, Tbg, Kc, X, ones(Int, Nx, Nz))
        @test all(iszero, Hr4)                                                                                    # nothing without an anomaly
    end

    @testset "decay of a Gaussian: the section reproduces the 3D solution" begin
        # a Gaussian anomaly of width s that is uniform in z and diffuses along x and y keeps its shape, with variance s² + 2κt and the amplitude s²/(s² + 2κt);
        # the planar section with the measured-width sink reproduces that on the section (the sink acts on the y-factor)
        K, ρ, cp = 3.0, 2700.0, 1000.0; κ = K / (ρ * cp); s = 3.0e3; θ0 = 400.0
        dt = 0.2 * dx^2 / κ; tend = 0.5 * s^2 / (2κ)
        T = [θ0 * exp(-0.5 * (x[i] / s)^2) for i in 1:Nx, k in 1:Nz]; Tn = copy(T)
        Kc = fill(K, Nx, Nz); Rho = fill(ρ, Nx, Nz); Cp = fill(cp, Nx, Nz); Hl = zeros(Nx, Nz); dϕdT = zeros(Nx, Nz); Tbg = zeros(Nx, Nz); Phases = ones(Int, Nx, Nz)
        nt = ceil(Int, tend / dt); dt = tend / nt
        for _ in 1:nt
            Hr = zeros(Nx, Nz)
            add_out_of_plane_loss!(Hr, T, Tbg, Kc, X, Phases; factor = 1.0)
            diffusion_step!(Tn, T, Kc, Rho, Cp, Hr, Hl, dt, (dx, dx), dϕdT)
            bc_zero_flux!(Tn, 2)                                                                                  # uniform in z
            T .= Tn
        end
        s2 = s^2 + 2κ * tend
        @test T[(Nx + 1) ÷ 2, 4] ≈ θ0 * s^2 / s2 rtol = 0.02
        @test out_of_plane_width2(T, X)[4] ≈ s2 rtol = 0.02
    end

    @testset "diffusion step: option and backends" begin
        N = (33, 33)
        function step(backend, out_of_plane; Tbg = true)
            Grid = CreateGrid(size = N, extent = (20.0e3, 20.0e3))
            names = (:T, :T_K, :Tnew, :T_it_old, :Tupdate, :Kc, :Rho, :Cp, :Hr, :Hl, :ϕ, :dϕdT, :R, :Z, :P, (Tbg ? (:Tbg,) : ())...)
            Arrays = CreateArrays(Dict(N => NamedTuple{names}(ntuple(_ -> 0, length(names)))); backend)
            GridArray!(Arrays.R, Arrays.Z, Grid)
            T0 = 20 .+ 300 .* exp.(-((Array(Arrays.R) .- 1.0e4) .^ 2 .+ (Array(Arrays.Z) .+ 1.0e4) .^ 2) ./ (2 * 2.5e3^2))
            copyto!(Arrays.T, T0); Arrays.Tnew .= Arrays.T; Arrays.Kc .= 3.0; Tbg && (Arrays.Tbg .= 20.0)     # Kc of the previous step enters the loss
            Phases = similar(Arrays.T, Int64); fill!(Phases, 1)
            Nonlinear_Diffusion_step!(Arrays, Mat_OOP, Phases, Grid, 1.0e11, Numeric_params(; out_of_plane))
            return Array(Arrays.Tnew)
        end
        Tnone = step(CPU(), false); Tloss = step(CPU(), true)
        @test maximum(Tloss) < maximum(Tnone)                                                                     # the loss cools the body
        @test step(JLArrays.JLBackend(), true) ≈ Tloss rtol = 1.0e-12
        @test_throws ArgumentError step(CPU(), true; Tbg = false)                                              # the background temperature is required
    end

    @testset "MTK_GeoParams run with the option" begin
        function run_model(out_of_plane)
            Num = NumParam(; Nx = 33, Nz = 33, SimName = mktempdir(), maxTime_Myrs = 0.003, fac_dt = 0.2, ω = 0.5, Geotherm = 30 / 1.0e3, Output_VTK = false, out_of_plane)
            Dikes = SillParams(sill = CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 20.0e3 * m, H = 500.0 * m), InjectionInterval_year = 500, nTr_dike = 30)
            Random.seed!(1234)
            Grid, Arrays, Tracers, Dikes, time_props = redirect_stdout(devnull) do
                MTK_GeoParams(Mat_OOP, Num, Dikes)
            end
            return Array(Arrays.T), Num.Tsurface_Celsius .- Num.Geotherm .* Array(Arrays.Z), haskey(Arrays, :Tbg) ? Array(Arrays.Tbg) : nothing
        end
        T0, _, Tbg0 = run_model(false); T1, geotherm, Tbg1 = run_model(true)
        @test all(isfinite, T1)
        @test Tbg0 === nothing                                                 # only allocated with the option
        @test Tbg1 ≈ geotherm && maximum(Tbg1) > 500                           # the background is the initial geotherm
        @test sum(T1) < sum(T0)                                                # heat is lost along the third dimension
    end
end

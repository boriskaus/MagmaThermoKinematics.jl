using Test
using MagmaThermoKinematics

const MatFloat32 = (
    SetMaterialParams(
        Name = "Rock", Phase = 1,
        Density = ConstantDensity(ρ = 2700kg / m^3),
        LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
        Conductivity = T_Conductivity_Whittington_parameterised(),
        HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
        Melting = SmoothMelting(MeltingParam_4thOrder())
    ),
)

"Run `nsteps` nonlinear diffusion steps of a hot blob in a geotherm on an `N`-cell grid; returns the state arrays."
function run_diffusion(N::Tuple, FT; nsteps = 3, dt = 1.0e9)
    dim = length(N)
    names = (:T, :T_K, :Tupdate, :Tnew, :T_it_old, :Kc, :Rho, :Cp, :Hr, :Hl, :ϕ, :dϕdT, :P, :Z)
    coords = dim == 2 ? (:X,) : (:X, :Y)
    Arrays = CreateArrays(Dict(N => NamedTuple{(names..., coords...)}(ntuple(_ -> 0, length(names) + length(coords)))); FloatType = FT)
    extent = ntuple(_ -> 20.0e3, dim)
    Grid = CreateGrid(size = N, extent = extent)
    GridArray!(map(k -> Arrays[k], (coords..., :Z))..., Grid)

    Arrays.T .= @. 10 - Arrays.Z * 0.03
    mid = ntuple(d -> (d == dim ? -10.0e3 : 10.0e3), dim)
    r2 = sum(ntuple(d -> ((d == dim ? Arrays.Z : (d == 1 ? Arrays.X : Arrays.Y)) .- mid[d]) .^ 2, dim))
    Arrays.T .= ifelse.(r2 .< (4.0e3)^2, FT(900), Arrays.T)
    Arrays.Tnew .= Arrays.T
    Phases = ones(Int64, N...)

    Num = Numeric_params()
    for _ in 1:nsteps
        Nonlinear_Diffusion_step!(Arrays, MatFloat32, Phases, Grid, dt, Num)
        Arrays.T .= Arrays.Tnew
    end
    return Arrays
end

@testset "Float32 on CPU" begin
    for N in ((21, 21), (11, 11, 11))
        A32 = run_diffusion(N, Float32)
        A64 = run_diffusion(N, Float64)
        @testset "$(length(N))D" begin
            for k in (:T, :Tnew, :Kc, :Rho, :Cp)
                @test A32[k] ≈ A64[k] rtol = 1.0e-4
            end
            # the 4th-order melting polynomial loses Float32 precision near the solidus
            @test A32.ϕ ≈ A64.ϕ rtol = 1.0e-3
        end
    end
end

@testset "Material laws evaluate in Float32" begin
    args = (; T = 1000.0f0, P = 0.0f0)
    for fn in (compute_meltfraction, compute_dϕdT, compute_density, compute_heatcapacity, compute_conductivity, compute_latent_heat)
        @test fn(MatFloat32, 1, args) isa Float32
    end
    argsA = (; T = fill(1000.0f0, 3, 3), P = zeros(Float32, 3, 3))
    A32, A64 = zeros(Float32, 3, 3), zeros(Float32, 3, 3)
    compute_phase_param!(A32, compute_conductivity, MatFloat32, ones(Int32, 3, 3), argsA)
    compute_phase_param!(A64, compute_conductivity, MatFloat32, ones(Int64, 3, 3), argsA)
    @test A32 == A64
    @test_throws "Phases must hold integer phase numbers" compute_phase_param!(A32, compute_density, MatFloat32, ones(Float32, 3, 3), argsA)
end

@testset "inject_sills stays Float32" begin
    using InjectSills, StructArrays
    x, z = range(0.0, 3.0e4, 33), range(-3.0e4, 0.0, 33)
    T = Float32[-zz / 1.0e3 * 20 for _ in x, zz in z]
    sill = PennyShapedSill(Center = Point2(1.5f4, -1.5f4) * InjectSills.m, Angle = Vec1(0.0f0), R = 5.0f3 * InjectSills.m, H = 500.0f0 * InjectSills.m, E = 1.5f10 * InjectSills.Pa, ν = 0.3f0 * InjectSills.NoUnits)
    Tr = StructArray{Tracer{Float32}}(undef, 1)
    _, Tnew, _, _, Vel = inject_sills(Tr, T, (x, z), sill, 900.0, 2, 10)
    @test eltype(Tnew) == Float32
    @test eltype(Vel[1]) == Float32
end

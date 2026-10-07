using Test
using MagmaThermoKinematics

const MatFloat32 = (SetMaterialParams(Name="Rock", Phase=1,
                        Density      = ConstantDensity(ρ=2700kg/m^3),
                        LatentHeat   = ConstantLatentHeat(Q_L=3.13e5J/kg),
                        Conductivity = T_Conductivity_Whittington_parameterised(),
                        HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                        Melting      = SmoothMelting(MeltingParam_4thOrder())),)

"Run `nsteps` nonlinear diffusion steps of a hot blob in a geotherm on an `N`-cell grid; returns the state arrays."
function run_diffusion(N::Tuple, FT; nsteps=3, dt=1e9)
    dim = length(N)
    names = (:T, :T_K, :Tupdate, :Tbuffer, :Tnew, :T_it_old, :Kc, :Rho, :Cp, :Hr, :Hl, :ϕ, :dϕdT, :P, :Z)
    coords = dim == 2 ? (:X,) : (:X, :Y)
    Arrays = CreateArrays(Dict(N => NamedTuple{(names..., coords...)}(ntuple(_ -> 0, length(names) + length(coords)))); FloatType=FT)
    extent = ntuple(_ -> 20e3, dim)
    Grid = CreateGrid(size=N, extent=extent)
    GridArray!(map(k -> Arrays[k], (coords..., :Z))..., Grid)

    Arrays.T .= @. 10 - Arrays.Z * 0.03
    mid = ntuple(d -> (d == dim ? -10e3 : 10e3), dim)
    r2 = sum(ntuple(d -> ((d == dim ? Arrays.Z : (d == 1 ? Arrays.X : Arrays.Y)) .- mid[d]) .^ 2, dim))
    Arrays.T .= ifelse.(r2 .< (4e3)^2, FT(900), Arrays.T)
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
            for k in (:T, :Tnew, :ϕ, :Kc, :Rho, :Cp)
                @test A32[k] ≈ A64[k] rtol=1e-4
            end
        end
    end
end

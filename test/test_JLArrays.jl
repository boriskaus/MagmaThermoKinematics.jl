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

const Mat_JL = (SetMaterialParams(Name="Rock", Phase=1,
                    Density      = ConstantDensity(ρ=2700kg/m^3),
                    LatentHeat   = ConstantLatentHeat(Q_L=3.13e5J/kg),
                    Conductivity = T_Conductivity_Whittington_parameterised(),
                    HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
                    Melting      = SmoothMelting(MeltingParam_4thOrder())),
                SetMaterialParams(Name="Sill", Phase=2,
                    Density      = ConstantDensity(ρ=2700kg/m^3),
                    LatentHeat   = ConstantLatentHeat(Q_L=3.5e5J/kg),
                    Conductivity = ConstantConductivity(k=2.5Watt/K/m),
                    HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
                    Melting      = SmoothMelting(MeltingParam_4thOrder())))

"Run `MTK_GeoParams` for a few time steps on `backend`; returns host copies of the results."
function run_model(backend, dim)
    if dim == 2
        Num  = NumParam(; Nx=33, Nz=33, SimName=mktempdir(), maxTime_Myrs=0.003, fac_dt=0.2, ω=0.5,
                        Geotherm=30/1e3, Output_VTK=false, backend)
        sill = CylindricalDikeTopAccretion(Center=Point2(0.0, -7.0e3) * m, W=20e3 * m, H=500.0 * m)
    else
        Num  = NumParam(; Nx=13, Ny=13, Nz=13, W=20e3, L=20e3, H=20e3, SimName=mktempdir(), maxTime_Myrs=0.02,
                        fac_dt=0.2, ω=0.5, Geotherm=30/1e3, Output_VTK=false, backend)
        sill = EllipticalIntrusion(Center=Point3(0.0, 0.0, -7000.0) * m, Angle=Vec2(0.0, 0.0) * NoUnits, W=5e3 * m, H=800.0 * m)
    end
    Dikes = SillParams(sill=sill, InjectionInterval_year=dim == 2 ? 500 : 5000, nTr_dike=30)
    Random.seed!(1234)      # new tracers are placed at random positions
    Grid, Arrays, Tracers, Dikes, time_props = redirect_stdout(devnull) do
        MTK_GeoParams(Mat_JL, Num, Dikes)
    end
    return (; T=Array(Arrays.T), ϕ=Array(Arrays.ϕ), Phases=Array(Arrays.Phases), on_jl=Arrays.T isa JLArray,
              MeltFraction=time_props.MeltFraction, InjectVol=Dikes.InjectVol)
end

"One nonlinear diffusion step of a random temperature field on `backend`; returns `Tnew` on the host."
function diffusion_step(backend, N, T0; axisymmetric=false)
    Grid   = CreateGrid(size=N, extent=ntuple(_ -> 20e3, length(N)))
    names  = (:T, :T_K, :Tnew, :T_it_old, :Tupdate, :Kc, :Rho, :Cp, :Hr, :Hl, :ϕ, :dϕdT, :R, :Y, :Z, :P)
    Arrays = CreateArrays(Dict(N => NamedTuple{names}(ntuple(_ -> 0, length(names)))); backend)
    length(N) == 2 ? GridArray!(Arrays.R, Arrays.Z, Grid) : GridArray!(Arrays.R, Arrays.Y, Arrays.Z, Grid)
    copyto!(Arrays.T, T0)
    Arrays.Tnew .= Arrays.T
    Phases = similar(Arrays.T, Int64)
    fill!(Phases, 1)
    Nonlinear_Diffusion_step!(Arrays, Mat_JL, Phases, Grid, 1e10, Numeric_params(; axisymmetric))
    return Array(Arrays.Tnew)
end

@testset "JLArrays backend" begin
    @testset "diffusion step $(length(N))D$(axisymmetric ? " axisymmetric" : "")" for (N, axisymmetric) in
            (((17, 17), false), ((17, 17), true), ((9, 9, 9), false))
        T0 = 900 .* rand(N...)
        @test diffusion_step(JLBackend(), N, T0; axisymmetric) == diffusion_step(CPU(), N, T0; axisymmetric)
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
        @test jl.MeltFraction ≈ cpu.MeltFraction rtol=1e-12    # GPU reductions sum in a different order
    end
end

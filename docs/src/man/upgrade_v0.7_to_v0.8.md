# Upgrading from v0.7 to v0.8

v0.8 replaces ParallelStencil with [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl). The same code now runs in 2D and 3D, on the CPU and on the GPU. The numerics are unchanged: on the CPU, v0.8 results agree with v0.7 to round-off (about 1e-11 °C in our test runs), except for the changes listed under [Changes without an error](#pitfalls). Injection and the nonlinear solver are faster and allocate less. Scripts need changes, because the backend setup, the `2D`/`3D` modules and the re-exported ParallelStencil macros are gone.

## Checklist

1. Remove `environment!(...)`, `using ParallelStencil...`, `@init_parallel_stencil(...)` and `using MagmaThermoKinematics.Diffusion2D` (or `Diffusion3D`, `Fields2D`, `Fields3D`, `MTK_GMG_2D`, `MTK_GMG_3D`). `using MagmaThermoKinematics` is enough.
2. Rename the functions listed in [Renamed functions](#renamed-functions). Most `_2D`/`_3D` names lost their suffix.
3. Replace `USE_GPU` with a backend: `NumParam(backend=CPU())` (the default) or, after `using CUDA`, `NumParam(backend=CUDABackend())`.
4. Replace `@parallel`, `@zeros`, `@ones` and `Data.Array` by plain Julia (see [Replacing ParallelStencil code](#replacing-parallelstencil-code)).
5. If you run on an NVIDIA GPU, add CUDA.jl to your own environment. It is no longer a dependency of MagmaThermoKinematics.
6. v0.8 requires GeoParams 0.9. Material laws evaluate in the precision of their input (`Float32` arrays give `Float32` results).
7. Plots, CairoMakie, MAT and TimerOutputs are no longer installed with MagmaThermoKinematics. Add the ones your scripts load to your environment. The examples plot with CairoMakie instead of Plots. `LoadPhaseDiagrams(...; PlotDiagrams=true)` plots with Makie and needs a backend (`using CairoMakie` or `using GLMakie`).

Quick check of a v0.7 script:

```bash
grep -nE "environment!|@parallel|@init_parallel_stencil|@zeros|@ones|Data\.Array|USE_GPU" my_script.jl
grep -nE "Diffusion[23]D|Fields[23]D|MTK_GMG_[23]D|_2D!|_3D!|_2D\(|_3D\(|bc[23]D_|assign!" my_script.jl
```

Every hit needs a change.

## Renamed functions

| v0.7 | v0.8 |
| --- | --- |
| `Nonlinear_Diffusion_step_2D!`, `Nonlinear_Diffusion_step_3D!` | `Nonlinear_Diffusion_step!` |
| `MTK_GMG_2D.MTK_GeoParams_2D`, `MTK_GMG_3D.MTK_GeoParams_3D` | `MTK_GeoParams` |
| `diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, H, Hl, dt, dx, dz, dϕdT)` | `diffusion_step!(Tnew, T, K, Rho, Cp, H, Hl, dt, (dx, dz), dϕdT)` |
| `diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, K, Kr, Kz, ...)` | `diffusion_step!(...; R)` |
| `diffusion3D_step_varK!(Tnew, T, qx, qy, qz, K, Kx, Ky, Kz, ...)` | `diffusion_step!(Tnew, T, K, Rho, Cp, H, Hl, dt, (dx, dy, dz), dϕdT)` |
| `bc2D_x!(T)`, `bc3D_x!(T)` | `bc_zero_flux!(T, 1)` |
| `bc2D_z!(T)`, `bc3D_y!(T)` | `bc_zero_flux!(T, 2)` |
| `bc2D_T!`, `bc3D_T!` | `bc_T!(Tnew, T)` |
| `bc2D_z_bottom_flux!`, `bc3D_z_bottom_flux!` | `bc_z_bottom_flux!(T, K, dz, q_z)` |
| `bc2D_z_bottom!(T)`, `bc3D_z_bottom!(T)` | `selectdim(T, ndims(T), 1) .= selectdim(T, ndims(T), 2)` |
| `compute_meltfraction_ps!`, `compute_density_ps!`, ... (and `_ps_3D!`) | `compute_phase_param!(A, compute_meltfraction, Mat_tup, Phases, args)` |
| `@parallel (...) GridArray!(X, Z, x, z)` | `GridArray!(X, Z, Grid)` |
| `@parallel assign!(A, B)` | `A .= B` |
| `@parallel assign!(A, B, c)` | `A .= B .+ c` |

`MTK_GeoParams` builds a 3D model if `Num.Ny > 0` or if `CartData_input` is 3D, and a 2D model otherwise. `diffusion_step!` no longer needs the face arrays `qx`, `qz`, `Kx`, `Kz`, `Rc`, ..., so you can drop them from `CreateArrays`. With `R` (the cell radii), the first dimension is radial.

## Replacing ParallelStencil code

v0.7:

```julia
const USE_GPU = false
using ParallelStencil, ParallelStencil.FiniteDifferences2D
using MagmaThermoKinematics
environment!(:cpu, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2)
using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D

Arrays = CreateArrays(Dict((Nx, Nz) => (T=0, Tnew=0, ...), (Nx-1, Nz) => (qx=0, Kx=0), (Nx, Nz-1) => (qz=0, Kz=0)))
Phases = ones(Int64, Nx, Nz)
@parallel (1:Nx, 1:Nz) GridArray!(Arrays.X, Arrays.Z, Grid.coord1D[1], Grid.coord1D[2])
...
Arrays.T .= Data.Array(Tnew_cpu)
Nonlinear_Diffusion_step_2D!(Arrays, MatParam, Phases, Grid, dt, Num)
@parallel assign!(Arrays.T, Arrays.Tnew)
```

v0.8:

```julia
using MagmaThermoKinematics
backend = CPU()            # or: using CUDA; backend = CUDABackend()

Arrays = CreateArrays(Dict((Nx, Nz) => (T=0, Tnew=0, ...)); backend)
Phases = similar(Arrays.T, Int64); fill!(Phases, 1)
GridArray!(Arrays.X, Arrays.Z, Grid)
...
copyto!(Arrays.T, Tnew_cpu)
Nonlinear_Diffusion_step!(Arrays, MatParam, Phases, Grid, dt, Num)
Arrays.T .= Arrays.Tnew
```

`CreateArrays` takes `backend` (default `CPU()`) and `FloatType` (default `Float64`). `NumParam` has the same two fields, which replace `USE_GPU`. On the CPU, set the number of threads with `julia -t auto`.

## Pitfalls

**Errors you will see**

- `UndefVarError` for `environment!`, `@parallel`, `@zeros`, `Data`, `Diffusion2D`, ...: the script still uses the v0.7 API.
- `Nonlinear_Diffusion_step!` throws an error if the Picard iterations do not converge within `max_iter`. In v0.7 it printed a warning and carried on with the unconverged temperature. Reduce `dt` or the relaxation parameter `ω`. A melting law whose `dϕ/dT` jumps (for example `MeltingParam_Assimilation()` at the liquidus) may need `SmoothMelting(...)`.
- `NumParam(USE_GPU=...)` fails, because the field no longer exists.
- `LoadPhaseDiagrams(names, true)` fails: `PlotDiagrams` is a keyword, `LoadPhaseDiagrams(names; PlotDiagrams=true)`.
- The `Arrays` returned by `MTK_GeoParams` no longer contain `qx`, `qz`, `Kx`, `Kz`, `Rc` (and `qy`, `Ky` in 3D). Callbacks that read them fail.
- A user-defined `MTK_inject_dikes` with the v0.7 signature `(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, Tnew_cpu)` is no longer called, because the solver calls `MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers)`. Drop the last argument and pass `Arrays.T` to `inject_sills`, which accepts arrays on any backend.

**Changes without an error**

- Models with more than one phase now update the phases from the tracers after each injection: cells filled by a sill take `SillPhase`, all others keep their initial phase (with `keep_init_RockPhases=true`, the default). In v0.7 `MTK_inject_dikes` wrote the new phases into a copy, so the sills kept the phase of the host rock. Results change if the sill phase has different material properties than the host rock.
- The default relaxation parameter of the nonlinear iterations is `ω = 0.5` (in `NumParam` and `Numeric_params`; v0.7 used 0.8, which does not converge at high resolution once a sill is present). Set `ω` explicitly to keep the v0.7 behavior.
- GeoParams 0.9 corrects the diffusivity coefficient of `T_Conductivity_Whittington` (567.3 instead of 576.3, as in Whittington et al., 2009). Models using this law change slightly (+0.17% total melt in our ZASSy test).
- The melt fraction `ϕ` is clamped to [0, 1]. Some melting laws, e.g. `SmoothMelting(MeltingParam_4thOrder())`, return values slightly outside this range (up to about 2e-4), which v0.7 kept.
- `inject_sills` moves existing tracers and the sill polygon from `x` to `x + u(x)`, with `u` the host-rock displacement at their own positions. v0.7 integrated the grid displacement field as a velocity over pseudo-time steps. Tracer positions shift by about a meter in typical 2D models and by up to the sill opening for tracers on the crack plane of a new sill; in our ZASSy test the total melt changes by −0.13%.
- In 3D, `inject_sills` advects the plotting polygon `dike_poly` as the x–z section through the sill center. In v0.7, 3D models with `advect_polygon=true` failed with a `BoundsError` at the first injection.
- `time_props.MeltFraction` is the mean melt fraction of the whole model in 3D. In v0.7 it was `Ny` times too large.
- A 2D `NumParam` with `Ny > 0` now runs a 3D model in `MTK_GeoParams`. Leave `Ny` at its default `0` for 2D models.
- `Numeric_params` has a new field `deactivationDepth` (default `-15e3` m). In v0.7, `deactivate_La_at_depth=true` with `Numeric_params` failed because this field was missing.

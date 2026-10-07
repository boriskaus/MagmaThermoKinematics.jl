# Upgrading from v0.6 to v0.8

Two things changed between v0.6 and v0.8:

- **v0.7:** dikes and sills come from [InjectSills.jl](https://github.com/JuliaGeodynamics/InjectSills.jl). The `Dike` API of MagmaThermoKinematics is gone.
- **v0.8:** ParallelStencil is replaced by KernelAbstractions. The `2D`/`3D` modules and `environment!` are gone. This step is described in [Upgrading from v0.7 to v0.8](upgrade_v0.7_to_v0.8.md). Do it after the steps below.

## Checklist

1. Replace every `Dike(...)` and `DikeParam(...)` with an InjectSills sill object and `SillParams` (see [Dikes become InjectSills sills](#dikes-become-injectsills-sills)).
2. Replace `InjectDike(...)` with `inject_sills(...)`, which also takes the temperature and phase of the magma.
3. In your `MTK_GMG` callbacks, change the type of the last argument from `DikeParameters` to `SillParameters`.
4. Rename the `DikeParam` fields `DikePhase`, `dike_poly`, `dike_inj` to `SillPhase`, `sill_poly`, `sill_inj`.
5. Apply the [v0.7 → v0.8 checklist](upgrade_v0.7_to_v0.8.md#checklist).

Quick check of a v0.6 script:

```bash
grep -nE "Dike\(|DikeParam|DikeParameters|InjectDike|AddDike|DikePoly|CreateDikePolygon|advect_dike_polygon|HostRockVelocityFromDike" my_script.jl
grep -nE "DikePhase|dike_poly|dike_inj|W_in|H_in|Type *=" my_script.jl
```

The second line also matches unrelated code. Check each hit.

## Dikes become InjectSills sills

The geometry of an intrusion is an InjectSills object with units. `SillParams` stores it in its `sill` field, together with the injection settings that `DikeParam` held.

| v0.6 `Type` | v0.8 sill type | size arguments |
| --- | --- | --- |
| `"ElasticDike"` | `PennyShapedSill` | `R = W/2` (radius), `H`, optional `E`, `ν` |
| `"EllipticalIntrusion"` | `EllipticalIntrusion` | `W` (full width), `H` |
| `"CylindricalDike_TopAccretion"` | `CylindricalDikeTopAccretion` | `W` (full width), `H` |
| `"CylindricalDike_TopAccretion_FullModelAdvection"` | `CylindricalDikeTopAccretionFullModelAdvection` | `W` (full width), `H` |
| `"SquareDike"` | `SquareDike` | `W` (full width), `H` |
| `"SquareDike_TopAccretion"` | `SquareDikeTopAccretion` | `W` (full width), `H` |

`Center` is a `Point2` (2D) or `Point3` (3D) and `Angle` a `Vec1` (2D, dip) or `Vec2` (3D, dip and strike). Lengths carry units (`m`), angles are `NoUnits`. InjectSills re-exports these names through MagmaThermoKinematics.

v0.6:

```julia
Dike_params = DikeParam(Type             = "ElasticDike",
                        W_in             = 5e3,
                        H_in             = 250,
                        Center           = [0.0, -7e3],
                        T_in_Celsius     = 1000,
                        InjectionInterval_year = 1000,
                        nTr_dike         = 300,
                        DikePhase        = 2)
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(MatParam, Num, Dike_params)
```

v0.8:

```julia
sill = PennyShapedSill(Center=Point2(0.0, -7e3)m, R=2.5e3m, H=250m, E=1.5e10Pa, ν=0.3NoUnits)
Sill_params = SillParams(sill                   = sill,
                         T_in_Celsius           = 1000,
                         InjectionInterval_year = 1000,
                         nTr_dike               = 300,
                         SillPhase              = 2)
Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params)
```

`W_in`, `H_in`, `Center`, `Angle`, `Type`, `AspectRatio`, `SillRadius` and `SillArea` are no longer fields of `SillParams`; they are part of `sill`. To move or rotate the sill during a run (for example in `MTK_update_ArraysStructs!`), use `InjectSills.update_abstractsill(Dikes.sill; Center=..., Angle=...)`.

## Injecting a sill in your own time loop

v0.6:

```julia
dike = Dike(W=W_in, H=H_in, Type="ElasticDike", T=T_in)
dike = Dike(dike, Center=cen, Angle=[Angle_rand])
Tracers, T_cpu, Vol = InjectDike(Tracers, T_cpu, Grid.coord1D, dike, nTr_dike)
```

v0.8:

```julia
sill = PennyShapedSill(Center=Point2(cen[1], cen[2])m, Angle=Vec1(Angle_rand)NoUnits, R=W_in/2*m, H=H_in*m)
Tracers, T_cpu, Vol, _, _ = inject_sills(Tracers, T_cpu, Grid.coord1D, sill, T_in, Phase_in, nTr_dike)
```

`inject_sills` returns the tracers, the temperature, the injected volume, the advected sill polygon (pass the previous one with the keyword `dike_poly`) and the displacement field. `AddDike(T, Tracers, Grid, dike, nTr)` becomes `add_dike(T, Tracers, Grid, sill, T_in, Phase_in, nTr)`, which stamps the sill without moving the host rock. The polygon helpers `CreateDikePolygon` and `advect_dike_polygon!` are replaced by `dike_polygon(sill)` and the `dike_poly` keyword. `HostRockVelocityFromDike` and `DikePoly` have no replacement.

## Results

The injected volumes use the same formulas as in v0.6 (`InjectSills.volume`). The host-rock displacement is computed by InjectSills, so results can differ slightly from v0.6. For the changes from v0.7 to v0.8, see [Upgrading from v0.7 to v0.8](upgrade_v0.7_to_v0.8.md#pitfalls).

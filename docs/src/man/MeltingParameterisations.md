# Melting Parameterizations

A number of melting parameterisations are implemented through GeoParams which can be set in the `MatParams` struct. For a complete list see [here](https://juliageodynamics.github.io/GeoParams.jl/dev/man/melting):

## Melt fraction from phase diagrams (MAGEMin)

Instead of a parameterised melting law, the melt fraction can be taken from a phase diagram computed for a given bulk composition with [MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin). GeoParams reads such a precomputed table with `MAGEMin_Diagram` and interpolates the melt fraction bilinearly in temperature and pressure:

```julia
using MagmaThermoKinematics

PD = MAGEMin_Diagram("Rhyolite.in")          # precomputed MAGEMin table
MatParam = (
    SetMaterialParams(Name = "Rock", Phase = 1,
        Density      = ConstantDensity(ρ = 2700kg/m^3),
        HeatCapacity = ConstantHeatCapacity(Cp = 1050J/kg/K),
        Conductivity = ConstantConductivity(k = 2.5Watt/K/m),
        LatentHeat   = ConstantLatentHeat(Q_L = 3.5e5J/kg),
        Melting      = PD),                  # melt fraction ϕ(T, P) from the table
)
Num = NumParam(solver = :implicit_enthalpy, ω = 1.0)   # the implicit solver is required, see below
```

Only the melt fraction (and its derivative ``\partial\phi/\partial T``) comes from the table; the latent heat, heat capacity, density and conductivity are those of the other laws of the phase. The table is evaluated at the temperature of every cell and at the lithostatic pressure that MagmaThermoKinematics computes from the density in every nonlinear iteration (`Arrays.P`, in Pa), so the melt fraction changes with depth as in the phase diagram.

**The table.** `MAGEMin_Diagram` reads a text file whose first 55 lines are a header: lines 1–48 are free comments, line 49 names the columns, and lines 50–55 give the lowest temperature [K], the temperature increment, the number of temperatures, the lowest pressure, the pressure increment and the number of pressures. The data follow with one row per (T, P) point, temperature varying fastest, and the columns in this order: melt density, melt fraction, solid density, Vp, Vs, specific heat capacity, temperature, pressure, for example

```
[meltRho, rho, kg/m^3] [meltFrac, wtPercent, NoUnits] [rockRho, rho, kg/m^3] [Vp, vp, km/s] [Vs, vs, km/s] [SpecificCp, scp, J/kg/K] [Temperature, T, K] [Pressure, P, Pa]
```

The table should cover the temperatures and pressures of the model: outside it the values of its edges are used. Since the melt fraction is interpolated linearly between the table points, the table should resolve the melting interval.

**Solver.** Use the implicit enthalpy solver (`NumParam(solver = :implicit_enthalpy)`, see [Implicit Enthalpy Solver](implicit_solver.md)). With linear interpolation, ``\partial\phi/\partial T`` is constant between table points and jumps at them, and the nonlinear iterations of the explicit solver, which uses ``\partial\phi/\partial T`` in an effective heat capacity, do not converge. The implicit solver balances the melt fraction itself and converges, usually with `ω = 1`. Computed phase diagrams can be very steep, for example when a large melt fraction appears within a few kelvin at the solidus; if the nonlinear iterations then do not converge, reduce the relaxation to `ω ≈ 0.3`.

**GPUs.** The table is an array, which GPU kernels can only read when it lives on the device. `MTK_GeoParams` copies the material parameters to the backend of the model (`Num.backend`) itself. When you call `Nonlinear_Diffusion_step!` directly with GPU arrays, copy them once with `materials_on_backend`:

```julia
MatParam_gpu = materials_on_backend(MatParam, CUDABackend())
```

**Tests.** `test/test_PhaseDiagram.jl` checks that a table computed from a melting law reproduces that law, also in a cooling sill with the implicit solver (temperature within 0.05 K); that the melt fraction follows the pressure dependence of the table at the model's lithostatic pressure; that a steep table converges with `ω = 0.3`; and that the results on the JLArrays backend equal those on the CPU, for a single step and for a full `MTK_GeoParams` simulation. The CUDA backend was checked separately against the CPU with this table and with a MAGEMin rhyolite table (differences below ``10^{-10}`` K).

# Implicit Enthalpy Solver

By default, the diffusion step of MagmaThermoKinematics is explicit (see [Numerics and Physics](numerics.md)), and its time step is bounded by the stability limit ``\Delta t \le \Delta^2/(2\kappa)``. With `NumParam(solver = :implicit_enthalpy)` (alias `:apt`) the diffusion and latent-heat step is instead solved implicitly, as a backward-Euler step of the enthalpy balance. The time step is then limited only by accuracy, so `fac_dt` can be much larger than 1. This pays off for long simulations, fine grids and strongly temperature-dependent latent heat. The implicit step runs on every KernelAbstractions backend (CPU, CUDA, ...), in 2D, axisymmetric and 3D models, and in single or double precision.

## Usage

```julia
using MagmaThermoKinematics

Num = NumParam(
    Nx = 269, Nz = 269, maxTime_Myrs = 1.0,
    solver = :implicit_enthalpy,   # or :apt
    fac_dt = 4.0,                  # time step = fac_dt · Δ²/κ/4; the explicit step needs fac_dt ≲ 0.5
    ω = 1.0,                       # relaxation of the nonlinear (outer) iterations
    apt_tol = 1.0e-6,              # tolerance of the pseudo-transient (inner) iterations
)
```

Everything else, including the material parameters, intrusions, tracers and output, is unchanged. The step is also available directly through `Nonlinear_Diffusion_step!(Arrays, Mat_tup, Phases, Grid, dt, Num)` when `Num.solver` is `:implicit_enthalpy`. `MTK_GeoParams` creates the work arrays it needs (`pt_C`, `pt_b`, `pt_phiold`, `pt_res`, `pt_Tk`, `pt_Tkm1`, `pt_Tn`).

## Equations

The step solves the enthalpy balance

```math
\frac{E(T) - E(T^{\mathrm{old}})}{\Delta t} = \nabla\cdot\left(k\,\nabla T\right) + H + s\,\bigl(E^* - E(T)\bigr), \qquad E(T) = \rho\,\bigl(c_p\,T + L\,\phi(T)\bigr),
```

for the temperature ``T`` at the new time level. Here ``E`` is the enthalpy per unit volume, ``\phi(T)`` the melt fraction, ``T^{\mathrm{old}}`` the temperature after the advection sub-step, and ``s`` (`Arrays.Qrelax`, 1/s) and ``E^*`` (`Arrays.Qenth`, J/m³) an optional relaxation source (see below; ``s = 0`` by default). Unlike the explicit step with an effective heat capacity, the latent heat is accounted for exactly in the time-discrete balance: the enthalpy released between two time levels is ``\rho L\,[\phi(T) - \phi(T^{\mathrm{old}})]``, however large the time step.

### Outer (nonlinear) iterations

``E(T)`` is nonlinear in ``T`` through ``\phi(T)``, and the material properties (``k``, ``\rho``, ``c_p``, ...) may also depend on ``T``. Each outer iteration evaluates the properties at the current iterate ``T_k`` and linearises the melt fraction around it,

```math
\phi(T) \approx \phi(T_k) + \frac{\partial \phi}{\partial T}\bigg|_{T_k}\,(T - T_k),
```

which is a Newton step for the latent heat. It leads to a linear problem ``A\,T = b`` with

```math
A = f\,C - \nabla\cdot\left(k\,\nabla\,\cdot\,\right), \qquad f = \frac{1}{\Delta t} + s, \qquad C = \rho\left(c_p + L\,\frac{\partial\phi}{\partial T}\Big|_{T_k}\right),
```

```math
b = f\,\rho L\left(\frac{\partial\phi}{\partial T}\Big|_{T_k} T_k - \phi(T_k)\right) + \frac{\rho\,\bigl(c_p\,T^{\mathrm{old}} + L\,\phi(T^{\mathrm{old}})\bigr)}{\Delta t} + s\,E^* + H .
```

``A`` is symmetric positive definite. Its solution is relaxed with `ω`, ``T_{k+1} = \omega\,T_{\mathrm{new}} + (1-\omega)\,T_k``, and the outer iterations stop when ``\lVert T_{\mathrm{new}} - T_k\rVert_2 / \max T_{\mathrm{new}} <`` `convergence` (at most `max_iter` iterations). Because of the Newton linearisation, `ω = 1` usually converges in a few iterations. If it does not (for example with very steep melting curves), reduce `ω` or the time step. As for the explicit step, ``\partial\phi/\partial T`` should be continuous, so use `SmoothMelting` (GeoParams) around kinks of the melting curve.

### Inner (pseudo-transient) iterations

The linear problem is solved matrix-free with a Jacobi-preconditioned, second-order ("accelerated") pseudo-transient iteration:

```math
T^{(j+1)} = T^{(j)} + \alpha\,D^{-1}\bigl(b - A\,T^{(j)}\bigr) + \beta\,\bigl(T^{(j)} - T^{(j-1)}\bigr), \qquad D = \mathrm{diag}(A).
```

The iteration parameters are the Chebyshev-optimal ones for the spectrum ``[m, M]`` of ``D^{-1}A``,

```math
\alpha = \frac{4}{\left(\sqrt{M} + \sqrt{m}\right)^2}, \qquad \beta = \left(\frac{\sqrt{M} - \sqrt{m}}{\sqrt{M} + \sqrt{m}}\right)^2 .
```

Since ``D^{-1}A = I - D^{-1}(D - A)`` and ``D - A`` has non-negative entries, the spectrum lies in ``[m,\, 2 - m]`` with ``m = \min_i (f C_i)/D_i``. These safe bounds are computed at every solve, so no parameter has to be tuned. The first iteration is a plain preconditioned Richardson step (``\beta = 0``).

Every iteration is one stencil sweep plus the boundary conditions, with no global reductions except a convergence check every `apt_check` iterations. That makes the iteration well suited to GPUs. The remaining temperature error is estimated from the preconditioned residual as ``\max|D^{-1}(b - A T)| / m``, and the iteration stops when it falls below `apt_tol` times ``\max(\max|T|, 1\,\mathrm{K})``. If `apt_max_iter` iterations are not enough, an error is thrown: increase `apt_max_iter` or reduce the time step. The number of iterations grows roughly with the square root of the time step in units of the explicit stability limit, instead of linearly as for explicit sub-stepping. The rate is ``\sqrt{m}``, and ``m \approx \Delta^2/(\kappa\,\Delta t)`` for large time steps.

### Boundary conditions and geometry

The boundary conditions are those of the explicit step: zero heat flux on the lateral faces, and on the bottom either the temperature of the previous time level (isothermal, the default) or a prescribed heat flux (`flux_bottom_BC = true`, `flux_bottom`). Axisymmetric models (`axisymmetric = true`) use the cylindrical face weights in the radial direction, and 3D models the 7-point stencil. Material properties are evaluated at cell centres, and the conductivity of a face is the arithmetic mean of the two adjacent cells, as in the explicit step.

### Relaxation source

`Arrays.Qrelax` (``s``, 1/s) and `Arrays.Qenth` (``E^*``) define an optional source ``s\,(E^* - E(T))`` that relaxes the enthalpy towards a target value at the rate ``s``, for example to represent a continuous, volume-averaged injection of magma of enthalpy ``E^*``. It enters the implicit problem through ``f`` and ``b`` and is therefore unconditionally stable, however large ``s\,\Delta t`` is. Both arrays are zero by default.

## Parameters

| `NumParam` field | default | meaning |
|:-----------------|:--------|:--------|
| `solver` | `:explicit` | `:implicit_enthalpy` (or `:apt`) selects the implicit step |
| `fac_dt` | 0.4 | time step factor, ``\Delta t =`` `fac_dt` ``\Delta_{\min}^2/(4\kappa_{\max})``; may be much larger than 1 with the implicit step |
| `ω` | 0.5 | relaxation of the outer iterations (1 is usually fine with the implicit step) |
| `convergence`, `max_iter` | 1e-4, 1500 | tolerance and limit of the outer iterations |
| `apt_tol` | 1e-6 | tolerance of the inner iterations, relative to ``\max(\max|T|, 1\,\mathrm{K})`` |
| `apt_max_iter` | 50 000 | maximum number of inner iterations per solve |
| `apt_check` | 10 | convergence check every `apt_check` inner iterations |

## Accuracy and choice of the time step

The implicit step is unconditionally stable, but it is first order in time like the explicit one, so the time step still controls the accuracy. The relevant time scale is the cooling time of the smallest thermal structure that matters, for example ``\tau = H^2/\kappa`` for an intrusion of thickness ``H``. The time step should be a small fraction of it.

As an example, a 500 m thick sill at 1000 °C (with latent heat, 4th-order melting curve) cools in a 20 km × 20 km box on a 201 × 201 grid (Δ = 100 m, ``\kappa = 1.1\cdot10^{-6}`` m²/s, ``\tau \approx 7`` kyr). This is the maximum temperature difference to an explicit reference (`fac_dt = 0.4`, Δt = 28 yr) after 3 kyr:

| `fac_dt` | Δt [yr] | steps | wall time, CPU 16 threads | max. abs. difference to the reference [K] |
|:---------|:--------|:------|:-------------|:----------------|
| 0.4 (explicit) | 28 | 106 | 0.9 s | reference |
| 1 | 70 | 43 | 0.8 s | 2.6 |
| 4 | 273 | 11 | 0.4 s | 6.9 |
| 16 | 1000 | 3 | 0.2 s | 21 |
| 64 | 3000 | 1 | 0.1 s | 57 |

The error grows roughly linearly with Δt. `fac_dt` of a few to about 10 keeps it to a few kelvin for such intrusions. Larger values are fine when only the long-term thermal state of the crust matters, for example in a spin-up phase. The gain over the explicit step grows with the resolution, because the explicit time step shrinks with ``\Delta^2``, while the accuracy limit of the implicit step does not depend on the grid.

## Verification

`test/test_PseudoTransient.jl` checks the implicit step against direct (dense) solves of the same discretisation (constant and variable conductivity, source terms, latent heat; 2D, 3D and axisymmetric), steady states with a source term, the agreement of the explicit and implicit steps for small time steps, the relaxation term, and the growth of the iteration count with the time step.

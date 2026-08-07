# Numerics and Physics

MagmaThermoKinematics.jl is built around a finite-difference energy solver with semi-Lagrangian advection and tracer-based tracking of injected sill material. 

## Governing Equation
The thermal state of the magmatic system is described by the advection–diffusion equation for temperature $T$ with latent heat release:

$$\rho C_p \left(\frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T \right) = \nabla \cdot (k \nabla T) + \rho L \frac{\partial \phi_s}{\partial t} + H \tag{A1}$$

where

| Symbol | Description |
|:-------|:------------|
| $\rho$ | Rock density (kg m⁻³) |
| $C_p$  | Isobaric heat capacity (J kg⁻¹ K⁻¹) |
| $T$    | Temperature (K or °C) |
| $t$    | Time (s) |
| $\mathbf{v}$ | Solid-state velocity field (m s⁻¹) induced by dike/sill emplacement |
| $k$    | Thermal conductivity (W m⁻¹ K⁻¹) |
| $L$    | Latent heat of crystallisation (J kg⁻¹) |
| $\phi_s = (1-\phi)$ | Solid fraction (dimensionless, 0–1), with $\phi$ being the melt fraction |
| $H$    | Radiogenic heat production (W m⁻³) |

## Operator Splitting

Equation (A1) is solved with a first-order operator split that separates advection from diffusion+latent heat. Each time step $\Delta t$ proceeds in two sub-steps:

**Step 1 — advection** (semi-Lagrangian):

$$T^* = T\!\left(\mathbf{x} - \mathbf{v}\,\Delta t,\; t^n\right) \tag{A2a}$$

The departure point $\mathbf{x} - \mathbf{v}\,\Delta t$ is found by back-tracking each grid node by one time step along the velocity field. The temperature at the departure point is obtained by bi-linear (2D) or tri-linear (3D) interpolation from the current grid values $T^n$.  The semi-Lagrangian scheme is unconditionally stable with respect to the advective CFL criterion.

**Step 2 — diffusion and latent heat**:

$$\rho C_p \frac{T^{n+1} - T^*}{\Delta t} = \nabla \cdot \left(k \nabla T^{n+1}\right) + \rho L \frac{\partial \phi_s}{\partial t}\bigg|^{n+1} + H \tag{A2b}$$

## Effective Heat Capacity Formulation

Because melt fraction $\phi$ depends on temperature, the latent heat term in (A2b) can be rewritten using the chain rule:

$$\rho L \frac{\partial \phi_s}{\partial t} = \rho L \frac{\partial \phi_s}{\partial T} \frac{\partial T}{\partial t}$$

Substituting into (A2b) and rearranging yields a modified heat capacity:

$$\rho C_p^{\text{eff}} \frac{T^{n+1} - T^*}{\Delta t} = \nabla \cdot \left(k \nabla T^{n+1}\right) + H \tag{A3}$$

where the *effective* heat capacity absorbs the latent heat contribution:

$$C_p^{\text{eff}} = C_p + L \frac{\partial \phi}{\partial T} \tag{A3a}$$

Where we made use of the fact that $\partial \phi_s/\partial t = -\partial \phi/\partial t$. This formulation is advantageous because it keeps the structure of the standard diffusion equation while automatically capturing latent heat release whenever $\partial\phi/\partial T \neq 0$ (i.e., within the two-phase region). As this latter effect is taken into account in an implicit manner, it is numerically more stable.

An important point to keep in mind, though, is that $\partial\phi/\partial T$ should be *continuous* throughout the domain, including at the solidus and liquidus to prevent numerical instabilities. For this reason, the `GeoParams` package implements smoothening functions for any melting curve (as many parameterisations in use are discontinuous).

## Explicit Time Discretisation

The diffusion operator in (A3) is discretised explicitly:

$$T^{n+1}_{i,j} = T^*_{i,j} + \frac{\Delta t}{\rho C_p^{\text{eff}}} \left[\left(\nabla \cdot k \nabla T\right)^n_{i,j} + H_{i,j}\right] \tag{A4}$$

### Staggered-Grid Finite Differences

The spatial derivatives in (A4) are evaluated on a staggered grid:
- Temperature $T$ and scalar material properties ($\rho$, $C_p$, $\phi$) are defined at cell centres.
- Heat fluxes $q = -k\,\nabla T$ are defined on cell faces (half-integer indices).
- Conductivity $k$ is harmonically averaged to cell faces to conserve flux continuity across material interfaces.

The discrete Laplacian in 2D is:

$$\left(\nabla \cdot k \nabla T\right)_{i,j} \approx
\frac{k_{i+\frac{1}{2},j}(T_{i+1,j}-T_{i,j}) - k_{i-\frac{1}{2},j}(T_{i,j}-T_{i-1,j})}{\Delta x^2}
+
\frac{k_{i,j+\frac{1}{2}}(T_{i,j+1}-T_{i,j}) - k_{i,j-\frac{1}{2}}(T_{i,j}-T_{i,j-1})}{\Delta z^2}$$

with the analogous 3D extension for the $y$-direction.

### CFL Stability Criterion

For the explicit thermal diffusion step the time step must satisfy the standard parabolic CFL condition:

$$\Delta t \leq \frac{1}{2} \frac{\min(\Delta x,\Delta y,\Delta z)^2}{\max(\kappa)} \tag{CFL}$$

where the thermal diffusivity is $\kappa = k / (\rho C_p^{\text{eff}})$. In practice the code evaluates $\kappa$ at every grid point and uses the global maximum to set a conservative $\Delta t$.

## Nonlinear Iterations (Picard)

Because $C_p^{\text{eff}}$ itself depends on $T^{n+1}$ through $\partial\phi/\partial T$, equation (A4) is nonlinear. The solver uses damped Picard (fixed-point) iterations to resolve this nonlinearity within each time step:

**Predictor** — evaluate material properties at the current iterate $T^{(k)}$:

$$C_p^{\text{eff},(k)} = C_p\!\left(T^{(k)}\right) - L\,\frac{\partial \phi}{\partial T}\bigg|_{T^{(k)}} \tag{A5}$$

**Corrector** — advance temperature with the updated effective heat capacity:

$$T^{(k+1)}_{i,j} = T^*_{i,j} + \frac{\Delta t}{\rho\,C_p^{\text{eff},(k)}} \left[\left(\nabla \cdot k^{(k)} \nabla T^{(k)}\right)_{i,j} + H_{i,j}\right] \tag{A6}$$

**Damping** — to aid convergence the update is damped:

$$T^{(k+1)} \leftarrow \alpha\, T^{(k+1)} + (1-\alpha)\, T^{(k)}, \qquad 0 < \alpha \leq 1$$

**Convergence** — iterations continue until the maximum pointwise temperature change falls below a prescribed tolerance $\varepsilon$:

$$\max_{i,j}\left|T^{(k+1)}_{i,j} - T^{(k)}_{i,j}\right| < \varepsilon$$

Typically three to five Picard iterations are sufficient per time step.

## Melt Intrusion Algorithms

Three distinct emplacement modes are available for adding new melt to the system:

| Mode | Description |
|:-----|:------------|
| **Geneva sill (under-accretion)** | New magma is injected at the base of the existing melt body and solid host-rock is displaced downward. |
| **UCLA-HD (central injection)** | Melt is added at the centre of the intrusion; host rock is displaced radially outward while conserving volume. |
| **Elastic dike** | An elliptical dike geometry is used and host rock is displaced by an analytically prescribed elastic displacement field. |

All three modes inject a `Dike` object, update the velocity field $\mathbf{v}$ for one time step (used in the semi-Lagrangian advection step A2a), and add tracer particles at the new melt location.

## Melt Withdrawal and Surface Deformation

The reverse process — removing melt once a chamber becomes eruptible — is handled by the eruption machinery, which offers a kinematic volume trigger and a physical chamber-overpressure trigger, and optionally deflates the chamber with a volume-conserving displacement field. See [Eruptions](eruptions.md).

The top of the domain can be tracked as a kinematic free surface, advected by the same host-rock displacement fields that emplace sills and deflate the chamber, so injection and eruption deform the ground surface. See [Free Surface](free_surface.md).

## Tracers and Temperature–Time Paths

Passive tracer particles are advected with the host-rock velocity field using the same semi-Lagrangian scheme as the temperature field.  At each time step each tracer records its current temperature, producing a continuous $T$–$t$ path.  These paths are the primary input to the [ZirconGrowth integration](zircon_growth.md).

Tracers are kept on the CPU, as they generally use a lot of memory. 

## Dimensions and Geometry

Supported configurations include:

- 2D Cartesian.
- 2D axisymmetric (via specific workflows in examples).
- 3D Cartesian.

## Material Parameter Handling

Material properties are integrated through GeoParams-based parameterizations and package-level helper routines for conductivity, density, heat capacity, melt fraction, and related quantities.

Melting parameterizations and material properties can be exchanged without touching solver internals; see [Melting Parameterizations](MeltingParameterisations.md) and [Conductivity](ConductivityParameterisations.md) for the available laws.

### Changing Melting Parameterizations

Material setup is defined with `SetMaterialParams`, so changing physics is extremely simple and only requires you to change your `MatParam` struct. For example, you can switch between the Caricchi melting parameterization and a simple 4th-order polynomial melting curve with:

:::code-group

```julia [Caricchi]
MatParam = (
	SetMaterialParams(Name="Rock", Phase=1,
		Density=ConstantDensity(ρ=2700kg/m^3),
		HeatCapacity=ConstantHeatCapacity(Cp=1000J/kg/K),
		Conductivity=T_Conductivity_Whittington_parameterised(),
		Melting=MeltingParam_Caricchi()
        ),
)
```

```julia [Smooth Polynomial]
MatParam = (
	SetMaterialParams(Name="Rock", Phase=1,
		Density=ConstantDensity(ρ=2700kg/m^3),
		HeatCapacity=ConstantHeatCapacity(Cp=1000J/kg/K),
		Conductivity=T_Conductivity_Whittington_parameterised(),
		Melting=SmoothMelting(MeltingParam_4thOrder())
        ),
)
# Example alternatives:
# Conductivity=ConstantConductivity(k=3.3W/K/m)
```
:::

This design makes it straightforward to test sensitivity to melt models and thermophysical assumptions without changing solver internals.

## Performance and Parallelism

Backend selection through `environment!` configures CPU threads, or CUDA execution, and the package composes with ParallelStencil finite-difference modules.

- CPU and CUDA workflows are typically run with `Float64` precision.

When your own script uses ParallelStencil macros directly (for example `@zeros` or `@parallel`), call `@init_parallel_stencil(...)` in script scope after `environment!(...)`.

If you run this on a CPU, you can run it in parallel by starting julia in multi-threading mode:

```bash
julia -t auto
```

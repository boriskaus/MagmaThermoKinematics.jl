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
- Temperature $T$ and scalar material properties ($\rho$, $C_p$, $\phi$) are defined at cell centers.
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

Each mode is described by an `InjectSills.AbstractSill` (`CylindricalDikeTopAccretion`, `EllipticalIntrusion` and `PennyShapedSill`, respectively). `inject_sills` uses the host-rock displacement of that sill as the velocity field $\mathbf{v}$ for one time step (used in the semi-Lagrangian advection step A2a) and adds tracer particles at the new melt location.

## Tracers and Temperature–Time Paths

Passive tracer particles are advected with the host-rock velocity field using the same semi-Lagrangian scheme as the temperature field.  At each time step each tracer records its current temperature, producing a continuous $T$–$t$ path.  These paths are the primary input to the [ZirconGrowth integration](zircon_growth.md).

Tracers are kept on the CPU, as they generally use a lot of memory.

## Dimensions and Geometry

Supported configurations include:

- 2D Cartesian.
- 2D axisymmetric (via specific workflows in examples).
- 3D Cartesian.

### Parameterised convection of melt-rich regions

Magma with a low crystal content convects vigorously, which keeps its interior close to isothermal and moves heat to its boundaries much faster than conduction. With `NumParam(solver = :implicit_enthalpy, convection = true)` this is mimicked by an enhanced ("hyper-") conductivity ``K_{eff} = Nu\,K`` in every connected melt-rich region, with the Nusselt number ``Nu`` from the Rayleigh number of the region. The same approach, an effective conductivity ``k = k_m Nu(Ra)``, is used for magma chambers by [Gerbault et al. (2025)](https://doi.org/10.5194/se-16-1307-2025). The default is `convection = false`, which leaves everything unchanged.

**Regions.** A region is a vertical run of cells in a column with melt fraction ``\phi \ge`` `convection_phi` (default 0.5), air cells (phase 0) excluded. Every run has its own thickness, temperature contrast and Rayleigh number, so chambers at different depths, separated by cells with less melt, convect independently. Nothing is prescribed: the thickness is measured from the model at every time step.

**Rayleigh and Nusselt number.** For each region

```math
Ra = \frac{\rho\, g\, \alpha\, \Delta T\, L^3}{\kappa\, \eta}
```

is computed from the thickness ``L`` of the region, the temperature contrast ``\Delta T`` across it (maximum minus minimum temperature), the mean density ``\rho`` and thermal diffusivity ``\kappa = K/(\rho c_p)``, the thermal expansivity ``\alpha`` and the viscosity ``\eta`` at the mean temperature and melt fraction of the region. The Nusselt number is ``Nu = 1`` for ``Ra`` below `convection_Ra_crit` (1708, the onset of convection of a layer with rigid boundaries) and otherwise

```math
\log_{10} Nu = a_2 \log_{10}(Ra)^2 + a_1 \log_{10}(Ra) + a_0 ,
```

with `convection_coeffs` ``= (a_2, a_1, a_0)``, bounded by `convection_Nu_max`. The default coefficients ``(0.0087, 0.1350, -0.2943)`` are the fit for very high Prandtl numbers (``Pr > 10^6``, where ``Nu`` no longer depends on ``Pr``) to the data of Grossmann & Lohse (2000, 2001). Gerbault et al. (2025, Eq. 10) adopted this fit from that earlier work for magma chambers and checked it with their own CFD calculations. They can be replaced, for example by ``Nu = 0.089\,Ra^{1/3}`` with `(0.0, 1/3, log10(0.089))`. Once the interior is homogenised within a time step a larger `convection_Nu_max` changes the result little but makes the implicit solve more expensive.

**Viscosity.** `convection_viscosity` is any GeoParams creep law, evaluated with the temperature and melt fraction of the region at the reference strain rate `convection_strainrate` (``10^{-12}`` 1/s; it only matters for strain-rate dependent laws). The default (`nothing`) is the viscosity of a hydrous rhyolitic melt of [Giordano et al. (2008)](https://doi.org/10.1016/j.epsl.2008.03.016) (72 wt % SiO₂, 6 wt % H₂O) with the crystal correction of Costa et al. (2009), `ViscosityPartialMelt_Costa_etal_2009`. For chambers of different composition `convection_viscosity` can be a tuple of laws, and `convection_z_split` the depths ``z`` [m] that separate them (descending, negative downward): law ``i`` applies to the regions whose mean depth is above split ``i``, and the last law to those below all splits.

**Conductive boundary layer.** The solver takes the conductivity of a cell face as the arithmetic mean of the two neighbouring cells. A jump of ``Nu`` across the edge of a region would therefore let the cold wall conduct at about ``Nu K/2`` instead of ``K``. With `convection_boundary_layer = true` (default) the outermost cells of every region (in all directions) keep ``Nu = 1``; this cell layer is the conductive thermal boundary layer of the region.

**Implementation.** The Nusselt numbers are computed once per time step from the state at the beginning of the step (`Arrays.Nu`, created by `MTK_GeoParams`), and the conductivity of every nonlinear iteration is multiplied by them; this keeps the iteration free of the on/off switching of the regions. The explicit solver is refused with an `ArgumentError`, because the enhanced conductivity is far above its stability limit. The option works for planar 2D, axisymmetric and 3D models and on all backends. With `convection = true` the VTK files contain the Nusselt number of the last step (`Nusselt`) and a flag for the convecting regions (`Convecting`). The scalar function `convective_nusselt` and the array routine `compute_convective_nusselt!` can also be used on their own.

| `NumParam` field | default | meaning |
|:-----------------|:--------|:--------|
| `convection` | `false` | switch on convection (needs `solver = :implicit_enthalpy`) |
| `convection_phi` | 0.5 | melt fraction above which a region convects |
| `convection_Nu_max` | 100 | upper bound of ``Nu`` |
| `convection_coeffs` | `(0.0087, 0.135, -0.2943)` | ``(a_2, a_1, a_0)`` of the ``Nu``-``Ra`` fit |
| `convection_Ra_crit` | 1708 | ``Nu = 1`` below this ``Ra`` |
| `convection_alpha` | ``3\cdot10^{-5}`` | thermal expansivity [1/K] |
| `convection_viscosity` | `nothing` | GeoParams creep law, or a tuple of laws (default: Giordano et al. 2008 rhyolite with Costa et al. 2009 crystal correction) |
| `convection_strainrate` | ``10^{-12}`` | strain rate [1/s] at which the law is evaluated |
| `convection_z_split` | `()` | depths [m] separating the laws of a tuple |
| `convection_boundary_layer` | `true` | outermost cells of a region keep ``Nu = 1`` |

```julia
Num = NumParam(
    solver = :implicit_enthalpy, convection = true, fac_dt = 4.0,
    convection_phi = 0.6,
    convection_viscosity = (rhyolite_law, basalt_law), convection_z_split = (-8.0e3,),   # rhyolite above 8 km depth, basalt below
)
```

The thickness is the vertical extent of the run of melt-rich cells, which is the relevant length for a layer that is wide compared with its thickness; a narrow body that is taller than it is wide convects less than this estimate gives.

## Material Parameter Handling

Material properties are integrated through GeoParams-based parameterizations and package-level helper routines for conductivity, density, heat capacity, melt fraction, and related quantities.

One of the highlights is that melting parameterizations and material properties can be changed very easily by using GeoParams (see ).

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

The model arrays live on a KernelAbstractions backend, selected with `NumParam(backend=...)`: `CPU()` (the default; multithreaded when julia is started with several threads) or `CUDABackend()` after `using CUDA`.

- The element type of the model arrays is set with `NumParam(FloatType=...)` (default `Float64`).

If you run this on a CPU, you can run it in parallel by starting julia in multi-threading mode:

```julia
$julia -t auto
```

# Numerics and Physics

MagmaThermoKinematics.jl is built around a finite-difference energy solver with semi-Lagrangian advection and tracer-based tracking.

## Core Model Ingredients

- Thermal diffusion with nonlinear material properties.
- Latent heat effects during melting and crystallization.
- Tracer advection and interpolation between tracers and grid.
- Kinematic dike/sill emplacement with host-rock displacement.

## Dimensions and Geometry

Supported configurations include:

- 2D Cartesian.
- 2D axisymmetric (via specific workflows in examples).
- 3D Cartesian.

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
# Conductivity=ConstantConductivity(k=3.3Watt/K/m)
```
:::

This design makes it straightforward to test sensitivity to melt models and thermophysical assumptions without changing solver internals.

## Performance and Parallelism

Backend selection through environment! configures CPU threads or CUDA execution, and the package composes with ParallelStencil finite-difference modules.

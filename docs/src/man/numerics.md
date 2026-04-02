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

## Performance and Parallelism

Backend selection through environment! configures CPU threads or CUDA execution, and the package composes with ParallelStencil finite-difference modules.

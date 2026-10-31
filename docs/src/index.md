```@raw html
---
layout: home

hero:
  name: MagmaThermoKinematics.jl
  text: Thermal evolution of magmatic systems in 2D and 3D
  tagline: Dike and sill emplacement, magma-chamber overpressure and eruptions, convective cooling, tracers and zircon ages, on CPUs and GPUs.
  actions:
    - theme: brand
      text: Get Started
      link: /man/installation
    - theme: alt
      text: API Reference
      link: /man/listfunctions
    - theme: alt
      text: View on GitHub
      link: https://github.com/boriskaus/MagmaThermoKinematics.jl
  image:
    src: /logo.png
    alt: MagmaThermoKinematics.jl

features:
  - icon: "🔥"
    title: Thermal Modeling
    details: Simulate cooling, crystallization, latent heat effects, and evolving melt fractions, including convective cooling of melt-rich regions and out-of-plane heat loss in 2D sections.
    link: /man/numerics

  - icon: "🧭"
    title: Dike and Sill Emplacement
    details: Kinematic intrusion workflows in 2D and 3D, including randomized emplacement scenarios.
    link: /man/examples

  - icon: "🌋"
    title: Overpressure and Eruptions
    details: Chamber tracking on the grid, overpressure of a compressible water-bearing magma relaxing into the host rock, eruptions when a dike can reach the surface, host-rock displacement and a moving free surface.
    link: /man/eruptions

  - icon: "💎"
    title: Tracers and Zircon Ages
    details: Temperature–time paths of the emplaced magma and zircon age distributions (UCLA and Geneva methods, or zircon growth with ZirconGrowth.jl).
    link: /man/zircon_growth

  - icon: "🗺️"
    title: Data-Driven Setups
    details: Build the model (topography, geological units, initial temperature) with GeophysicalModelGenerator.jl and compare it with geophysical data.
    link: /man/example_mtk_gmg

  - icon: "🔗"
    title: Material Parameterizations
    details: GeoParams-based material properties and melting parameterizations for flexible physics setups.
    link: /man/MeltingParameterisations

  - icon: "⚙️"
    title: CPU and GPU Backends
    details: Built on KernelAbstractions, with the backend (threaded CPU or CUDA GPU) selected through `NumParam(backend=...)`. GPU backends are not tested in CI.
    link: /man/quickstart

  - icon: "📚"
    title: Full API Coverage
    details: Automatically generated API docs for public and internal symbols in the package.
    link: /man/listfunctions
---
```

## What Is MagmaThermoKinematics.jl?

MagmaThermoKinematics.jl is a Julia package for simulating the thermal evolution of magmatic systems in 2D, 2D axisymmetric and 3D. It supports intrusive events such as dike and sill emplacement, tracks tracers, and couples thermal transport with material properties and phase/melting effects.

Beyond the thermal evolution, it can follow the magma chamber that forms from the intrusions:

- **Overpressure and eruptions**: the overpressure of the chamber follows from the mass balance of a compressible, water-bearing magma and relaxes viscously into the host rock (with a shell viscosity from the temperature field and GeoParams creep laws). The chamber erupts once the overpressure is high enough for a dike to reach the surface; the erupted volume is removed, and the host rock and the free surface move accordingly ([Eruptions](man/eruptions.md)).
- **Convective cooling**: convection in melt-rich regions is parameterised by an effective conductivity from a Nusselt–Rayleigh relation, and heat lost out of the plane of a 2D section can be included ([Numerics and Physics](man/numerics.md)).
- **Zircon ages**: tracers record the temperature–time paths of the emplaced magma, from which zircon age distributions are computed ([Zircon ages](man/zircon_growth.md)).

The package uses finite-difference discretizations, explicit or implicit ([Implicit Enthalpy Solver](man/implicit_solver.md)) time stepping on uniform or refined grids ([Grid Refinement](man/grid_refinement.md)), and runs on CPUs and GPUs.

## Start Here

- Installation and dependencies: [Installation](man/installation.md)
- First workflow with backend setup: [Quick Start](man/quickstart.md)
- End-to-end scripts and outputs: [Examples](man/examples.md)
- Complete symbol-level API: [Function Reference](man/listfunctions.md)

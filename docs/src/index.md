```@raw html
---
layout: home

hero:
  name: MagmaThermoKinematics.jl
  text: Thermal evolution of magmatic systems in 2D and 3D
  tagline: Finite-difference thermal solver with dike and sill emplacement, tracers, and CPU/GPU execution.
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

features:
  - icon: "🔥"
    title: Magma Thermal Modeling
    details: Simulate cooling, crystallization, latent heat effects, and evolving melt fractions.
    link: /man/numerics

  - icon: "🧭"
    title: Dike and Sill Emplacement
    details: Kinematic intrusion workflows in 2D and 3D, including randomized emplacement scenarios.
    link: /man/examples

  - icon: "⚙️"
    title: CPU and GPU Backends
    details: Built on ParallelStencil with a backend selection workflow for threaded CPU and CUDA GPU runs.
    link: /man/quickstart

  - icon: "📚"
    title: Full API Coverage
    details: Automatically generated API docs for public and internal symbols in the package.
    link: /man/listfunctions
---
```

## What Is MagmaThermoKinematics.jl?

MagmaThermoKinematics.jl is a Julia package for simulating the thermal evolution of magmatic systems in 2D and 3D. It supports intrusive events such as dike and sill emplacement, tracks tracers, and couples thermal transport with material properties and phase/melting effects.

The package uses finite-difference discretizations and is designed for high-performance execution on CPUs and GPUs.

## Start Here

- Installation and dependencies: [Installation](man/installation.md)
- First workflow with backend setup: [Quick Start](man/quickstart.md)
- End-to-end scripts and outputs: [Examples](man/examples.md)
- Complete symbol-level API: [Function Reference](man/listfunctions.md)

---
title: "MagmaThermoKinematics.jl: A Julia package for simulating the thermal evolution of magmatic systems"
authors:
  - name: Boris J.P. Kaus
    orcid: 0000-0002-0247-8660
    corresponding: true
    affiliation: 1
  - name: Pascal Aellig
    orcid: 0009-0008-9039-5646
    affiliation: 1
  - name: Albert de Montserrat
    orcid: 0000-0003-1694-3735
    affiliation: 2
affiliations:
  - index: 1
    name: Johannes Gutenberg University Mainz, Germany
  - index: 2
    name: ETH Zürich, Switzerland
date: 5 November 2026
bibliography: paper.bib
tags:
  - Julia
  - magma
  - volcanoes
  - thermal evolution
  - finite differences
  - GPU computing
  - zircon ages
---

# Summary

`MagmaThermoKinematics.jl` is a Julia package that simulates the thermal evolution of magmatic systems in the crust, in 2D, 2D axisymmetric, and 3D. Magma is added as a sequence of dikes or sills that displace the host rock, and the temperature then evolves by diffusion, while taking the latent heat of crystallisation or convective cooling of melt-rich bodies into account. The package also includes a simplified eruption parameterisation, along with the resulting deformation of the host rock and of the free surface. The solvers run on multi-threaded CPUs and on GPUs through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) [@KernelAbstractions], with explicit or implicit time stepping. Material properties such as melting parameterisations and temperature-dependent conductivity come from [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) [@boris_kaus_2025_15719680] and can therefore be easily exchanged, intrusions are emplaced with [InjectSills.jl](https://github.com/JuliaGeodynamics/InjectSills.jl) [@InjectSills], setups with topography and geological units can be built from geophysical and geological data with [GeophysicalModelGenerator.jl](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) [@Kaus2024], and zircon ages can be computed from the temperature–time paths of tracers with [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl) [@ZirconGrowth].

# Statement of need

How quickly a magmatic system assembles, cools, and erupts is a central question in volcanology and igneous petrology. Thermal-kinematic models, in which a reservoir grows by successive intrusions, have been used to address it for several decades [@Dufek2005; @Annen2006; @Caricchi2014; @Tierney2016]. They solve the heat equation with advection, latent heat, and kinematic emplacement of the intrusions; deformation is prescribed rather than computed, which keeps the models cheap while retaining the first-order controls on melt fraction, eruptibility, and thermochronological observables such as zircon ages.

The source code of most existing thermal-kinematic models is unavailable or distributed as compiled binaries, and none of them runs in 3D or on GPUs, which limits resolution and domain size. Comparing such models with eruptive histories, erupted volumes, or surface uplift also requires a link between the thermal model, the pressure of the chamber, and the deformation of the surface. `MagmaThermoKinematics.jl` is open source and modular, runs in 2D and 3D on CPUs and GPUs, and couples the thermal evolution to chamber overpressure, eruptions, and surface deformation.

It is intended for volcanologists, petrologists, and geodynamicists who need a transparent and extensible thermal-kinematic model.

# State of the field

@Annen2006 and @Dufek2005 describe widely used thermal-kinematic models, @Caricchi2014 and @Tierney2016 build on similar frameworks, and @Melnik2021 present a more recent 2D model in Julia with an emphasis on eruptions and zircon records. These codes are closed source, distributed as binaries, or restricted to 2D on CPUs. We started `MagmaThermoKinematics.jl` during the workshop that led to @Schmitt2023, to understand why two widely used codes gave quite different results for the same magma flux, and built it on reusable JuliaGeodynamics components.

# Physics

`MagmaThermoKinematics.jl` solves the advection–diffusion equation for temperature $T$ with latent heat:

$$\rho C_p \left(\frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T \right) = \nabla \cdot \left(k \nabla T\right) + \rho L \frac{\partial \phi_s}{\partial t} + H$$

where $\rho$ is density, $C_p$ heat capacity, $k$ thermal conductivity, $L$ latent heat, $\phi_s = 1-\phi$ the solid fraction, $\phi$ the melt fraction, $H$ (radiogenic) heat production, and $\mathbf{v}$ the velocity field induced by the kinematic emplacement of dikes and sills. 

Each intrusion, typically a penny-shaped crack in an elastic medium, pushes the surrounding host rock aside. The temperature field is advected with this displacement, and so are passive tracers, which are added with every intrusion and record the phase and $T$–$t$ path of the intruded magma. Convection in melt-rich regions ($\phi \geq 0.5$) is approximated by raising their conductivity to $k_{\text{eff}} = \mathrm{Nu}\,k$, with the Nusselt number $\mathrm{Nu}$ computed from the Rayleigh number of each region. In planar 2D sections, conduction in the missing third direction can be approximated by a heat sink $-k\,(T - T_{\text{bg}})/L^2$, where $T_{\text{bg}}$ is the background temperature and $L$ the out-of-plane width of the thermal anomaly. Melt fractions follow from parameterisations in GeoParams.jl or from precomputed MAGEMin phase diagrams [@Riel2022].

The pressure of the magma chamber follows the lumped chamber model of @Degruyter2014, in the form used for growing chambers by @Townsend2019: recharge and the density change of cooling and crystallising magma pressurise the chamber, and elastic deformation and viscous creep of the surrounding crust accommodate its volume change. The chamber erupts when the overpressure is critical and a dike can reach the surface before it freezes [@Rubin1995]. The injected and erupted volumes become a displacement field that moves the host rock, the tracers, and the free surface; a sticky-air layer holds the surface at the surface temperature.

# Software design

The package is modular, so that the solvers run unchanged on CPUs and GPUs and the physics can be changed without touching them. Material properties, intrusions, model setups, and zircon growth are handled by the packages listed above (ZirconGrowth.jl as a package extension), and MagmaThermoKinematics.jl contains the solvers. The kernels are written with KernelAbstractions.jl and run on (parallel) CPU threads or GPUs in single or double precision. Melting parameterisations or conductivity laws are changed through a single argument of `SetMaterialParams`, and simulations are customised with callbacks and multiple dispatch.

Each time step is an operator split: a semi-Lagrangian advection step, followed by a staggered-grid finite-difference diffusion step with an effective heat capacity $C_p^{\text{eff}} = C_p + L\,\partial\phi/\partial T$ and damped Picard iterations for the nonlinear material parameters. The package provides both explicit and implicit solvers.
The explicit diffusion step has a compact stencil and needs no global linear solve, which suits GPUs, but its time step is bounded by $\Delta t \le \Delta^2/(2\kappa)$, a fraction of a year at metre-scale grid spacing. 
An implicit alternative is therefore available, a backward-Euler step of the enthalpy balance

$$\frac{E(T) - E(T^{\text{old}})}{\Delta t} = \nabla \cdot \left(k \nabla T\right) + H, \qquad E(T) = \rho \left(C_p T + L\,\phi(T)\right),$$

where $E$ is the enthalpy per unit volume and $T^{\text{old}}$ the temperature after advection. Writing the balance for $E$ conserves energy, and the latent heat released in a step, $\rho L\,[\phi(T) - \phi(T^{\text{old}})]$, is exact for any time step. Each nonlinear iteration linearises $\phi(T)$ around the current iterate (a Newton step for the latent heat), which gives a symmetric positive-definite linear system. This system is solved matrix-free with a Jacobi-preconditioned accelerated pseudo-transient iteration [@Rass2022], which keeps it suitable for GPUs; the iteration parameters follow from spectral bounds computed at every solve [@Duretz2026] and do not have to be set by hand. The time step is then limited by accuracy only (25 years at 2.5 m grid spacing in \autoref{fig:highres}). Grids can be refined independently along each direction to resolve the intrusion zone within a large domain. 

A test suite compares the solvers with analytical solutions and covers the other parts of the package. It runs on GitHub Actions for the long-term-support and current Julia releases, and the package has [online documentation](https://boriskaus.github.io/MagmaThermoKinematics.jl/) with worked examples.

# Research impact statement

The package was benchmarked against the codes of the UCLA and Geneva groups in @Schmitt2023. The codes agree closely for the same emplacement assumptions, and the differences between emplacement modes are larger than those between codes. It has since been used to study volcano deformation [@Biggs2024; @Weber2025], the thermogenic decarbonation during the emplacement of large igneous provinces [@Wang2026], the sensitivity of magnetotelluric data to the melt fraction of a magma reservoir [@Qiao2026], and magma fluxes inferred from zircon petrochronology [@Lewis2026]; several further manuscripts are in preparation. The ZirconGrowth.jl extension allows modelled zircon age spectra to be compared with measured ones.

# Example Applications

\autoref{fig:highres} shows two high-resolution simulations of the last 150 kyr of the magmatic system below Unzen volcano (Japan), which use most of the features described above. The 2D section has 160 million cells (2.5 m spacing) and took 20 h; the 3D model has 265 million cells (50 m spacing) and took 22 h, both on a single NVIDIA GH200 GPU. Both inject 2.36 km³/kyr (in 2D, the rate through a section of disc-shaped zones), 70% at 3–7 km depth and 30% at 8–16 km. In 2D, 842 eruptions remove 73 km³ of magma, and the surface above the intrusive complex subsides by up to 200 m. In 3D, the sills raise the surface by up to 400 m and the land area grows from 555 to 789 km², while 44 eruptions remove 26 km³. Every sill carries 20 tracers that record its $T$–$t$ path, from which ZirconGrowth.jl computes the 67,181 zircon ages of \autoref{fig:highres}f.

![High-resolution simulations of the last 150 kyr below Unzen volcano (Japan). (a) Present-day temperature in a 2D section ($16001 \times 10001$ grid points); thin lines mark the initial depth of the host rock, the white contour a melt fraction of 0.5. (b) The shallow intrusion zone at full resolution; cyan lines are the sills and dikes of the last 1 kyr. (c) 3D run ($801 \times 661 \times 501$ grid points): temperature (left), and the intrusive complex (blue) and the melt fraction (right). (d, e) Erupted volumes and (f, g) zircon ages of the 3D run. \label{fig:highres}](HighRes_Unzen.png)

# AI usage disclosure
The development of this package started well before the appearance of powerful AI models. More recently, Claude (Anthropic) was used to refactor the code, find bugs, improve and add new features such as convective cooling, eruptions, and free surface deformation. The tools also assisted in drafting portions of this paper and the package documentation. All generated content was reviewed, corrected, and approved by the authors.

# Acknowledgements

Development of `MagmaThermoKinematics.jl` was supported by the European Research Council through the MAGMA project (ERC Consolidator Grant, grant no. 771143). We thank Gregor Weber and Oscar Lovera for discussions and benchmark comparisons with their code that shaped the package design.

# References

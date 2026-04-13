---
title: "MagmaThermoKinematics.jl: A Julia package for simulating the thermal evolution of magmatic systems"
authors:
  - name: Boris J.P. Kaus
    orcid: 0000-0002-0247-8660
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
date: 12 April 2026
bibliography: paper.bib
keywords:
  - Julia
  - magma
  - volcanoes
  - thermal evolution
  - finite differences
  - GPU computing
  - zircon ages
---

# Summary

`MagmaThermoKinematics.jl` is an open-source Julia package for simulating the 2D/3D thermal evolution of crustal magmatic systems below volcanoes. It models the repeated injection of magma into the crust as a sequence of dikes or sills and tracks how the resulting temperature evolves over time through heat diffusion, latent heat release during crystallisation, and advection of the host rock displaced by each intrusion. The package runs in 2D Cartesian, 2D axisymmetric, and 3D Cartesian geometries, and supports both multi-threaded CPU and GPU execution through [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) [@Omlin2022]. Material properties such as melting parameterisations, temperature-dependent conductivity, and other parameters are handled by [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) [@boris_kaus_2025_15719680], making them easy to change without touching the solver internals. Setups can be generated with the [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) package [@Kaus2024] from available data, and zircon ages can be computed from the output, for example by integrating with [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl).

# Statement of Need

How long it takes to assemble, cool, and potentially erupt a magmatic system is a central question in volcanology and igneous petrology. Thermal-kinematic models that track the build-up of a magma reservoir through successive intrusions have been a key quantitative tool for addressing this question for several decades [@Dufek2005; @Annen2006; @Caricchi2014; @Tierney2016]. These models solve the heat equation with advection, latent heat, and kinematic dike emplacement. No mechanical deformation is taken into account, which keeps them computationally tractable while capturing first-order thermal controls on melt fraction, eruption likelihood, and thermochronological observables such as zircon ages.

Despite their wide use, the source code of existing thermal-kinematic packages is largely unavailable or provided only as compiled binaries. None of the established implementations support 3D domains and GPU acceleration, which limits achievable spatial resolution and domain size. `MagmaThermoKinematics.jl` fills this gap: it is fully open-source, runs in 3D, and scales from a laptop CPU to high-end GPUs, enabling ultra-high-resolution simulations that were previously impractical.

The target audience includes volcanologists and petrologists who want to model the thermal history of a specific volcanic system, as well as geodynamicists and software developers who need a transparent, extensible reference implementation of the thermal-kinematic approach.

# State of the Field

Several codes implement the thermal-kinematic approach. @Annen2006 and @Dufek2005 describe commonly-cited models, and @Caricchi2014 and @Tierney2016 build on similar frameworks. A recent Julia reimplementation for 2D dike accretion is described in @Melnik2021. All of these are either closed-source, distributed as compiled binaries, or limited to 2D CPU execution.

`MagmaThermoKinematics.jl` is, to our knowledge, the only open-source thermal-kinematic package that (i) supports both 2D and 3D Cartesian geometries, (ii) runs on GPUs, (iii) integrates with modern Julia ecosystems for material properties (GeoParams.jl) and model setup (GeophysicalModelGenerator.jl), and (iv) provides a direct link to zircon age computation (ZirconGrowth.jl). 

# Physics

`MagmaThermoKinematics.jl` solves the advection–diffusion equation for temperature $T$ with latent heat:

$$\rho C_p \left(\frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T \right) = \nabla \cdot \left(k \nabla T\right) + \rho L \frac{\partial \phi_s}{\partial t} + H$$

where $\rho$ is density, $C_p$ heat capacity, $k$ thermal conductivity, $L$ latent heat, $\phi_s = 1-\phi$ the solid fraction, $\phi$ the melt fraction, $H$ radiogenic heat production, and $\mathbf{v}$ the velocity field induced by kinematic dike/sill emplacement. A detailed derivation and discretisation are given in the appendix of @Schmitt2023 and in the package documentation.


# Software Design

The primary motivation was to make a modular package, that runs on both CPU's and GPU's. We resolved it by separating concerns across three layers: (1) material physics in GeoParams.jl, (2) parallel stencil kernels in ParallelStencil.jl, and (3) the time-stepping loop and dike-injection logic in MagmaThermoKinematics.jl itself. This means the solver kernel is written once and dispatched to CPU threads or GPU without modification, while users can swap melting parameterisations or conductivity models by changing a single argument to `SetMaterialParams`. All aspects of the solver can be overwritten using Julia's multiple dispatch methid, such that simulations can be highly customized. 

The governing equation is solved with an explicit operator split: a semi-Lagrangian advection step followed by an explicit staggered-grid diffusion step with an effective heat capacity $C_p^{\text{eff}} = C_p + L\,\partial\phi/\partial T$ that absorbs latent heat implicitly. Explicit time integration was chosen over implicit because the stencil footprint is compact (nearest neighbours only), making it trivially parallelisable on GPUs without the communication overhead of a global linear solve. The cost is a parabolic CFL stability constraint, which is acceptable because the time scales of interest (thousands to millions of years) are long compared to the diffusive time step at typical crustal resolutions.

Nonlinearity in $C_p^{\text{eff}}$ and potentially other material parameters is handled by damped Picard iterations within each time step. Dike emplacement uses a prescribed velocity field active for one time step; three modes are available (basal under-accretion, central injection, and an elastic penny-shaped crack model). Cide accuracy is verified in the test suite correctness by comparing the 2D and 3D diffusion solver and the semi-Lagrangian advection scheme against analytical solutions.


# Research Impact

The package was benchmarked against other, widely used, codes [@Schmitt2023], showing excellent agreement across a wide range of injection scenarios and confirming that differences between codes are dominated by emplacement-mode assumptions rather than numerical implementation. It has since been used in published volcanological studies [@Biggs2024; @Weber_Biggs_Annen_2025], and in several manuscripts. The optional ZirconGrowth.jl extension enables direct comparison of modelled and measured zircon age spectra (Figure 3).

# Example Applications

Figure 1 shows a 2D simulation of repeated dike injections beneath a generic volcanic system and a 3D simulation. Figure 2 illustrates an ultra-high-resolution 2D run ($10{,}000 \times 10{,}000$ grid points) and a 3D GPU simulation ($512^3$ cells). Figure 3 shows zircon age spectra computed from tracer $T$–$t$ paths of a ZASSy-style simulation [@Schmitt2023], compared with measured distributions from a natural sample.

![2D and 3D examples. Left: simulated temperature field for 2D dike injection. Right: 3D temperature isosurface for the Lanin volcano setup. \label{fig:examples}](figures/fig_examples.png)

![High-resolution and GPU simulations. Left: $10{,}000 \times 10{,}000$ grid-point 2D simulation. Right: 3D GPU simulation of a volcanic system. \label{fig:highres}](figures/fig_highres.png)

![Zircon age distributions computed with ZirconGrowth.jl from tracer Tt-paths. Simulated probability density (blue) compared with a measured zircon age spectrum (grey). \label{fig:zircon}](figures/fig_zircon.png)

# AI Usage Disclosure

Claude (Anthropic) was used to assist in drafting portions of this paper and the package documentation. All generated content was reviewed, corrected, and approved by the authors. The software itself was written by the authors; AI tools were not used to generate production code, even though it was used to detect bugs, reduce memory requirements, for extensions such as ZirconGrowth.jl.

# Acknowledgements

Development of `MagmaThermoKinematics.jl` was supported by the European Research Council through the MAGMA project (ERC Consolidator Grant, grant no. 771143). We thank Gregor Weber and Oscar Lovera for discussions and benchmark comparisons with their code that shaped the package design.

# References

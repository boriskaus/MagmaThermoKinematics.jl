# Benchmarking

MagmaThermoKinematics.jl was benchmarked against the thermal-kinematic codes used by the UCLA (Oscar Lovera) and Geneva (Gregor Weber, Luca Caricchi) research groups as part of the ZASSy intercomparison. Good agreement was found between the codes; the comparison, the equations solved, and the sensitivity of the results to parameters such as temperature-dependent conductivity are described in

- Schmitt, A.K., Sliwinski, J., Caricchi, L., Bachmann, O., Riel, N., Kaus, B.J.P., de Léon, A.C., Cornet, J., Friedrichs, B., Lovera, O., Sheldrake, T., Weber, G. (2023). Zircon age spectra to quantify magma evolution. Geosphere 19. https://doi.org/10.1130/GES02563.1

The benchmark scenario is included in the repository:

```julia
include("examples/Example2D_ZASSy.jl")
```

Note that the thermal structure can differ substantially between models at the *same* magma flux, depending on the assumption made about where and how magma is intruded (underplating, central injection, injection through dikes). This difference is often larger than the effect of varying the material parameters.

# Melting Parameterizations

The melt fraction $\phi(T)$ and its derivative $\partial\phi/\partial T$ enter the energy equation through the effective heat capacity $C_p^{\text{eff}} = C_p + L\,\partial\phi/\partial T$ (see [Numerics and Physics](numerics.md)). Because the solver differentiates the melting curve, $\partial\phi/\partial T$ should be *continuous* across the whole temperature range, including at the solidus and liquidus — many published parameterizations are not, so GeoParams provides `SmoothMelting` to regularize them.

Melting laws are supplied through the `Melting` field of `SetMaterialParams`, so switching between them requires no change to the solver:

```julia
Melting = MeltingParam_Caricchi()                 # as published
Melting = SmoothMelting(MeltingParam_4thOrder())  # regularized at solidus/liquidus
```

For the complete list of implemented parameterizations and their parameters, see the [GeoParams melting documentation](https://juliageodynamics.github.io/GeoParams.jl/dev/man/melting).

# Conductivity

Thermal conductivity $k$ enters the diffusion operator $\nabla\cdot(k\nabla T)$ and is harmonically averaged onto cell faces, so that flux is continuous across material interfaces (see [Numerics and Physics](numerics.md)). Temperature-dependent conductivity is supported: $k$ is re-evaluated at every Picard iteration, so a nonlinear $k(T)$ costs no extra machinery.

Conductivity laws are supplied through the `Conductivity` field of `SetMaterialParams`:

```julia
Conductivity = ConstantConductivity(k = 3.3Watt/K/m)
Conductivity = T_Conductivity_Whittington_parameterised()   # k(T) for crustal rocks
```

Because temperature-dependent conductivity drops significantly at magmatic temperatures, it can measurably change cooling times relative to a constant-$k$ run; the sensitivity is quantified in the Schmitt et al. (2023) benchmark (see [Benchmarking](benchmarking.md)).

For the complete list of implemented parameterizations, see the [GeoParams conductivity documentation](https://juliageodynamics.github.io/GeoParams.jl/dev/man/conductivity).

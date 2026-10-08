# Installation

Install the package from the Julia package manager:

```julia
using Pkg
Pkg.add("MagmaThermoKinematics")
```

Run the test suite:

```julia
using Pkg
Pkg.test("MagmaThermoKinematics")
```

To update an existing installation:

```julia
using Pkg
Pkg.update("MagmaThermoKinematics")
```

## Optional Packages for Examples and Visualization

The examples use packages that are not installed with MagmaThermoKinematics. Add the ones a script loads to your environment:

```julia
using Pkg
Pkg.add("CairoMakie")       # plotting (or GLMakie)
Pkg.add(["MAT", "TimerOutputs"])   # ZASSy example
```

`LoadPhaseDiagrams(...; PlotDiagrams=true)` draws its figures with Makie: load a backend first, e.g. `using CairoMakie` or `using GLMakie`.

## Development Install

For local development from a clone:

```julia
using Pkg
Pkg.develop(path=".")
```

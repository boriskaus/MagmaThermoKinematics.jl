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

Some examples rely on plotting or VTK output packages:

```julia
using Pkg
Pkg.add("Plots")
Pkg.add("Makie")
Pkg.add("WriteVTK")
```

## Development Install

For local development from a clone:

```julia
using Pkg
Pkg.develop(path=".")
```

# Compute zircon ages

MagmaThermoKinematics has 3 ways to compute zircon ages:

1. Following the method of Oscar Lovera ("UCLA" method)
2. Following the method of Gregor Weber & coworkers ("Geneva" method)
3. Following the approach of Melnik & Bindemann in which 1D radially symmetric models are used to compute the details of how zircon crystals grow.

Method 1 and 2 were benchmarked in the ZASSy paper of [Schmitt et al. (2023)](https://pubs.geoscienceworld.org/gsa/geosphere/article/19/4/1006/624461/Zircon-age-spectra-to-quantify-magma-evolution) and are implemented in the script [Plot_ZirconAgeStatistics.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/Plot_ZirconAgeStatistics.jl). See the citations in the ZASSy paper for details. We obtained a reasonable good agreement between our scripts and the other codes that participated in the benchmark, but note that we have not performed a very thorough comparison between the two methods.
Method 3 grows a zircon crystal following the temporal cooling history of every tracer in MTK simulations and will be described in more detail here.

## Growing zircon crystals with ZirconGrowth.jl

MagmaThermoKinematics has an optional integration with [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl), a package that simulates zircon crystal growth along a magma $T$–$t$ path using an implicit finite-difference scheme on a 1-D spherical grid. The integration is implemented as a [Julia package extension](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) and is loaded automatically when you do `using ZirconGrowth` in your session.

#### Installation

ZirconGrowth.jl is a separate package:

```julia
] add https://github.com/JuliaGeodynamics/ZirconGrowth.jl   # while unregistered
] add ZirconGrowth                                          # once registered
```

#### Workflow

A typical MagmaThermoKinematics run records the temperature–time history of each passive tracer particle.  After the simulation, these Tt-paths can be passed directly to the ZirconGrowth model:

```julia
using MagmaThermoKinematics
using ZirconGrowth          # activates the extension

# Option 1 — point at an output directory
age_years, zircon_radius_um = simulate_zircon_growth_from_tracers("MyRun/")

# Option 2 — pass tracers already loaded in memory
using JLD2
Tracers = JLD2.load("MyRun/Tracers_SimParams.jld2", "Tracers")
age_years, zircon_radius_um = simulate_zircon_growth_from_tracers(Tracers)

# Option 3 — single tracer
result = simulate_zircon_growth_from_tracers(Tracers[1])
```
`ZirconGrowth` simulates how a single zircon crystal grows with time. `age_years` is the volume-averaged age of the crystal, in years — the age one would measure on the whole zircon crystal rather than on a single growth zone.

#### Return values

`simulate_zircon_growth_from_tracers` returns a `NamedTuple` with one entry per successfully simulated tracer (tracers with fewer than 2 time steps are skipped):

| Field               | Type              | Description |
|:--------------------|:------------------|:------------|
| `age_years`         | `Vector{Float64}` | Volume-averaged crystallisation age in years before the end of the simulation |
| `zircon_radius_um`  | `Vector{Float64}` | Final crystal radius in µm |


#### Keyword arguments

```julia
simulate_zircon_growth_from_tracers(Tracers;
    params         = nothing,          # ZirconGrowth.GrowthParams (one per tracer if nothing)
    nx             = 100,              # number of 1-D spatial grid points
    elements       = ZirconGrowth.default_element_data(),  # trace elements to track
    filename       = nothing,          # save age_years & zircon_radius_um to this JLD2 file
    return_results = false)            # also return Vector{SimulationResult}
```

#### `nx` — spatial resolution

The ZirconGrowth solver discretises the crystal on a 1-D spherical grid with `nx` points
(default 100).  Adjusting it trades speed against spatial resolution:

```julia
# fast, lower-resolution run
age_years, zircon_radius_um = simulate_zircon_growth_from_tracers(Tracers; nx=100)

# high-resolution run
age_years, zircon_radius_um = simulate_zircon_growth_from_tracers(Tracers; nx=1000)
```

`nx` is ignored when a fully constructed `params` object is passed explicitly.

##### `return_results`

Set `return_results = true` to retrieve the full `ZirconGrowth.SimulationResult` objects.
This uses more memory but gives access to the complete crystal profile, trace-element concentrations, growth rates, etc.:

```julia
age_years, zircon_radius_um, results = simulate_zircon_growth_from_tracers(
    Tracers; return_results = true)

# example: inspect the radius–time history of the first crystal
results[1].time_years
results[1].zircon_radius_um
```

##### `filename`

When a `filename` is provided the summary vectors are saved to a JLD2 file:

```julia
# explicit filename
simulate_zircon_growth_from_tracers(Tracers; filename = "MyRun/ZirconGrowth.jld2")

# directory form saves to dirname/ZirconGrowth.jld2 automatically
simulate_zircon_growth_from_tracers("MyRun/")

# suppress saving when using the directory form
simulate_zircon_growth_from_tracers("MyRun/"; filename = nothing)
```

#### Performance
As a full 1D calculation is run for every tracer, this can take quite some time. Each tracer is independent, so the loop is parallelised with `Threads.@threads`.  

Start Julia with multiple threads for a proportional speedup:
```bash
julia --threads auto
```

or set the environment variable permanently:

```bash
export JULIA_NUM_THREADS=auto
```

Check the thread count at runtime with `Threads.nthreads()`.

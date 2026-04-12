module ZirconGrowthExt

using MagmaThermoKinematics
using MagmaThermoKinematics: JLD2
using ZirconGrowth

"""
    simulate_zircon_growth_from_tracers(Tracers;
                                        params         = nothing,
                                        elements       = ZirconGrowth.default_element_data(),
                                        filename       = nothing,
                                        return_results = false)
    simulate_zircon_growth_from_tracers(tr::Tracer; kwargs...)
    simulate_zircon_growth_from_tracers(dirname::AbstractString; kwargs...)

Run the [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl) crystal-growth
model on tracer Tt-paths recorded during a MagmaThermoKinematics simulation.

Three call forms are supported:
- Pass a `StructArray` of `Tracer` objects directly.
- Pass a single `Tracer`.
- Pass a directory path; `dirname/Tracers_SimParams.jld2` is loaded automatically
  (saves to `dirname/ZirconGrowth.jld2` by default).

Tracers with fewer than 2 time steps are skipped.
`Tracer.time_vec` must be in **Myr** and `Tracer.T_vec` in **°C**.
The loop runs on all available Julia threads (`julia --threads auto`).

## Return value

By default returns a `NamedTuple` with two `Vector{Float64}` fields, one entry per
successfully simulated tracer:

| Field               | Description |
|:--------------------|:------------|
| `age_years`         | Volume-averaged crystallisation age in years before the end of the simulation (see [`volume_averaged_age`](@ref)) |
| `zircon_radius_um`  | Final crystal radius in µm |

When `return_results = true` a third field `results` is included, containing a
`Vector{SimulationResult}` with the full ZirconGrowth output for each tracer.

## Arguments
- `params`         : `ZirconGrowth.GrowthParams`; if `nothing` (default) a fresh instance
                     is constructed from each tracer's Tt-path individually.
- `elements`       : `ZirconGrowth.ElementData` selecting which trace elements are tracked;
                     defaults to `ZirconGrowth.default_element_data()`.
- `filename`       : path to a JLD2 file; if provided the `age_years` and
                     `zircon_radius_um` vectors (and `results` when requested) are saved
                     there.  Default `nothing` (no saving) for the `Tracers` form;
                     `dirname/ZirconGrowth.jld2` for the `dirname` form.
- `return_results` : set to `true` to include the full `Vector{SimulationResult}` in the
                     return value.  Default `false` to save memory.

## Examples

```julia
using MagmaThermoKinematics
using ZirconGrowth          # triggers the extension

# from a directory produced by a simulation run
age_years, zircon_radius_um = simulate_zircon_growth_from_tracers("MyRun/")

# from tracers already in memory
using JLD2
Tracers = JLD2.load("MyRun/Tracers_SimParams.jld2", "Tracers")
age_years, zircon_radius_um = simulate_zircon_growth_from_tracers(Tracers)

# single tracer
res = simulate_zircon_growth_from_tracers(Tracers[1])

# also retrieve full SimulationResult objects
age_years, zircon_radius_um, results = simulate_zircon_growth_from_tracers(
    Tracers; return_results = true)
```
"""
function MagmaThermoKinematics.simulate_zircon_growth_from_tracers(Tracers;
        params::Union{Nothing, ZirconGrowth.GrowthParams} = nothing,
        elements::ZirconGrowth.ElementData = ZirconGrowth.default_element_data(),
        filename::Union{Nothing, AbstractString} = nothing,
        return_results::Bool = false)

    n                = length(Tracers)
    age_years        = Vector{Union{Nothing, Float64}}(nothing, n)
    zircon_radius_um = Vector{Union{Nothing, Float64}}(nothing, n)
    _results         = return_results ? Vector{Union{Nothing, ZirconGrowth.SimulationResult}}(nothing, n) : nothing
    done             = Threads.Atomic{Int}(0)
    interval         = max(1, n ÷ 20)

    println("Simulating zircon growth for $n tracers on $(Threads.nthreads()) thread(s)...")
    Threads.@threads for i in eachindex(Tracers)
        tr = Tracers[i]
        isempty(tr.time_vec) && continue

        time_Myr = Float64.(tr.time_vec)   # already in Myr
        T_C      = Float64.(tr.T_vec)

        length(time_Myr) < 2 && continue

        p = isnothing(params) ?
            ZirconGrowth.GrowthParams(time_Myr, T_C) : params

        res = ZirconGrowth.simulate_from_cooling_path(time_Myr, T_C;
                  params = p, elements = elements)

        age_years[i]        = MagmaThermoKinematics.volume_averaged_age(res)
        zircon_radius_um[i] = res.zircon_radius_um[end]
        return_results && (_results[i] = res)

        k = Threads.atomic_add!(done, 1) + 1
        k % interval == 0 && print("\r  $k / $n done...")
    end

    age_years        = Float64[v for v in age_years        if !isnothing(v)]
    zircon_radius_um = Float64[v for v in zircon_radius_um if !isnothing(v)]
    println("\rDone: $(length(age_years)) / $n tracers simulated.          ")

    if !isnothing(filename)
        JLD2.jldsave(filename; age_years, zircon_radius_um)
        println("Saved ZirconGrowth results to: $filename  ($(length(age_years)) tracers)")
    end

    if return_results
        results = ZirconGrowth.SimulationResult[r for r in _results if !isnothing(r)]
        return (; age_years, zircon_radius_um, results)
    end

    return (; age_years, zircon_radius_um)
end

function MagmaThermoKinematics.simulate_zircon_growth_from_tracers(tr::MagmaThermoKinematics.Tracer; kwargs...)
    return MagmaThermoKinematics.simulate_zircon_growth_from_tracers((tr,); kwargs...)
end

"""
    simulate_zircon_growth_from_tracers(dirname::AbstractString; kwargs...)

Load tracers from `dirname/Tracers_SimParams.jld2` and call
`simulate_zircon_growth_from_tracers(Tracers; kwargs...)`.
"""
function MagmaThermoKinematics.simulate_zircon_growth_from_tracers(dirname::AbstractString;
        filename::Union{Nothing, AbstractString} = joinpath(dirname, "ZirconGrowth.jld2"),
        kwargs...)

    Tracers = JLD2.load(joinpath(dirname, "Tracers_SimParams.jld2"), "Tracers")
    return MagmaThermoKinematics.simulate_zircon_growth_from_tracers(Tracers;
               filename = filename, kwargs...)
end

"""
    volume_averaged_age(result::SimulationResult) -> Float64

Compute the volume-averaged crystallisation age (in years) of a single zircon crystal.

Each concentric shell crystallised at a different time. The shell between radii
`r[i]` and `r[i+1]` has volume ∝ r[i+1]³ − r[i]³ and an age of
`time_years[end] − time_years[i]` (measured back from the end of the simulation).
Shells with zero or negative growth are excluded.

Returns the volume-weighted mean age in **years**.
"""
function MagmaThermoKinematics.volume_averaged_age(result::ZirconGrowth.SimulationResult)
    t = result.time_years
    r = result.zircon_radius_um

    age_sum = 0.0
    vol_sum = 0.0
    t_end   = t[end]

    for i in 1:(length(r) - 1)
        dV = r[i+1]^3 - r[i]^3   # proportional to shell volume; 4π/3 cancels
        dV <= 0 && continue
        age_mid  = t_end - 0.5*(t[i] + t[i+1])   # age of shell midpoint
        age_sum += age_mid * dV
        vol_sum += dV
    end

    return vol_sum > 0 ? age_sum / vol_sum : 0.0
end

"""
    volume_averaged_age(results::AbstractVector) -> Vector{Float64}

Apply `volume_averaged_age` to each element of `results` and return a `Vector{Float64}`.
"""
function MagmaThermoKinematics.volume_averaged_age(results::AbstractVector)
    return [MagmaThermoKinematics.volume_averaged_age(r) for r in results]
end

end # module ZirconGrowthExt

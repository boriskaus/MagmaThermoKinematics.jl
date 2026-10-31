module Grid
# This creates a computational grid
import Base: show
import GeophysicalModelGenerator: CartData

export GridData, CreateGrid, refined_coordinates, cell_widths, min_spacing

# `R` and `RC` are the concrete types of the coordinate fields: ranges for a grid with
# constant spacing, `Vector`s for a refined (non-uniform) grid. `StepRangeLen{FT}` alone
# leaves three of the range's four parameters free, so every read of `coord1D` would
# return an abstractly typed range and every operation on it would dispatch dynamically.
# Construct through `CreateGrid`, which fills them in.
struct GridData{FT, D, R <: AbstractVector{FT}, RC <: AbstractVector{FT}}
    ConstantΔ::Bool                         # Constant spacing in every direction (the coordinates are ranges), or a refined grid (the coordinates are vectors)
    N::NTuple{D, Int}                # Number of grid points in every direction
    Δ::NTuple{D, FT}                 # spacing in every direction (the smallest one for a refined grid)
    L::NTuple{D, FT}                 # Domain size
    min::NTuple{D, FT}                 # start of the grid in every direction
    max::NTuple{D, FT}                 # end of the grid in every direction
    coord1D::NTuple{D, R}                  # Tuple with 1D vectors in all directions
    coord1D_cen::NTuple{D, RC}                 # Tuple with 1D vectors of center points in all directions

    # Explicit parameters only: an empty grid would leave `FT` undetermined.
    GridData{FT, D, R, RC}(ConstantΔ, N, Δ, L, min, max, coord1D, coord1D_cen) where {FT, D, R, RC} =
        new{FT, D, R, RC}(ConstantΔ, N, Δ, L, min, max, coord1D, coord1D_cen)
end


"""

Creates a 1D, 2D or 3D grid of given size. Grid can be created by defining the size and either the `extent` (length) of the grid in all directions, or by defining start & end points 

Spacing is constant, unless the grid is refined with `refine` or given by its coordinates with `coords`.

Note: since this is mostly for Solid Earth geoscience applications, in 2D the second dimension is called z (vertical)

# Refined grids
With `refine`, the points of a direction are concentrated in a zone of interest (e.g. around a magma chamber)
while their number stays the same: `refine` is a named tuple with an entry per refined direction (`x`, `y`, `z`;
in 2D `x` and `z`), each holding the keyword arguments of [`refined_coordinates`](@ref), e.g.

```julia
Grid = CreateGrid(size=(301, 301), extent=(20e3, 20e3), refine=(x = (zone = (0.0, 5e3), ratio = 4), z = (zone = (-10e3, -4e3), ratio = 4)))
```

gives a spacing 4 times smaller inside the zones than outside, with a smooth transition. Alternatively,
`coords` gives the 1D coordinate vectors of all directions, e.g. `coords = (x, z)`. The grid is then a tensor
product of 1D coordinate vectors (`Grid.ConstantΔ == false`, `Grid.coord1D` holds `Vector`s and `Grid.Δ` the
smallest spacing per direction); all directions are stored as vectors if one of them is non-uniform.

# Examples
====

```julia
Grid = CreateGrid(size=(10,20),x=(0.,10), z=(2.,10))
Grid{Float64, 2} 
           size: (10, 20) 
         length: (10.0, 8.0) 
         domain: x ∈ [0.0, 10.0], z ∈ [2.0, 10.0] 
 grid spacing Δ: (1.1111111111111112, 0.42105263157894735) 
```
"""
function CreateGrid(;
        size = (),
        x = nothing, z = nothing, y = nothing,
        extent = nothing,
        refine = (;),
        coords = nothing,
    )

    isnothing(coords) || return _grid_from_coords(Tuple(coords))

    if isa(size, Number)
        size = (size,)  # transfer to tuple
    end
    if isa(extent, Number)
        extent = (extent,)
    end
    N = size
    dim = length(N)

    # Specify domain by length in every direction
    if !isnothing(extent)
        x, y, z = nothing, nothing, nothing
        x = (0.0, extent[1])
        if dim == 2
            z = (-extent[2], 0.0)       # vertical direction (negative)
        end
        if dim == 3
            y = (0.0, extent[2])
            z = (-extent[3], 0.0)       # vertical direction (negative)
        end
    end

    FT = typeof(x[1])
    if dim == 1
        L = (x[2] - x[1],)
        X₁ = (x[1],)
    elseif dim == 2
        L = (x[2] - x[1], z[2] - z[1])
        X₁ = (x[1], z[1])
    else
        L = (x[2] - x[1], y[2] - y[1], z[2] - z[1])
        X₁ = (x[1], y[1], z[1])
    end
    Xₙ = X₁ .+ L
    Δ = L ./ (N .- 1)

    if !isempty(refine)
        names = _dim_names(dim)
        unknown = setdiff(collect(keys(refine)), collect(names))
        isempty(unknown) || throw(ArgumentError("refine: unknown direction(s) $(Tuple(unknown)) for a $(dim)D grid; use $(names)"))
        coords = ntuple(dim) do d
            haskey(refine, names[d]) ? refined_coordinates(X₁[d], Xₙ[d], N[d]; refine[names[d]]...) : collect(range(X₁[d], Xₙ[d]; length = N[d]))
        end
        return _grid_from_coords(coords)
    end

    # Generate 1D coordinate arrays of vertices in all directions
    coord1D = ()
    for idim in 1:dim
        coord1D = (coord1D..., range(X₁[idim], Xₙ[idim]; length = N[idim]))
    end

    # Generate 1D coordinate arrays centers in all directions
    coord1D_cen = ()
    for idim in 1:dim
        coord1D_cen = (coord1D_cen..., range(X₁[idim] + Δ[idim] / 2, Xₙ[idim] - Δ[idim] / 2; length = N[idim] - 1))
    end

    ConstantΔ = true
    return GridData{typeof(first(Δ)), dim, eltype(coord1D), eltype(coord1D_cen)}(ConstantΔ, N, Δ, L, X₁, Xₙ, coord1D, coord1D_cen)

end

_dim_names(dim) = dim == 1 ? (:x,) : dim == 2 ? (:x, :z) : (:x, :y, :z)

# `true` if the spacing of the coordinate vector `c` is constant (to a relative tolerance)
_is_uniform(c; rtol = 1.0e-10) = (h = diff(c); all(abs.(h .- h[1]) .<= rtol * abs(h[1])))

# Grid from the 1D coordinate vectors of all directions: ranges if every direction has a constant spacing, vectors otherwise
function _grid_from_coords(coords::Tuple)
    dim = length(coords)
    FT = float(promote_type(map(eltype, coords)...))
    for (d, c) in enumerate(coords)
        length(c) >= 2 || throw(ArgumentError("coordinate vector $d needs at least 2 points"))
        all(>(0), diff(c)) || throw(ArgumentError("coordinate vector $d must be strictly increasing"))
    end
    lo = ntuple(d -> FT(first(coords[d])), dim)
    hi = ntuple(d -> FT(last(coords[d])), dim)
    N = ntuple(d -> length(coords[d]), dim)
    all(_is_uniform, coords) && return CreateGrid(size = N, x = (lo[1], hi[1]), y = dim == 3 ? (lo[2], hi[2]) : nothing, z = dim == 1 ? nothing : (lo[dim], hi[dim]))
    coord1D = ntuple(d -> collect(FT, coords[d]), dim)
    coord1D_cen = ntuple(d -> (coord1D[d][1:(end - 1)] .+ coord1D[d][2:end]) ./ 2, dim)
    Δ = ntuple(d -> minimum(diff(coord1D[d])), dim)
    L = hi .- lo
    return GridData{FT, dim, Vector{FT}, Vector{FT}}(false, N, Δ, L, lo, hi, coord1D, coord1D_cen)
end

"""
    x = refined_coordinates(lo, hi, n; zone, ratio = 4.0, Δ = nothing, transition = nothing, max_growth = 1.05)

`n` coordinates from `lo` to `hi` that are concentrated in the interval `zone = (a, b)`: the spacing is constant
inside the zone and grows smoothly (as `1 - cos`) over the distance `transition` outside it, to `ratio` times the
spacing in the zone. Instead of `ratio`, the spacing in the zone can be given with `Δ`. By default `transition`
is chosen such that the spacing of neighbouring points differs by at most a factor `max_growth` (the spacing reaches
`ratio` times that of the zone only if the domain is long enough for this transition). The spacing of the zone follows
from `n`: compared with the constant spacing `(hi - lo)/(n - 1)`, a refined direction resolves the zone better with
the same number of points. The smallest spacing that can be reached this way is limited (a longer transition costs
points); a larger `max_growth` or a given `transition` allows stronger refinement.

With `transition = 0` the spacing changes abruptly at the edges of the zone, which are then grid points: constant spacing
in the zone and (about `ratio` times larger) constant spacing outside it. No points are spent on a transition, so the zone is
resolved best for a given number of points; the discretisation stays conservative across the jump.
"""
function refined_coordinates(lo, hi, n::Integer; zone, ratio = 4.0, Δ = nothing, transition = nothing, max_growth = 1.05)
    lo, hi = float(lo), float(hi)
    FT = typeof(lo)
    a, b = FT.(zone)
    lo < hi || throw(ArgumentError("refined_coordinates: need lo < hi"))
    n >= 3 || throw(ArgumentError("refined_coordinates: need at least 3 points"))
    (lo <= a < b <= hi) || throw(ArgumentError("refined_coordinates: zone ($a, $b) must be an interval within ($lo, $hi)"))
    max_growth > 1 || throw(ArgumentError("refined_coordinates: max_growth must be > 1"))
    if !isnothing(transition) && transition == 0
        isnothing(Δ) && ratio < 1 && throw(ArgumentError("refined_coordinates: ratio must be ≥ 1"))
        return _abrupt_nodes(lo, hi, n, a, b, FT(ratio), Δ)
    end
    if !isnothing(Δ)
        # the spacing of the zone first decreases with `ratio`, until the transition (which lengthens with `ratio`) no longer fits in the domain
        hzone(r) = minimum(diff(_refined_nodes(lo, hi, n, a, b, r, transition, max_growth)))
        rs = exp10.(range(0, 3; length = 61))
        hs = hzone.(rs)
        j = argmin(hs)
        Δ >= hs[j] || throw(ArgumentError("refined_coordinates: Δ = $Δ cannot be reached with $n points and max_growth = $max_growth (smallest spacing $(hs[j])); use more points, a smaller zone or a larger max_growth"))
        Δ < hs[1] || throw(ArgumentError("refined_coordinates: Δ = $Δ is not smaller than the constant spacing $(hs[1])"))
        r_lo, r_hi = 1.0, rs[j]
        for _ in 1:50
            r = (r_lo + r_hi) / 2
            hzone(r) > Δ ? (r_lo = r) : (r_hi = r)
        end
        ratio = r_hi
    end
    ratio >= 1 || throw(ArgumentError("refined_coordinates: ratio must be ≥ 1"))
    return _refined_nodes(lo, hi, n, a, b, FT(ratio), transition, max_growth)
end

# abrupt change of the spacing at the edges of the zone, which are grid points: `nz` intervals in the zone, the others outside it
function _abrupt_nodes(lo, hi, n, a, b, ratio, Δ)
    Lz, Lo = b - a, (a - lo) + (hi - b)
    nout = (a > lo) + (hi > b)                      # outer segments, each needs at least one interval
    if isnothing(Δ)
        nz = Lo == 0 ? n - 1 : round(Int, Lz / ((Lz + Lo / ratio) / (n - 1)))
    else
        Δ < (hi - lo) / (n - 1) || throw(ArgumentError("refined_coordinates: Δ = $Δ is not smaller than the constant spacing $((hi - lo) / (n - 1))"))
        nz = round(Int, Lz / Δ)
        nz <= n - 1 - nout || throw(ArgumentError("refined_coordinates: Δ = $Δ cannot be reached with $n points (the zone alone needs $nz intervals)"))
    end
    nz = clamp(nz, 1, n - 1 - nout)
    rest = n - 1 - nz
    nl = a > lo ? (hi > b ? clamp(round(Int, rest * (a - lo) / Lo), 1, rest - 1) : rest) : 0
    nr = rest - nl
    x = collect(range(a, b; length = nz + 1))
    nl > 0 && (x = vcat(collect(range(lo, a; length = nl + 1))[1:(end - 1)], x))
    nr > 0 && (x = vcat(x, collect(range(b, hi; length = nr + 1))[2:end]))
    x[1], x[end] = lo, hi
    return x
end

# relative spacing: 1 in the zone [a, b], rising as (1 - cos) to `ratio` over the distance `Lt` outside it
@inline function _relative_spacing(x, a, b, ratio, Lt)
    d = x < a ? a - x : (x > b ? x - b : zero(x))
    return d >= Lt ? ratio : 1 + (ratio - 1) * (1 - cos(π * d / Lt)) / 2
end

function _refined_nodes(lo, hi, n, a, b, ratio, transition, max_growth)
    ratio == 1 && return collect(range(lo, hi; length = n))
    if isnothing(transition)
        # the neighbour ratio is about 1 + h_zone ds/dx ≤ 1 + h_zone (ratio - 1) π/(2 Lt): iterate on the spacing of the zone
        h = (hi - lo) / (n - 1)
        x = nothing
        for _ in 1:8
            Lt = h * (ratio - 1) * π / (2 * (max_growth - 1))
            x = _nodes_for_spacing(lo, hi, n, a, b, ratio, Lt)
            h = minimum(diff(x))
        end
        return x
    end
    return _nodes_for_spacing(lo, hi, n, a, b, ratio, float(transition))
end

# nodes equally spaced in u(x) = ∫ dx/s(x), so that the spacing is proportional to s
function _nodes_for_spacing(lo, hi, n, a, b, ratio, Lt)
    m = 64 * (n - 1) + 1
    xs = range(lo, hi; length = m)
    w = [inv(_relative_spacing(x, a, b, ratio, Lt)) for x in xs]
    u = zeros(m)
    for i in 2:m
        u[i] = u[i - 1] + (w[i - 1] + w[i]) / 2 * step(xs)
    end
    x = similar(u, n)
    j = 1
    for i in 1:n
        t = u[end] * (i - 1) / (n - 1)
        while j < m - 1 && u[j + 1] < t
            j += 1
        end
        x[i] = xs[j] + (t - u[j]) / (u[j + 1] - u[j]) * step(xs)
    end
    x[1], x[end] = lo, hi
    return x
end

"""
    cell_widths(c)

Width of the control volume of every point of the 1D coordinate vector `c` (half the distance to both neighbours;
the outermost points take the distance to their only neighbour): a number for a range (constant spacing), a vector otherwise.
"""
cell_widths(c::AbstractRange) = step(c)
cell_widths(c::AbstractVector) = (h = diff(c); [_cw(h, k, length(c)) for k in eachindex(c)])

# width of the control volume of point `k` of `n` from the spacing `h` (a number, or the distances between neighbouring points)
@inline _cw(h::Number, k, n) = h
@inline _cw(h::AbstractVector, k, n) = k == 1 ? h[1] : k == n ? h[n - 1] : (h[k - 1] + h[k]) / 2

"Smallest distance between neighbouring points of the 1D coordinate vector `c`."
min_spacing(c::AbstractRange) = step(c)
min_spacing(c::AbstractVector) = minimum(diff(c))


"""
    Grid = CreateGrid(d::CartData; m_to_km::Bool=true)
Creates a grid that can be used within MTK, from a 2D or 3D `CartData` object (generated within the GeophysicalModelGenerator).
The grid is regularly spaced, unless the coordinates of `d` are not (e.g. a `CartData` set created with `xyz_grid` from refined coordinate vectors).
"""
function CreateGrid(d::CartData; m_to_km::Bool = true)
    Nx, Ny, Nz = size(d.x.val)
    if Nz == 1
        dim = 2
    else
        dim = 3
    end
    if m_to_km
        scaling = 1.0e3
    else
        scaling = 1
    end

    # 1D coordinate vectors along the grid directions; a refined grid if one of them is non-uniform
    c1D = dim == 3 ? (d.x.val[:, 1, 1], d.y.val[1, :, 1], d.z.val[1, 1, :]) : (d.x.val[:, 1, 1], d.z.val[1, :, 1])
    if all(c -> length(c) > 1 && all(>(0), diff(c)), c1D) && !all(_is_uniform, c1D)
        return _grid_from_coords(map(c -> c .* scaling, c1D))
    end

    x = extrema(d.x.val) .* scaling
    y = extrema(d.y.val) .* scaling
    z = extrema(d.z.val) .* scaling

    if dim == 3
        Grid = CreateGrid(size = (Nx, Ny, Nz), x = x, y = y, z = z)
    else
        Grid = CreateGrid(size = (Nx, Ny), x = x, z = z)
    end
    return Grid
end


# view grid object
function show(io::IO, g::GridData{FT, DIM}) where {FT, DIM}

    return print(
        io, "Grid{$FT, $DIM} \n",
        "           size: $(g.N) \n",
        "         length: $(g.L) \n",
        "         domain: $(domain_string(g)) \n",
        g.ConstantΔ ? " grid spacing Δ: $(g.Δ) \n" : " refined grid, Δ: $(g.Δ) - $(map(c -> maximum(diff(c)), g.coord1D)) \n"
    )

end

# nice printing of grid
function domain_string(grid::GridData{FT, DIM}) where {FT, DIM}

    xₗ, xᵣ = grid.coord1D[1][1], grid.coord1D[1][end]
    if DIM > 1
        yₗ, yᵣ = grid.coord1D[2][1], grid.coord1D[2][end]
    end
    if DIM > 2
        zₗ, zᵣ = grid.coord1D[3][1], grid.coord1D[3][end]
    end
    if DIM == 1
        return "x ∈ [$xₗ, $xᵣ]"
    elseif DIM == 2
        return "x ∈ [$xₗ, $xᵣ], z ∈ [$yₗ, $yᵣ]"
    elseif DIM == 3
        return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ], z ∈ [$zₗ, $zᵣ]"
    end
end


end

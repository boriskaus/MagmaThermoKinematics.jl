# FreeSurface.jl
#
# Kinematic sticky-air free surface (issue 4).
#
# The surface is tracked as a topography `z_surf` on the fixed grid (one
# elevation per surface column). z increases upward. A grid cell is "air" when
# it lies strictly above the topography; air cells are stamped with the surface
# temperature, zero melt fraction, and the air phase. The surface moves with the
# host rock: it is advected vertically by the host-rock displacement field that
# `inject_sills` (inflation) and `deflate_hostrock!` (eruption deflation)
# already produce, so the same kinematics that open sills / deflate chambers
# also raise and lower the ground surface.
#
# Pure, array-level core (no ParallelStencil); the time-loop wiring lives in the
# MTK_GMG drivers.

"""
    z_surf = init_free_surface(Grid::GridData; z0=0.0, topography=nothing)

Allocate a free-surface topography. In 2D this is a length-`Nx` vector (one
elevation per x-column); in 3D an `Nx×Ny` matrix.

By default (`topography === nothing`) the surface is **flat** at elevation `z0`
(backward-compatible). A non-flat initial topography can be supplied via
`topography`, which may be either:

- an `AbstractArray` of explicit per-column elevations — length `Nx` in 2D, size
  `Nx×Ny` in 3D (shape-validated); or
- a `Function` evaluated at the column coordinates — `f(x)` in 2D, `f(x, y)` in
  3D (with `x`, `y` taken from `Grid.coord1D`).

The returned `z_surf` plugs straight into [`apply_free_surface!`](@ref) and
[`advect_surface!`](@ref), which already operate per column and therefore handle
an arbitrary (sloped / topographic) surface without further changes.
"""
function init_free_surface(Grid::GridData; z0::Real=0.0, topography=nothing)
    N     = Grid.N
    dim   = length(N)
    coord = Grid.coord1D
    if dim == 2
        zs = fill(Float64(z0), N[1])
        if topography isa AbstractArray
            length(topography) == N[1] ||
                error("init_free_surface: topography vector length $(length(topography)) ≠ Nx=$(N[1])")
            zs .= Float64.(topography)
        elseif topography isa Function
            @inbounds for i in 1:N[1]
                zs[i] = Float64(topography(coord[1][i]))
            end
        elseif !isnothing(topography)
            error("init_free_surface: `topography` must be an AbstractVector, a Function f(x), or nothing (2D)")
        end
        return zs
    elseif dim == 3
        zs = fill(Float64(z0), N[1], N[2])
        if topography isa AbstractArray
            size(topography) == (N[1], N[2]) ||
                error("init_free_surface: topography size $(size(topography)) ≠ (Nx,Ny)=($(N[1]),$(N[2]))")
            zs .= Float64.(topography)
        elseif topography isa Function
            @inbounds for j in 1:N[2], i in 1:N[1]
                zs[i,j] = Float64(topography(coord[1][i], coord[2][j]))
            end
        elseif !isnothing(topography)
            error("init_free_surface: `topography` must be an AbstractMatrix, a Function f(x,y), or nothing (3D)")
        end
        return zs
    else
        error("init_free_surface: only 2D and 3D grids are supported (got $(dim)D)")
    end
end

"""
    apply_free_surface!(T, ϕ, Phases, z_surf, Grid::GridData, Tair, air_phase; fill_phase=nothing)

Stamp "air" onto every grid cell that lies strictly above the topography
`z_surf` (i.e. `z_cell > z_surf[col]`): set `T = Tair`, `ϕ = 0`, and
`Phases = air_phase`. Cells at or below the surface are left untouched. Works in
2D (`z_surf::AbstractVector`, length `Nx`) and 3D (`z_surf::AbstractMatrix`,
size `Nx×Ny`). Mutates `T`, `ϕ`, `Phases` in place.

If `fill_phase` is given, the *complementary* operation is also applied: any cell
that is **at or below** `z_surf` but still carries the `air_phase` is back-filled
with rock, so the air/rock interface in the phase array follows `z_surf` exactly.
This matters because `z_surf` is advected as a continuous float, whereas the
discrete phase advection ([`advect_phases!`](@ref)) rounds each (often sub-cell)
inflation step to the nearest node and can therefore leave a stale flat air band
sitting below the risen surface. The fill phase is taken per column from the
nearest rock phase *below* the cell (preserving any layering), falling back to
`fill_phase` (typically the host/background phase) when none is found. Only the
phase is changed on back-fill — `T`/`ϕ` are left to the advection + diffusion.
"""
function apply_free_surface!(T::AbstractArray, ϕ::AbstractArray, Phases::AbstractArray,
                             z_surf::AbstractArray, Grid::GridData, Tair::Real, air_phase::Integer;
                             fill_phase=nothing)
    coord = Grid.coord1D
    dim   = length(coord)
    if dim == 2
        z = coord[2]
        @inbounds for i in axes(T, 1)
            last_rock = fill_phase                       # deepest known rock (fallback)
            for j in eachindex(z)                        # sweep upward (z increasing)
                if z[j] > z_surf[i]
                    T[i,j]      = Tair
                    ϕ[i,j]      = 0.0
                    Phases[i,j] = air_phase
                elseif !isnothing(fill_phase)
                    if Phases[i,j] == air_phase
                        Phases[i,j] = last_rock          # below surface ⇒ can't be air
                    else
                        last_rock = Phases[i,j]
                    end
                end
            end
        end
    elseif dim == 3
        z = coord[3]
        @inbounds for j in axes(T, 2), i in axes(T, 1)
            last_rock = fill_phase
            for k in eachindex(z)
                if z[k] > z_surf[i,j]
                    T[i,j,k]      = Tair
                    ϕ[i,j,k]      = 0.0
                    Phases[i,j,k] = air_phase
                elseif !isnothing(fill_phase)
                    if Phases[i,j,k] == air_phase
                        Phases[i,j,k] = last_rock
                    else
                        last_rock = Phases[i,j,k]
                    end
                end
            end
        end
    else
        error("apply_free_surface!: only 2D and 3D grids are supported (got $(dim)D)")
    end
    return nothing
end

"""
    advect_phases!(Phases, Velocity, Grid::GridData)

Nearest-neighbour semi-Lagrangian advection of the *integer* phase field
`Phases` by the host-rock displacement field `Velocity`. `Velocity` is a tuple
of per-direction displacement arrays (same shape as `Phases`, in metres) — i.e.
exactly the field returned by [`inject_sills`](@ref) (injection inflation) and
[`deflate_hostrock!`](@ref) (eruption subsidence).

Each cell takes the phase of its *departure point* `x - u(x)`, snapped to the
nearest grid node and clamped to the domain. Because it is nearest-neighbour
(no interpolation), phase indices stay valid integers — this is what makes the
host rock (and previously-injected sills) move *with* the free surface instead
of staying pinned. Non-finite displacements are treated as zero. Mutates
`Phases` in place (via an internal copy of the source state) and returns it.
"""
function advect_phases!(Phases::AbstractArray, Velocity, Grid::GridData)
    Δ   = Grid.Δ
    N   = Grid.N
    dim = length(N)
    src = copy(Phases)
    _shift(u, d) = isfinite(u) ? round(Int, u / d) : 0
    if dim == 2
        dx, dz = Δ[1], Δ[2]
        @inbounds for j in axes(Phases, 2), i in axes(Phases, 1)
            ii = clamp(i - _shift(Velocity[1][i,j], dx), 1, N[1])
            jj = clamp(j - _shift(Velocity[2][i,j], dz), 1, N[2])
            Phases[i,j] = src[ii,jj]
        end
    elseif dim == 3
        dx, dy, dz = Δ[1], Δ[2], Δ[3]
        @inbounds for k in axes(Phases, 3), j in axes(Phases, 2), i in axes(Phases, 1)
            ii = clamp(i - _shift(Velocity[1][i,j,k], dx), 1, N[1])
            jj = clamp(j - _shift(Velocity[2][i,j,k], dy), 1, N[2])
            kk = clamp(k - _shift(Velocity[3][i,j,k], dz), 1, N[3])
            Phases[i,j,k] = src[ii,jj,kk]
        end
    else
        error("advect_phases!: only 2D and 3D grids are supported (got $(dim)D)")
    end
    return Phases
end

# nearest vertical-grid index to an elevation `zs` (z is a uniform, increasing range)
@inline function _znearest(zs, z)
    return clamp(round(Int, (zs - z[1]) / (z[2] - z[1])) + 1, 1, length(z))
end

"""
    advect_surface!(z_surf, Dz, Grid::GridData; clamp_to_domain=true)

Advect the free surface vertically by the host-rock displacement field `Dz`
(same shape as the temperature field), sampled at the current surface elevation
of each column (nearest vertical node). Injection inflation (`Dz>0`) raises the
surface; eruption deflation (`Dz<0`) lowers it. With `clamp_to_domain=true` the
surface is kept within the grid's vertical extent `[z₁, zₙ]`. Mutates and
returns `z_surf`.
"""
function advect_surface!(z_surf::AbstractArray, Dz::AbstractArray, Grid::GridData; clamp_to_domain::Bool=true)
    coord = Grid.coord1D
    dim   = length(coord)
    z     = coord[end]
    zlo, zhi = min(z[1], z[end]), max(z[1], z[end])
    if dim == 2
        @inbounds for i in eachindex(z_surf)
            k          = _znearest(z_surf[i], z)
            z_surf[i] += Dz[i,k]
            clamp_to_domain && (z_surf[i] = clamp(z_surf[i], zlo, zhi))
        end
    elseif dim == 3
        @inbounds for j in axes(z_surf, 2), i in axes(z_surf, 1)
            k            = _znearest(z_surf[i,j], z)
            z_surf[i,j] += Dz[i,j,k]
            clamp_to_domain && (z_surf[i,j] = clamp(z_surf[i,j], zlo, zhi))
        end
    else
        error("advect_surface!: only 2D and 3D grids are supported (got $(dim)D)")
    end
    return z_surf
end

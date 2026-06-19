# InjectSills_utils.jl
#
# Thin wrappers around the InjectSills.jl package that give the same
# workflow as InjectDike / AddDike in Dikes.jl, but rely entirely on
# the InjectSills API:
#
#   • hostrock_displacement  – displacement / velocity field
#   • inside                 – point-in-sill predicate
#   • new_point_inside_sill  – random tracer placement
#   • volume / area          – injected volume / area
#   • dike_polygon           – 2-D plotting outline
#
# What is NOT stored in AbstractSill and must be supplied as arguments:
#   • T_in     – temperature assigned to newly-intruded material [°C]
#   • Phase_in – rock-phase index assigned to newly-intruded tracers
#   • nTr_dike – number of new tracers to seed inside the intrusion
#
# The sill itself (including Center, Angle, W, H, E, ν, …) is fully
# described by the AbstractSill object passed in.  To inject at a
# different location or orientation, update the sill with
# `update_abstractsill(sill; Center=…, Angle=…)` before calling
# `inject_sills`.

"""
    p = random_point_in_sill(sill, dim)

Return a random point inside `sill`. Prefers the package sampler
`InjectSills.new_point_inside_sill`, but falls back to bounding-box rejection
sampling (built from `Center` ± `r`) for sphere sources (Mogi/McTigue) whose
`new_point_inside_sill` is not defined. Works for any source that supports
`InjectSills.inside`.
"""
function random_point_in_sill(sill::InjectSills.AbstractSill, dim::Integer)
    if hasproperty(sill, :W)
        return InjectSills.new_point_inside_sill(sill)      # native sampler (penny/elliptical sills)
    end
    # Sphere source: sample within the [Center-r, Center+r] box and reject
    c = sill.Center.val
    r = sill.r.val
    for _ in 1:100_000
        if dim == 2
            p = InjectSills.Point2{Float64}(c[1]-r+2r*rand(), c[2]-r+2r*rand())
        else
            p = InjectSills.Point3{Float64}(c[1]-r+2r*rand(), c[2]-r+2r*rand(), c[3]-r+2r*rand())
        end
        InjectSills.inside(p, sill) && return p
    end
    error("random_point_in_sill: could not find a point inside the sill after 1e5 tries")
end

"""
    Tnew, Tracers = add_dike(Tfield, Tracers, Grid, sill, T_in, Phase_in, nTr_dike)

Set the temperature to `T_in` at every grid point that lies inside `sill`, and
seed `nTr_dike` new tracers (with temperature `T_in` and phase `Phase_in`)
randomly distributed inside the sill.

This is the InjectSills-based replacement for `AddDike` in `Dikes.jl`.
The sill's center and orientation are encoded in the `sill` object itself;
no external rotation is required here.
"""
function add_dike(Tfield, Tr, Grid, sill::InjectSills.AbstractSill, T_in::Float64, Phase_in::Int64, nTr_dike::Int64)

    dim = length(Grid)

    # ------------------------------------------------------------------
    # 1.  Set temperature inside the sill
    # ------------------------------------------------------------------
    if dim == 2
        x, z = Grid[1], Grid[2]
        for ix in eachindex(x), iz in eachindex(z)
            pt = InjectSills.Point2{Float64}(x[ix], z[iz])
            if InjectSills.inside(pt, sill)
                Tfield[ix, iz] = T_in
            end
        end
    elseif dim == 3
        x, y, z = Grid[1], Grid[2], Grid[3]
        for ix in eachindex(x), iy in eachindex(y), iz in eachindex(z)
            pt = InjectSills.Point3{Float64}(x[ix], y[iy], z[iz])
            if InjectSills.inside(pt, sill)
                Tfield[ix, iy, iz] = T_in
            end
        end
    end

    # ------------------------------------------------------------------
    # 2.  Seed new tracers inside the sill
    # ------------------------------------------------------------------
    for _ in 1:nTr_dike

        pt = random_point_in_sill(sill, dim)           # Point{N, Float64} (sphere-source safe)

        number = isassigned(Tr, 1) ? Tr.num[end] + 1 : 1

        FT = isassigned(Tr, 1) ? eltype(Tr[1].time_vec) : Float32
        coord      = [Float64(pt[i]) for i in 1:dim]  # Vector{Float64}
        new_tracer = Tracer{FT}(num=number, coord=coord, T=T_in, Phase=Phase_in)

        if !isassigned(Tr, 1)
            if length(Tr) == 0
                StructArrays.foreachfield(v -> deleteat!(v, 1), Tr)
            end
            Tr = StructArray([new_tracer])
        else
            push!(Tr, new_tracer)
        end
    end

    return Tfield, Tr
end

# Accept generic numeric inputs and normalize to the concrete method used internally.
function add_dike(Tfield, Tr, Grid, sill::InjectSills.AbstractSill, T_in::Real, Phase_in::Integer, nTr_dike::Integer)
    return add_dike(Tfield, Tr, Grid, sill, Float64(T_in), Int64(Phase_in), Int64(nTr_dike))
end

"""
    stamp_phase_inside_sill!(Phases, Grid::GridData, sill, Phase_in)

Set `Phases = Phase_in` at every grid node that lies geometrically inside `sill`
(using `InjectSills.inside`). This produces a *dense*, gap-free sill in the phase
array — unlike reconstructing phases from sparse tracers
([`PhasesFromTracers!`](@ref) with `InterpolationMethod="Constant"`), which only
marks nodes that happen to have a tracer within one cell and therefore leaves
host-rock nodes scattered through the intrusion (the "spotty" sill). Mutates
`Phases` in place and returns it.
"""
function stamp_phase_inside_sill!(Phases::AbstractArray, Grid, sill::InjectSills.AbstractSill, Phase_in::Integer)
    coord = Grid.coord1D
    dim   = length(coord)
    if dim == 2
        x, z = coord[1], coord[2]
        @inbounds for j in eachindex(z), i in eachindex(x)
            if InjectSills.inside(InjectSills.Point2{Float64}(x[i], z[j]), sill)
                Phases[i,j] = Phase_in
            end
        end
    elseif dim == 3
        x, y, z = coord[1], coord[2], coord[3]
        @inbounds for k in eachindex(z), j in eachindex(y), i in eachindex(x)
            if InjectSills.inside(InjectSills.Point3{Float64}(x[i], y[j], z[k]), sill)
                Phases[i,j,k] = Phase_in
            end
        end
    else
        error("stamp_phase_inside_sill!: only 2D and 3D grids are supported (got $(dim)D)")
    end
    return Phases
end


"""
    Tracers, Tnew, InjectedVolume, dike_poly, Velocity =
        inject_sills(Tracers, T, Grid, sill, T_in, Phase_in, nTr_dike;
                     AdvectionMethod="RK2", InterpolationMethod="Linear",
                     dike_poly=[])

Inject a sill/dike described by the `InjectSills.AbstractSill` object `sill`
into the temperature field `T` defined on the regular grid `Grid`.

# Arguments
- `Tracers`             – `StructArray` of `Tracer` objects (may be unassigned on first call)
- `T`                   – temperature array [°C], mutated in-place
- `Grid`                – 1-D coordinate vectors `(x, z)` in 2-D or `(x, y, z)` in 3-D
- `sill`                – `AbstractSill` (e.g. `PennyShapedSill`) with the desired center,
                          orientation, size, and elastic parameters already set
- `T_in`                – temperature of the injected magma [°C]
- `Phase_in`            – rock-phase index assigned to new tracers
- `nTr_dike`            – number of new tracers to seed inside the sill

# Keyword arguments
- `AdvectionMethod`     – `"RK2"` (default) or `"Euler"`
- `InterpolationMethod` – `"Linear"`, `"Quadratic"`, or `"Cubic"` (default `"Linear"`)
- `dike_poly`           – optional plotting polygon that is advected with the host rock

# Returns
`(Tracers, Tnew, InjectedVolume, dike_poly, Velocity)`

## Algorithm
The sill is opened gradually over `nsteps` pseudo-time steps so that the
displacement per step stays below `0.5 * min(dx, dz)`.  For each pseudo-step
the temperature field and all existing tracers are advected using the
displacement field returned by `InjectSills.hostrock_displacement`.  After
pseudo-advection, `add_dike` sets `T = T_in` inside the sill and seeds the
new tracers.

The displacement field (= velocity for pseudo-time `dt_total = 1`) is
obtained directly from the sill object, which already encodes the center and
orientation of the intrusion — no external rotation is needed.
"""
function inject_sills(Tracers, T::Array, Grid,
                      sill::InjectSills.AbstractSill,
                      T_in::Float64, Phase_in::Int64, nTr_dike::Int64;
                      AdvectionMethod="RK2", InterpolationMethod="Linear",
                      dike_poly=[])

    dim = length(Grid)

    # ------------------------------------------------------------------
    # Build full-grid coordinate arrays
    # ------------------------------------------------------------------
    if dim == 2
        coords   = collect(Iterators.product(Grid[1], Grid[2]))
        X        = (c -> c[1]).(coords)
        Z        = (c -> c[2]).(coords)
        GridFull = (X, Z)
    elseif dim == 3
        coords   = collect(Iterators.product(Grid[1], Grid[2], Grid[3]))
        X        = (c -> c[1]).(coords)
        Y        = (c -> c[2]).(coords)
        Z        = (c -> c[3]).(coords)
        GridFull = (X, Y, Z)
    end

    # ------------------------------------------------------------------
    # Displacement field (= velocity for pseudo-time dt_total = 1.0)
    # hostrock_displacement handles centering + rotation internally.
    # ------------------------------------------------------------------
    if dim == 2
        Dx, Dz   = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Z))
        Velocity = (Dx, Dz)
    elseif dim == 3
        Dx, Dy, Dz = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Y), Float64.(Z))
        Velocity   = (Dx, Dy, Dz)
    end

    # ------------------------------------------------------------------
    # Regularize the displacement field (fixes the "interior blows up"
    # behaviour of point/sphere sources).
    #
    # The prescribed displacement moves HOST ROCK. Inside the intrusion the
    # material is (re)set to magma by `add_dike` below, so the field there is
    # irrelevant to the result — and for a Mogi/McTigue sphere the analytic
    # displacement is singular in the core (it grows like 1/r² toward the
    # centre, where this InjectSills version returns NaN for McTigue and
    # values up to ~1e33 m for both). Zeroing the displacement at every node
    # *inside* the sill removes that singular core. This is applied ONLY to
    # sphere/point sources (Mogi/McTigue, which lack W/H): the interior there
    # is the pressurized cavity / new magma, not displaced host rock. Penny /
    # elliptical sills are left exactly as before (their interior opening is a
    # real part of the kinematics). As a belt-and-braces guard we also zero any
    # remaining non-finite entry for all sources.
    # ------------------------------------------------------------------
    is_sphere_source = !(hasproperty(sill, :W) && hasproperty(sill, :H))   # Mogi/McTigue lack W,H
    if is_sphere_source
        @inbounds for ii in eachindex(X)
            ptin = dim == 2 ? InjectSills.Point2{Float64}(X[ii], Z[ii]) :
                              InjectSills.Point3{Float64}(X[ii], Y[ii], Z[ii])
            if InjectSills.inside(ptin, sill)
                for V in Velocity; V[ii] = 0.0; end
            end
        end
    end
    for V in Velocity
        @inbounds for i in eachindex(V)
            if !isfinite(V[i]); V[i] = 0.0; end
        end
    end

    # ------------------------------------------------------------------
    # Number of pseudo-time steps. Keep the per-step displacement below
    # 0.5*min(dx,dz) EVERYWHERE by sizing nsteps from the ACTUAL maximum of
    # the (host-rock) displacement field — not only from the sill thickness H.
    #
    # For a penny / elliptical sill the field max ≈ H/2 ≤ H, so the H-based
    # term dominates and nsteps (hence the result) is unchanged from before.
    # For a Mogi/McTigue source the near-source host-rock displacement is ≫ H,
    # so the field-based term takes over and supplies the extra sub-steps
    # needed. All arithmetic stays in Float until a final, clamped Int
    # conversion so a pathological field cannot overflow `ceil(Int, …)`.
    # ------------------------------------------------------------------
    Spacing  = [Grid[i][2] - Grid[i][1] for i in 1:dim]
    d        = minimum(Spacing) * 0.5
    # H-floor: penny/elliptical sills expose `H` (max opening); sphere
    # sources (Mogi/McTigue) do not. When absent the field-based count governs.
    H_floor  = hasproperty(sill, :H) ? sill.H.val : 0.0    # maximum opening thickness [m], if defined
    if dim == 2
        max_disp = maximum(@. sqrt(Dx^2 + Dz^2))
    else
        max_disp = maximum(@. sqrt(Dx^2 + Dy^2 + Dz^2))
    end
    nsteps_cap   = 100_000                                 # guard against a (near-)singular field
    ratio_field  = isfinite(max_disp) ? max_disp / d : Inf
    nsteps_req   = max(H_floor / d, ratio_field)           # Float; may be Inf
    nsteps       = clamp(isfinite(nsteps_req) ? ceil(Int, min(nsteps_req, Float64(nsteps_cap))) : nsteps_cap,
                         2, nsteps_cap)
    if ratio_field > nsteps_cap
        @warn "inject_sills: host-rock displacement field is near-singular (max |u| = $(round(max_disp, sigdigits=4)) m " *
              "⇒ ~$(round(Int, min(ratio_field, 1e18))) sub-steps requested, capped at $nsteps_cap). " *
              "The exterior advection is fine but very small grid spacing near the source is under-resolved."
    end
    dt = 1.0 / nsteps

    # ------------------------------------------------------------------
    # Pseudo-timestep advection: open the sill gradually
    # ------------------------------------------------------------------
    Tnew = zeros(size(T))
    for _ in 1:nsteps
        Tnew = AdvectTemperature(T, Grid, GridFull, Velocity, dt, AdvectionMethod, InterpolationMethod)
        if isassigned(Tracers, 1)
            AdvectTracers!(Tracers, Grid, Velocity, dt)
        end
        T .= Tnew
    end

    # ------------------------------------------------------------------
    # Set T = T_in inside the sill and seed new tracers
    # ------------------------------------------------------------------
    Tnew, Tracers = add_dike(T, Tracers, Grid, sill, T_in, Phase_in, nTr_dike)

    # ------------------------------------------------------------------
    # Injected volume — always the equivalent 3D volume [m³], in *both* 2D and
    # 3D. MTK reports fluxes/volumes in km³ regardless of model dimensionality (a
    # 2D run is treated as a slice through a 3D body), so we keep a true 3D volume
    # here. We take it from `InjectSills.volume`, implemented per source type
    # (e.g. EllipticalIntrusion `volume = 4/3·π·(W/2)²·(H/2)`, W,H = full
    # diameters). This still fixes the previous hand-rolled formula, which treated
    # W as a radius rather than a diameter and so over-estimated by a factor of 4.
    #
    # The 2D *eruptible* volume is lifted to a 3D volume separately (assuming a
    # Gaussian out-of-plane distribution) so it stays comparable to this injected
    # volume and to `EruptionParams.V_crit` — see `erupt_magma!`.
    #
    # (`volume` returns a Unitful.Quantity for most sources but a plain Float64
    # for PennyShapedSill, so `ustrip` gets the bare SI number in both cases.)
    # ------------------------------------------------------------------
    InjectedVolume = ustrip(InjectSills.volume(sill))

    # ------------------------------------------------------------------
    # Optionally advect a plotting polygon
    # ------------------------------------------------------------------
    if !isempty(dike_poly)
        # Keep polygon tracking local to InjectSills to avoid depending on legacy Dikes.jl symbols.
        poly_vel = AdvectPoints((dike_poly[1], dike_poly[2]), Grid, Velocity, 1.0)
        for i in eachindex(dike_poly)
            dike_poly[i] .+= poly_vel[i]
        end
    end

    return Tracers, Tnew, InjectedVolume, dike_poly, Velocity
end

# Convenience overload to accept Int/Float combinations from user-facing scripts.
function inject_sills(Tracers, T::Array, Grid,
                      sill::InjectSills.AbstractSill,
                      T_in::Real, Phase_in::Integer, nTr_dike::Integer;
                      AdvectionMethod="RK2", InterpolationMethod="Linear",
                      dike_poly=[])
    return inject_sills(Tracers, T, Grid, sill, Float64(T_in), Int64(Phase_in), Int64(nTr_dike);
                        AdvectionMethod=AdvectionMethod,
                        InterpolationMethod=InterpolationMethod,
                        dike_poly=dike_poly)
end


# =====================================================================
#  Eruptions
# =====================================================================

"""
    V, mask = eruptible_volume(ϕ, Δ, ϕ_erupt; zc=nothing, EruptAbove=-Inf)

Volume of *eruptible* melt: the melt contained in cells whose melt fraction `ϕ`
exceeds `ϕ_erupt`. `Δ` is the grid spacing tuple. Returns the volume [m³] (in 2D
a per-unit-depth volume = area·1 m) and the boolean eruptible-cell `mask`.

An optional **depth cap** excludes deep cells: when `zc` (the vertical-coordinate
vector along the last dimension of `ϕ`) is supplied and `EruptAbove` is finite,
only cells at elevation `zc ≥ EruptAbove` can be eruptible. This prevents deep
background melt (e.g. host-rock partial melt under a hot geotherm at the base of
the domain) from being counted as eruptible chamber.
"""
function eruptible_volume(ϕ::AbstractArray, Δ, ϕ_erupt::Real; zc=nothing, EruptAbove::Real=-Inf)
    Vcell = prod(filter(>(0), collect(Δ)))     # cell area (2D) / volume (3D)
    mask  = ϕ .> ϕ_erupt
    if zc !== nothing && isfinite(EruptAbove)   # exclude cells below the eruptible floor
        dim = ndims(ϕ)
        @inbounds for I in CartesianIndices(ϕ)
            if mask[I] && zc[I[dim]] < EruptAbove
                mask[I] = false
            end
        end
    end
    V     = 0.0
    @inbounds for i in eachindex(ϕ)
        if mask[i]
            V += ϕ[i]*Vcell
        end
    end
    return V, mask
end

"""
    deflate_hostrock!(T, Tracers, Grid, src; AdvectionMethod, InterpolationMethod)

Advect the host rock (temperature field `T` and `Tracers`) by the displacement of
the (withdrawal / negative-ΔP) Mogi `src`, i.e. the eruption counterpart of
`inject_sills` without intruding any new magma. Uses the same interior-mask /
non-finite sanitation / sub-stepping safeguards as `inject_sills` so the
near-source field cannot blow up. Mutates `T` and `Tracers` in place.
"""
function deflate_hostrock!(T::Array, Tracers, Grid, src::InjectSills.AbstractSill;
                           AdvectionMethod="RK2", InterpolationMethod="Linear")
    dim = length(Grid)
    if dim == 2
        coords = collect(Iterators.product(Grid[1], Grid[2]))
        X = (c->c[1]).(coords); Z = (c->c[2]).(coords)
        GridFull = (X, Z)
        Dx, Dz   = InjectSills.hostrock_displacement(src, Float64.(X), Float64.(Z)); Velocity = (Dx, Dz)
    else
        coords = collect(Iterators.product(Grid[1], Grid[2], Grid[3]))
        X = (c->c[1]).(coords); Y = (c->c[2]).(coords); Z = (c->c[3]).(coords)
        GridFull = (X, Y, Z)
        Dx, Dy, Dz = InjectSills.hostrock_displacement(src, Float64.(X), Float64.(Y), Float64.(Z)); Velocity = (Dx, Dy, Dz)
    end

    # interior mask (sphere source) + non-finite sanitation — see inject_sills.
    @inbounds for ii in eachindex(X)
        ptin = dim == 2 ? InjectSills.Point2{Float64}(X[ii], Z[ii]) :
                          InjectSills.Point3{Float64}(X[ii], Y[ii], Z[ii])
        if InjectSills.inside(ptin, src)
            for V in Velocity; V[ii] = 0.0; end
        end
    end
    for V in Velocity, i in eachindex(V)
        @inbounds if !isfinite(V[i]); V[i] = 0.0; end
    end

    Spacing = [Grid[i][2] - Grid[i][1] for i in 1:dim]
    d       = minimum(Spacing)*0.5
    max_disp = dim == 2 ? maximum(@. sqrt(Dx^2 + Dz^2)) : maximum(@. sqrt(Dx^2 + Dy^2 + Dz^2))
    nsteps_cap = 100_000
    ratio    = isfinite(max_disp) ? max_disp/d : Inf
    nsteps   = clamp(isfinite(ratio) ? ceil(Int, min(ratio, Float64(nsteps_cap))) : nsteps_cap, 2, nsteps_cap)
    dt       = 1.0/nsteps

    Tnew = copy(T)
    for _ in 1:nsteps
        Tnew = AdvectTemperature(T, Grid, GridFull, Velocity, dt, AdvectionMethod, InterpolationMethod)
        if isassigned(Tracers, 1)
            AdvectTracers!(Tracers, Grid, Velocity, dt)
        end
        T .= Tnew
    end
    return T, Tracers, Velocity
end

"""
    deflate_hostrock!(T, Tracers, Grid, srcs::AbstractVector; AdvectionMethod, InterpolationMethod)

Multi-source variant of [`deflate_hostrock!`](@ref): advect the host rock by the
**superposition** of the displacement fields of several sources `srcs`. Each
source's displacement is added into the total field, with that source's own
singular interior (the points `InjectSills.inside(pt, src)`) and any non-finite
entries excluded from *its* contribution — exactly the per-source regularization
the single-source method applies. The summed field is then sub-stepped and used
to advect `T` and `Tracers` with the same CFL/`nsteps` safeguards. This is what
gives a spatially irregular (melt-distribution-following) chamber deflation when
`erupt_magma!` builds one source per eruptible cell. Returns `(T, Tracers, Velocity)`
with `Velocity` the summed displacement field.
"""
function deflate_hostrock!(T::Array, Tracers, Grid, srcs::AbstractVector;
                           AdvectionMethod="RK2", InterpolationMethod="Linear")
    dim = length(Grid)
    if dim == 2
        coords = collect(Iterators.product(Grid[1], Grid[2]))
        X = Float64.((c->c[1]).(coords)); Z = Float64.((c->c[2]).(coords))
        GridFull = (X, Z)
        Velocity = (zeros(size(X)), zeros(size(Z)))
    else
        coords = collect(Iterators.product(Grid[1], Grid[2], Grid[3]))
        X = Float64.((c->c[1]).(coords)); Y = Float64.((c->c[2]).(coords)); Z = Float64.((c->c[3]).(coords))
        GridFull = (X, Y, Z)
        Velocity = (zeros(size(X)), zeros(size(Y)), zeros(size(Z)))
    end

    # superpose each source's field; skip its own singular core + non-finite entries
    for src in srcs
        comp = dim == 2 ? InjectSills.hostrock_displacement(src, X, Z) :
                          InjectSills.hostrock_displacement(src, X, Y, Z)
        @inbounds for ii in eachindex(X)
            ptin = dim == 2 ? InjectSills.Point2{Float64}(X[ii], Z[ii]) :
                              InjectSills.Point3{Float64}(X[ii], Y[ii], Z[ii])
            InjectSills.inside(ptin, src) && continue        # this source's own interior
            for d in 1:dim
                v = comp[d][ii]
                isfinite(v) && (Velocity[d][ii] += v)
            end
        end
    end
    for V in Velocity, i in eachindex(V)                     # final sanitation of the sum
        @inbounds if !isfinite(V[i]); V[i] = 0.0; end
    end

    Spacing = [Grid[i][2] - Grid[i][1] for i in 1:dim]
    d       = minimum(Spacing)*0.5
    max_disp = dim == 2 ? maximum(@. sqrt(Velocity[1]^2 + Velocity[2]^2)) :
                          maximum(@. sqrt(Velocity[1]^2 + Velocity[2]^2 + Velocity[3]^2))
    nsteps_cap = 100_000
    ratio    = isfinite(max_disp) ? max_disp/d : Inf
    nsteps   = clamp(isfinite(ratio) ? ceil(Int, min(ratio, Float64(nsteps_cap))) : nsteps_cap, 2, nsteps_cap)
    dt       = 1.0/nsteps

    Tnew = copy(T)
    for _ in 1:nsteps
        Tnew = AdvectTemperature(T, Grid, GridFull, Velocity, dt, AdvectionMethod, InterpolationMethod)
        if isassigned(Tracers, 1)
            AdvectTracers!(Tracers, Grid, Velocity, dt)
        end
        T .= Tnew
    end
    return T, Tracers, Velocity
end

# Build one negative-Mogi source per eruptible cell, sized to the melt withdrawn
# from that cell (ΔV_i = η·ϕ_i·Vcell ⇒ a_i = radius of a sphere/disk of that
# volume), centred on the cell. The summed source strength Σ ΔV_i = η·Ve, so the
# per-cell deflation removes exactly the booked erupted volume — distributed where
# the melt actually is. `coord` is `Grid.coord1D`; `Vcell` is the cell volume (2D:
# area, per-unit-depth).
function _percell_deflation_sources(ϕ, mask, coord, Vcell::Real, η::Real, Erupt::EruptionParameters, dim::Integer)
    srcs = InjectSills.AbstractSill[]
    @inbounds for I in CartesianIndices(ϕ)
        mask[I] || continue
        ΔV = η * ϕ[I] * Vcell
        ΔV > 0 || continue
        if dim == 2
            a   = sqrt(ΔV/π)                                  # disk radius (per-unit-depth)
            ctr = InjectSills.Point2{Float64}(coord[1][I[1]], coord[2][I[2]])
        else
            a   = cbrt(3*ΔV/(4π))                             # sphere radius
            ctr = InjectSills.Point3{Float64}(coord[1][I[1]], coord[2][I[2]], coord[3][I[3]])
        end
        push!(srcs, MogiSphere(Center = ctr*m, r = a*m,
                               ΔP = (-Erupt.ΔP)*Pa, G = Erupt.G*Pa, ν = Erupt.ν*NoUnits))
    end
    return srcs
end

"""
    erupted = erupt_magma!(T, ϕ, dϕdT, Tracers, Grid, Erupt, time)

Erupt magma when the eruptible volume reaches the critical volume. Steps:
1. Compute the eruptible volume `Ve` (melt in cells with `ϕ > Erupt.ϕ_erupt`). In
   2D this is per-unit-depth; when `Erupt.out_of_plane_3D` it is lifted to a true
   3D volume via a Gaussian out-of-plane profile so it is comparable (in km³) to
   `V_crit` and the injected volume.
2. If `Ve < Erupt.V_crit`, do nothing and return `false`.
3. Otherwise remove a fraction `Erupt.erupt_efficiency` of the eruptible melt by
   *thermal extraction*: each eruptible cell is cooled by
   `ΔT = efficiency·(ϕ - ϕ_erupt)/(dϕ/dT)`, the linearized temperature drop that
   removes the extracted melt (uses the locally-available `dϕ/dT`).
4. If `Erupt.deflate`, also deflate the chamber with a negative-ΔP Mogi source
   centred on the melt-weighted centroid of the eruptible region (host-rock
   subsidence; this is what a moving free surface will track — see issue 4).
5. Update the bookkeeping on `Erupt` (count, cumulative + per-event volume/time).

`T`, `ϕ`, `dϕdT` are the (CPU) temperature, melt-fraction and dϕ/dT arrays; they
are mutated in place. Returns `true` if an eruption occurred.

If a free-surface topography `z_surf` is supplied (keyword), the host-rock
subsidence of the deflation source is also applied to it, so the ground surface
tracks the chamber deflation (issue 4). Likewise, if a phase field `Phases` is
supplied (keyword), it is advected by the same subsidence so the material column
moves with the deflation. Both are only used on the deflation path
(`Erupt.deflate == true`).
"""
function erupt_magma!(T::Array, ϕ::Array, dϕdT::Array, Tracers, Grid, Erupt::EruptionParameters, time::Real; z_surf=nothing, Phases=nothing)
    Erupt.erupt || return false
    Δ      = Grid.Δ
    coord  = Grid.coord1D
    dim    = length(coord)

    Ve, mask = eruptible_volume(ϕ, Δ, Erupt.ϕ_erupt; zc=coord[end], EruptAbove=Erupt.EruptAbove)

    Vcell   = prod(filter(>(0), collect(Δ)))

    # melt-weighted centroid + bulk volume of the eruptible region (used for the
    # deflation source and, in 2D, for the out-of-plane Gaussian volume). `sx2` is
    # the weighted Σx² needed for the horizontal standard deviation in 2D.
    sx = sy = sz = 0.0; sx2 = 0.0; wsum = 0.0; Vbulk = 0.0
    if dim == 2
        @inbounds for j in axes(ϕ,2), i in axes(ϕ,1)
            if mask[i,j]
                x = coord[1][i]
                w = ϕ[i,j]; wsum += w; Vbulk += Vcell
                sx += w*x; sx2 += w*x^2; sz += w*coord[2][j]
            end
        end
    else
        @inbounds for k in axes(ϕ,3), j in axes(ϕ,2), i in axes(ϕ,1)
            if mask[i,j,k]
                w = ϕ[i,j,k]; wsum += w; Vbulk += Vcell
                sx += w*coord[1][i]; sy += w*coord[2][j]; sz += w*coord[3][k]
            end
        end
    end
    wsum == 0 && return false                   # no eruptible melt ⇒ nothing to do
    cx = sx/wsum

    # (1→3D) In 2D, lift the per-unit-depth eruptible volume to a true 3D volume
    # by assuming a Gaussian out-of-plane (y) melt profile: the effective out-of-
    # plane length is √(2π)·σ, with σ the melt-weighted horizontal half-width of
    # the eruptible region (floored at the in-plane spacing so even a single cell
    # gets a sensible thickness). This keeps `Ve` in km³ — directly comparable to
    # `V_crit` and the (3D) injected volume. No-op in 3D, where `Ve` is already a
    # volume.
    if dim == 2 && Erupt.out_of_plane_3D
        σx  = sqrt(max(sx2/wsum - cx^2, 0.0))
        Ve *= sqrt(2π) * max(σx, Δ[1])
    end

    Ve < Erupt.V_crit && return false           # trigger on the (3D) eruptible volume
    V_erupt = Erupt.erupt_efficiency * Ve

    # Freeze the tracers caught up in the eruption — a random fraction
    # `erupt_efficiency` of the melt-rich (eruptible-cell) tracers — *before* any
    # deflation advection below, so they keep their eruption-instant position and
    # Tt-path. Their preserved history is the erupted "zircon cargo". The final
    # history point is recorded in Myr to match the tracer time convention used by
    # the drivers (see `update_Tvec!`).
    freeze_erupted_tracers!(Tracers, Grid, mask, Erupt.erupt_efficiency, time/SecYear*1e-6)

    # deflation-source geometry (melt-weighted centroid + characteristic radius)
    if dim == 2
        a = sqrt(Vbulk/π)                       # characteristic radius (2D: area→radius)
        center = InjectSills.Point2{Float64}(cx, sz/wsum)
    else
        a = cbrt(3*Vbulk/(4π))                  # characteristic radius (3D)
        center = InjectSills.Point3{Float64}(cx, sy/wsum, sz/wsum)
    end

    # (3) withdraw a fraction η of the mobile melt from each eruptible cell:
    #     ϕ → (1-η)·ϕ, realized thermally by cooling the cell by ΔT = η·ϕ/(dϕ/dT)
    #     (the local linearized drop that removes η·ϕ of melt). Booked volume
    #     (below) = Σ η·ϕ·Vcell = η·Ve, exactly the melt removed here.
    η = Erupt.erupt_efficiency
    @inbounds for i in eachindex(ϕ)
        if mask[i]
            if dϕdT[i] > 0
                # cool to remove η·ϕ of melt; clamp at Erupt.T_min so a tiny
                # dϕdT (flat melt curve near saturation) cannot drive T to an
                # unphysical value that would destabilize the diffusion solver.
                T[i] = max(T[i] - η*ϕ[i]/dϕdT[i], Erupt.T_min)
            end
            ϕ[i] = (1 - η)*ϕ[i]            # withdraw η of the mobile melt
        end
    end

    # (4) optional kinematic chamber deflation (negative-Mogi withdrawal)
    if Erupt.deflate
        if Erupt.deflate_percell
            # per-cell superposition (default): one negative-Mogi source per
            # eruptible cell, sized to the melt withdrawn there (ΔV_i = η·ϕ_i·Vcell),
            # so the subsidence follows the irregular melt distribution and the total
            # subsidence volume = Σ ΔV_i = η·Ve (the booked erupted volume).
            srcs = _percell_deflation_sources(ϕ, mask, coord, Vcell, η, Erupt, dim)
            _, _, Vel = deflate_hostrock!(T, Tracers, coord, srcs)
        else
            # region-scale (opt-out): a single source at the melt-weighted centroid,
            # radius from the bulk eruptible volume — spatially symmetric, cheaper.
            src = MogiSphere(Center = center*m, r = a*m,
                             ΔP = (-Erupt.ΔP)*Pa, G = Erupt.G*Pa, ν = Erupt.ν*NoUnits)
            _, _, Vel = deflate_hostrock!(T, Tracers, coord, src)
        end
        # track the host-rock subsidence on the free surface, if one is supplied
        if !isnothing(z_surf)
            advect_surface!(z_surf, Vel[end], Grid)
        end
        # advect the phase field with the same subsidence, if one is supplied
        if !isnothing(Phases)
            advect_phases!(Phases, Vel, Grid)
        end
    end

    # (5) bookkeeping
    Erupt.n_eruptions   += 1
    Erupt.erupted_volume += V_erupt
    push!(Erupt.eruption_times,   Float64(time))
    push!(Erupt.eruption_volumes, V_erupt)
    return true
end

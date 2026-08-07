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
    # centre). Zeroing the displacement at every node
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
    enthalpy(T, Rho, Cp, Hl, ϕ, Grid::GridData)

Total thermal enthalpy of the domain [J] (in 2D a per-unit-depth energy, J/m):

    Σ ρ · (Cp·T + Hl·ϕ) · Vcell

i.e. the sensible heat plus the latent heat carried by the melt fraction, summed
over cells. `T` is in °C; `Rho`, `Cp`, `Hl`, `ϕ` are the density, heat-capacity,
latent-heat and melt-fraction arrays that the diffusion step already fills on
`Arrays`. `Vcell` is the cell area (2D) / volume (3D), matching
[`eruptible_volume`](@ref).

Conservation diagnostic: snapshot `enthalpy` before and after an injection or an
[`erupt_magma!`](@ref) call. Injection adds enthalpy (hot magma) and eruptive
withdrawal removes it, so a like-for-like drift is the *residual* beyond those
intended changes — the numerical energy drift from advecting `T` intensively
(non-conservatively) with the kinematic host-rock displacement. Measure it before
building any conservative energy remap. Mirrors QMagma's `column_enthalpy`.
"""
function enthalpy(T, Rho, Cp, Hl, ϕ, Grid::GridData)
    Vcell = prod(filter(>(0), collect(Grid.Δ)))
    H = 0.0
    for i in eachindex(T, Rho, Cp, Hl, ϕ)
        H += Rho[i] * (Cp[i]*T[i] + Hl[i]*ϕ[i])
    end
    return H * Vcell
end

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

# Sub-step advection of `T` and `Tracers` by a prescribed displacement field
# `Velocity` (one array per grid dimension, matching the grid shape). Non-finite
# entries are zeroed, then the step is split into `nsteps` so the per-sub-step
# displacement stays ≲ half a cell (semi-Lagrangian stability). `GridFull` holds
# the full coordinate arrays. Shared by every `deflate_hostrock!` method.
function _advect_hostrock!(T, Tracers, Grid, GridFull, Velocity;
                           AdvectionMethod="RK2", InterpolationMethod="Linear")
    dim = length(Grid)
    for V in Velocity, i in eachindex(V)
        @inbounds if !isfinite(V[i]); V[i] = 0.0; end
    end

    Spacing  = [Grid[i][2] - Grid[i][1] for i in 1:dim]
    d        = minimum(Spacing)*0.5
    mag2     = reduce((s, V) -> s .+ V.^2, Velocity; init = zero(first(Velocity)))
    max_disp = maximum(sqrt, mag2)
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

# Build the full-coordinate arrays `(X, Z)` / `(X, Y, Z)` for a 1D-coordinate Grid.
function _grid_full(Grid)
    if length(Grid) == 2
        coords = collect(Iterators.product(Grid[1], Grid[2]))
        return (Float64.((c->c[1]).(coords)), Float64.((c->c[2]).(coords)))
    else
        coords = collect(Iterators.product(Grid[1], Grid[2], Grid[3]))
        return (Float64.((c->c[1]).(coords)), Float64.((c->c[2]).(coords)), Float64.((c->c[3]).(coords)))
    end
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
    dim      = length(Grid)
    GridFull = _grid_full(Grid)
    if dim == 2
        X, Z = GridFull
        Dx, Dz = InjectSills.hostrock_displacement(src, X, Z); Velocity = (Dx, Dz)
    else
        X, Y, Z = GridFull
        Dx, Dy, Dz = InjectSills.hostrock_displacement(src, X, Y, Z); Velocity = (Dx, Dy, Dz)
    end

    # zero this source's singular interior (see inject_sills)
    @inbounds for ii in eachindex(GridFull[1])
        ptin = dim == 2 ? InjectSills.Point2{Float64}(GridFull[1][ii], GridFull[2][ii]) :
                          InjectSills.Point3{Float64}(GridFull[1][ii], GridFull[2][ii], GridFull[3][ii])
        if InjectSills.inside(ptin, src)
            for V in Velocity; V[ii] = 0.0; end
        end
    end

    return _advect_hostrock!(T, Tracers, Grid, GridFull, Velocity; AdvectionMethod, InterpolationMethod)
end

"""
    deflate_hostrock!(T, Tracers, Grid, srcs::AbstractVector; AdvectionMethod, InterpolationMethod)

Multi-source variant of [`deflate_hostrock!`](@ref): advect the host rock by the
**superposition** of the displacement fields of several sources `srcs`. Each
source's displacement is added into the total field, with that source's own
singular interior (the points `InjectSills.inside(pt, src)`) and any non-finite
entries excluded from *its* contribution — exactly the per-source regularization
the single-source method applies. The summed field is then sub-stepped and used
to advect `T` and `Tracers` with the same CFL/`nsteps` safeguards. Returns
`(T, Tracers, Velocity)` with `Velocity` the summed displacement field.
"""
function deflate_hostrock!(T::Array, Tracers, Grid, srcs::AbstractVector;
                           AdvectionMethod="RK2", InterpolationMethod="Linear")
    dim      = length(Grid)
    GridFull = _grid_full(Grid)
    Velocity = ntuple(_ -> zeros(size(GridFull[1])), dim)

    # superpose each source's field; skip its own singular core + non-finite entries
    for src in srcs
        comp = dim == 2 ? InjectSills.hostrock_displacement(src, GridFull[1], GridFull[2]) :
                          InjectSills.hostrock_displacement(src, GridFull[1], GridFull[2], GridFull[3])
        @inbounds for ii in eachindex(GridFull[1])
            ptin = dim == 2 ? InjectSills.Point2{Float64}(GridFull[1][ii], GridFull[2][ii]) :
                              InjectSills.Point3{Float64}(GridFull[1][ii], GridFull[2][ii], GridFull[3][ii])
            InjectSills.inside(ptin, src) && continue        # this source's own interior
            for d in 1:dim
                v = comp[d][ii]
                isfinite(v) && (Velocity[d][ii] += v)
            end
        end
    end

    return _advect_hostrock!(T, Tracers, Grid, GridFull, Velocity; AdvectionMethod, InterpolationMethod)
end

# Vertical displacement field of a volume-conserving chamber deflation: the host
# rock at each node sinks by the total melt void opened *beneath* it in its
# column. Withdrawing `η·ϕ` of a cell frees a thickness `η·ϕ·Δz` (Δz = vertical
# spacing); walking each column from the bottom up and accumulating that void
# gives `Dz`, purely vertical (rigid overburden — no horizontal motion). The top
# of every column (the surface) sinks by the full column void `Σ η·ϕ·Δz`, so the
# surface-subsidence integral `Σ Dz_top·dA = Σ η·ϕ·Vcell = η·Ve` equals the
# booked erupted volume exactly — no elastic parameters, no source geometry. `ϕ`
# is the pre-withdrawal melt fraction; the vertical axis is the last dimension.
function _column_subsidence_velocity(ϕ, mask, Δz::Real, η::Real, dim::Integer)
    Velocity = ntuple(_ -> zeros(size(ϕ)), dim)
    Dz = Velocity[end]
    if dim == 2
        @inbounds for i in axes(ϕ, 1)
            cum = 0.0
            for k in axes(ϕ, 2)                     # bottom (k=1) → top
                Dz[i, k] = -cum
                mask[i, k] && (cum += η*ϕ[i, k]*Δz)
            end
        end
    else
        @inbounds for j in axes(ϕ, 2), i in axes(ϕ, 1)
            cum = 0.0
            for k in axes(ϕ, 3)
                Dz[i, j, k] = -cum
                mask[i, j, k] && (cum += η*ϕ[i, j, k]*Δz)
            end
        end
    end
    return Velocity
end

# Horizontal index window (a `lo:hi` UnitRange) within `rcut` of grid index
# `Ik` along one horizontal dimension, capped at `max_cells` and clamped to the
# domain — independent of `Δ`, so the cost never grows with grid resolution.
_hwindow(Ik::Integer, rcut::Real, Δk::Real, Nk::Integer, max_cells::Integer) =
    (n = min(ceil(Int, rcut/Δk), max_cells); max(1, Ik-n):min(Nk, Ik+n))

"""
    Vel = _local_mogi_deflation_velocity(ϕ, mask, coord, Δ, Vcell, η, dim;
                                          cutoff_factor=4.0, max_window_cells=25)

Volume-conserving, spatially-smoothed chamber deflation: superpose one
negative-Mogi source per eruptible cell, using the same `U(R) ∝ (p-Center)/R³`
radial shape as [`deflate_hostrock!`](@ref). The closed form is inlined here
rather than routed through `InjectSills.hostrock_displacement`/`MogiSphere`,
whose Unitful dispatch is ~100-500× too slow to evaluate once per eruptible
cell.

Each source is evaluated over its full vertical reach — its field has to
reach the surface to deflate it at all — but only within `max_window_cells`
cells *horizontally*, bounding the cost at `O(N_eruptible × K)` with `K` the
horizontal window width, capped independently of `Δ`. `cutoff_factor` sets
the horizontal reach as a multiple of each source's own depth before that cap
applies.

The per-source amplitude is proportional to *that cell's own* withdrawn
volume rather than to a physical `ΔP`/`G`/`ν` through the Mogi volume
relation: at crustal `G` and `ν`, expressing a realistic chamber volume
change requires either `ΔP ≈ G/(3(1-ν))` ~ 4.4 GPa or a source radius larger
than the domain. The summed field is instead rescaled so the swept volume at
the top of the domain equals the true withdrawn volume `η·Ve` exactly, which
also absorbs whatever the horizontal cutoff truncated away. The kernel's
constants therefore set its relative shape only, never its final amplitude.
"""
function _local_mogi_deflation_velocity(ϕ, mask, coord, Δ, Vcell::Real, η::Real, dim::Integer;
                                         cutoff_factor::Real=4.0, max_window_cells::Integer=25)
    N        = length.(coord)
    Velocity = ntuple(_ -> zeros(size(ϕ)), dim)
    a2       = (0.5*minimum(Δ))^2         # guard radius, squared (shape only, not volume-derived)
    shape    = 1 - 0.25                   # (1-ν), fixed ν=0.25 (shape only, not physical here)
    rcut_min = 2*minimum(Δ)               # floor so very shallow melt still gets a few cells of spread

    @inbounds if dim == 2
        for I in CartesianIndices(ϕ)
            mask[I] || continue
            ΔV = η*ϕ[I]*Vcell
            ΔV > 0 || continue
            cx, cz = coord[1][I[1]], coord[2][I[2]]
            d      = abs(cz)
            rcut   = max(cutoff_factor*d, rcut_min)
            K      = -ΔV*shape                        # negative ⇒ withdrawal (points move toward the source)
            for k in axes(ϕ, 2), i in _hwindow(I[1], rcut, Δ[1], N[1], max_window_cells)
                Δx = coord[1][i]-cx; Δz = coord[2][k]-cz
                R2 = Δx^2+Δz^2
                R2 < a2 && continue
                C  = K/R2^1.5
                Velocity[1][i,k] += C*Δx
                Velocity[2][i,k] += C*Δz
            end
        end
    else
        for I in CartesianIndices(ϕ)
            mask[I] || continue
            ΔV = η*ϕ[I]*Vcell
            ΔV > 0 || continue
            cx, cy, cz = coord[1][I[1]], coord[2][I[2]], coord[3][I[3]]
            d      = abs(cz)
            rcut   = max(cutoff_factor*d, rcut_min)
            K      = -ΔV*shape
            jr = _hwindow(I[2], rcut, Δ[2], N[2], max_window_cells)
            ir = _hwindow(I[1], rcut, Δ[1], N[1], max_window_cells)
            for k in axes(ϕ, 3), j in jr, i in ir
                Δx = coord[1][i]-cx; Δy = coord[2][j]-cy; Δz = coord[3][k]-cz
                R2 = Δx^2+Δy^2+Δz^2
                R2 < a2 && continue
                C  = K/R2^1.5
                Velocity[1][i,j,k] += C*Δx
                Velocity[2][i,j,k] += C*Δy
                Velocity[3][i,j,k] += C*Δz
            end
        end
    end

    # Rescale so the swept volume at the top of the domain equals -η·Ve exactly
    # (negative: the surface subsides; matches _column_subsidence_velocity's sign
    # convention and mass_budget's Δsurface = -erupted for a deflation-only event).
    # Corrects for the cutoff truncation and the arbitrary a/shape constants.
    if dim == 2
        top_sum = sum(@view Velocity[end][:, N[2]]) * Δ[1]
    else
        top_sum = sum(@view Velocity[end][:, :, N[3]]) * Δ[1]*Δ[2]
    end
    target = -η * sum(Vcell*ϕ[I] for I in CartesianIndices(ϕ) if mask[I]; init=0.0)
    scale  = (isfinite(top_sum) && top_sum != 0) ? target/top_sum : 0.0
    for V in Velocity, i in eachindex(V)
        V[i] *= scale
    end
    return Velocity
end

"""
    ρ_fn = magma_density_fn(density, solub, Erupt::EruptionParameters)

Build the `ρ(T_K, ϕ, P) -> kg/m³` density callback consumed by
[`step_overpressure!`](@ref) from a `Mat_tup` entry's `Density` and
`Solubility` laws (`Mat_tup[idx].Density[1]`, `Mat_tup[idx].Solubility[1]`).

For a plain melt+crystal law (`ConstantDensity`, `MeltDependent_Density`, ...)
this is `compute_density(density, (; T=T_K, P=P, ϕ=ϕ))` — `ϕ` is the melt
fraction, unrelated to any gas phase; `solub` is unused.

For `GeoParams.ThreePhase_Density` (the Degruyter & Huber 2014 melt+crystal+gas
mixture), the exsolved gas fraction is diagnosed at each call from a
quasi-equilibrium water mass balance: `solub` (e.g. `Liu2005_Solubility`) gives
the dissolved H₂O mass fraction of the melt at `(P, T_K, Erupt.X_co2)`; the
excess of `Erupt.m_h2o_total` over that solubility limit exsolves, converted to
a volume fraction `ϕ_gas` via the gas-free (melt+crystal) density and
`density.ρgas`. This is the `ρ(P,T,m_w)` closure the eruptions page documents
as needed for crystallization to genuinely *pressurize* a closed chamber (E4)
via second boiling, rather than only depressurize it.

`Erupt.m_h2o_total` is a **fixed constant**, not a chamber state — this is the
constant-`m_w` approximation, not a genuine mass-conserving water budget.
Recharge and eruption do not add or remove water from it, so second-boiling
strength does not respond to how much wet magma has actually accumulated or
erupted. A real water budget needs `m_w` (or the exsolved-gas fraction itself)
as its own state on `ChamberState`, integrated through `Ṁ_in` and the erupted
mass, alongside `P` — not rederived from a fixed total at every call.

!!! warning "`density.ρgas` is also evaluated outside this function"
    A `Mat_tup` entry's `Density` law is not exclusive to this callback — the
    main solver's `compute_density_ps!` evaluates it every cell/iteration too,
    with only `(T, P)` (`ϕ_gas`/`ϕ_x` default to 0). `ThreePhase_Density` still
    *evaluates* `ρgas` there to form the (zero-weighted) mixture sum, so a
    `ρgas` law with a narrow validity window — e.g. `RedlichKwong_Density`,
    fitted for ~30-400 MPa/873-1173 K — can return `NaN` at an out-of-range
    cell and poison the field even at zero weight (`0*NaN = NaN`). Prefer
    `IdealGas_Density` (finite for any `T>0`, `P≥0`) unless every cell of the
    phase using this `Density` law is guaranteed to stay within `ρgas`'s
    validity range for its whole thermal history.
"""
function magma_density_fn(density, solub, Erupt::EruptionParameters)
    if !(density isa ThreePhase_Density)
        return (T_K, ϕ, P) -> GeoParams.compute_density(density, (; T = T_K, P = P, ϕ = ϕ))
    end

    solub === nothing && error("magma_density_fn: a ThreePhase_Density magma phase requires its Mat_tup entry's Solubility to be set (a GeoParams AbstractSolubility, e.g. Liu2005_Solubility())")
    m_h2o_total, X_co2 = Erupt.m_h2o_total, Erupt.X_co2

    return function (T_K, ϕ, P)
        m_exsolved = max(m_h2o_total - compute_dissolved(solub, P, T_K, X_co2)[1], 0.0)

        ρ_melt = GeoParams.compute_density(density.ρmelt, (; T = T_K, P = P))
        ρ_x    = GeoParams.compute_density(density.ρx, (; T = T_K, P = P))
        ρ_gas  = GeoParams.compute_density(density.ρgas, (; T = T_K, P = P))
        ρ_lc   = ϕ*ρ_melt + (1 - ϕ)*ρ_x                    # gas-free melt+crystal density

        ϕ_gas = (m_exsolved/ρ_gas) / (m_exsolved/ρ_gas + (1 - m_exsolved)/ρ_lc)
        return (1 - ϕ_gas)*ρ_lc + ϕ_gas*ρ_gas
    end
end

"""
    V_out = step_overpressure!(state, Erupt, T_mush_K, ϕ_mush, ρ, V_e, Ṁ_in, Δt)

Integrate the QMagma (Degruyter & Huber 2014) chamber-overpressure ODE over one
thermal step and drain the chamber when it crosses the failure threshold. Port
of QMagma's `step_overpressure!` (`ThermalCode_1D.jl`), generalized from a 1D
column to MTK's native (2D per-unit-depth / 3D true) volume convention:

    (1/β_r + 1/β_m) dP/dt = Ṁ_in/(ρV_e) − (1/ρ)dρ/dt|_{T,ϕ} − (P−P_lith)/η_r

`ρ(T_K, ϕ, P)` computes the magma density [kg/m³] — build it with
[`magma_density_fn`](@ref); `1/β_m = (1/ρ)∂ρ/∂P` is a finite difference at
fixed `(T_mush_K, ϕ_mush)`, and the crystallization source `-(1/ρ)dρ/dt|_{T,ϕ}`
is a finite difference against the previous call's mush state at fixed `P`.
With a plain melt+crystal density law this source term reflects only the
melt/crystal density contrast: for the usual case `ρ_solid > ρ_melt`,
cooling-driven crystallization at fixed mass needs *less* chamber volume and so
*depressurizes*. Genuine second-boiling pressurization needs an exsolving gas
phase (QMagma's `ϕg < ϕ_g_crit` lock-up check is not modelled here) — supply a
`ThreePhase_Density` magma phase with `Erupt.Solub` set to get it via
`magma_density_fn`. `Ṁ_in` is a mass recharge rate [kg/s, 2D: per-unit-depth].
`state.P_lith` must be set by the caller (from the mush centroid depth) before
calling — this function does not compute it.

The step is sub-divided so `|ΔP|` moves ≲¼·`ΔP_crit` per sub-step (a single
Euler step can overshoot far past `ΔP_crit` when recharge is stiff relative to
`Δt`, over-counting the drained volume); every sub-step that crosses `ΔP_crit`
drains `V_e·(ΔP−ΔP_relax)·(1/β_r+1/β_m)` and resets `P = P_lith + ΔP_relax`.
Returns the volume drained over the whole step (`0.0` if the chamber only
charged), in the same convention `V_e` was given in. The first call (`!state.init`)
or any call with `V_e ≤ 0` only initializes `state` (`P = P_lith`, `T_prev`,
`ϕ_prev`) and returns `0.0`.
"""
function step_overpressure!(state::ChamberState, Erupt::EruptionParameters,
                            T_mush_K::Real, ϕ_mush::Real, ρ, V_e::Real, Ṁ_in::Real, Δt::Real)
    if !state.init || V_e <= 0
        state.P      = state.P_lith
        state.T_prev = T_mush_K
        state.ϕ_prev = ϕ_mush
        state.init   = true
        return 0.0
    end

    ρ0 = ρ(T_mush_K, ϕ_mush, state.P)

    # magma compressibility 1/β_m = (1/ρ)∂ρ/∂P (finite difference at fixed T,ϕ)
    dP     = max(1e3, 1e-4*max(state.P, 1e5))
    ρp     = ρ(T_mush_K, ϕ_mush, state.P + dP)
    inv_βm = (ρp - ρ0)/(ρ0*dP)
    state.inv_βm = inv_βm

    # thermodynamic source -(1/ρ)dρ/dt at fixed P (T,ϕ evolve on the thermal Δt)
    ρ_old   = ρ(state.T_prev, state.ϕ_prev, state.P)
    dρdt_TP = (ρ0 - ρ_old)/Δt
    S       = Ṁ_in/(ρ0*V_e) - dρdt_TP/ρ0

    inv_βr = 1.0/Erupt.β_r
    invβ   = inv_βr + inv_βm

    # sub-step count: keep |ΔP| change per sub-step ≲ ¼·ΔP_crit so the drain
    # doesn't overshoot. Clamp in float space BEFORE the Int cast: a soft
    # (floored) η_r can make |dPdt0| huge, and ceil(Int, >typemax(Int)) would
    # overflow before a post-hoc clamp could cap it.
    dPdt0 = (S - (state.P - state.P_lith)/Erupt.η_r) / invβ
    nsub   = round(Int, clamp(ceil(abs(dPdt0)*Δt / (0.25*Erupt.ΔP_crit)), 1.0, 10_000.0))
    dt_sub = Δt/nsub

    V_out = 0.0
    for _ in 1:nsub
        dPdt      = (S - (state.P - state.P_lith)/Erupt.η_r) / invβ
        state.P  += dPdt*dt_sub
        if (state.P - state.P_lith) >= Erupt.ΔP_crit
            V_out   += V_e*(state.P - state.P_lith - Erupt.ΔP_relax)*invβ
            state.P  = state.P_lith + Erupt.ΔP_relax
        end
    end

    state.T_prev = T_mush_K
    state.ϕ_prev = ϕ_mush
    return V_out
end

"""
    erupted = erupt_magma!(T, ϕ, dϕdT, Tracers, Grid, Erupt, time)

Erupt magma once the eruption trigger fires. Steps:
1. Compute the eruptible volume `Ve` (melt in cells with `ϕ > Erupt.ϕ_erupt`). In
   2D this is per-unit-depth; in 3D a true volume.
2. Decide whether an eruption fires, and the withdrawal fraction `η`:
   - `Erupt.overpressure = false` (default, kinematic trigger): `Ve` is lifted to
     a true 3D volume (km³, comparable to `V_crit`) when `Erupt.out_of_plane_3D`;
     if `Ve < Erupt.V_crit`, do nothing and return `false`. Otherwise
     `η = Erupt.erupt_efficiency` (fixed).
   - `Erupt.overpressure = true` (physical trigger): the mush-mean temperature
     and melt fraction are stepped through the chamber-overpressure ODE
     ([`step_overpressure!`](@ref), using `Erupt.chamber`); if it does not drain
     this step, do nothing and return `false`. Otherwise `η` is the emergent
     drained-volume fraction `V_out/Ve` (volume-conserving withdrawal), and the
     `Δt`, `ρ`, `Ṁ_in` keywords are required.
3. Remove `η` of the eruptible melt by *thermal extraction*: each eruptible cell
   is cooled by `ΔT = η·ϕ/(dϕ/dT)`, the linearized temperature drop that removes
   `η·ϕ` of melt (uses the locally-available `dϕ/dT`).
4. If `Erupt.deflate`, also deflate the chamber by volume-conserving column
   subsidence: the host rock in each column sinks by the melt void withdrawn
   beneath it (purely vertical), so the surface-subsidence integral equals the
   booked erupted volume `η·Ve`. This is what a moving free surface tracks.
5. Update the bookkeeping on `Erupt` (count, cumulative + per-event volume/time).

`T`, `ϕ`, `dϕdT` are the (CPU) temperature, melt-fraction and dϕ/dT arrays; they
are mutated in place. Returns `true` if an eruption occurred.

If a free-surface topography `z_surf` is supplied (keyword), the host-rock
subsidence of the deflation source is also applied to it, so the ground surface
tracks the chamber deflation (issue 4). Likewise, if a phase field `Phases` is
supplied (keyword), it is advected by the same subsidence so the material column
moves with the deflation. Both are only used on the deflation path
(`Erupt.deflate == true`).

`Δt` (the caller's timestep [s]), `ρ` (a `ρ(T_K, ϕ, P) -> kg/m³` density
callback) and `Ṁ_in` (the recharge mass rate [kg/s]) are only used when
`Erupt.overpressure == true`; `Ṁ_in` defaults to `0.0` (pure cooling, no
recharge).
"""
function erupt_magma!(T::Array, ϕ::Array, dϕdT::Array, Tracers, Grid, Erupt::EruptionParameters, time::Real;
                       z_surf=nothing, Phases=nothing, Δt::Real=NaN, ρ=nothing, Ṁ_in::Real=0.0)
    Erupt.erupt || return false
    Δ      = Grid.Δ
    coord  = Grid.coord1D
    dim    = length(coord)

    Ve, mask = eruptible_volume(ϕ, Δ, Erupt.ϕ_erupt; zc=coord[end], EruptAbove=Erupt.EruptAbove)

    Vcell   = prod(filter(>(0), collect(Δ)))

    # melt-weighted centroid + bulk volume of the eruptible region (used for the
    # deflation source and, in 2D, for the out-of-plane Gaussian volume), and the
    # melt-weighted mean temperature (used by the overpressure trigger). `sx2` is
    # the weighted Σx² needed for the horizontal standard deviation in 2D.
    sx = sy = sz = sT = 0.0; sx2 = 0.0; wsum = 0.0; Vbulk = 0.0
    if dim == 2
        @inbounds for j in axes(ϕ,2), i in axes(ϕ,1)
            if mask[i,j]
                x = coord[1][i]
                w = ϕ[i,j]; wsum += w; Vbulk += Vcell
                sx += w*x; sx2 += w*x^2; sz += w*coord[2][j]; sT += w*T[i,j]
            end
        end
    else
        @inbounds for k in axes(ϕ,3), j in axes(ϕ,2), i in axes(ϕ,1)
            if mask[i,j,k]
                w = ϕ[i,j,k]; wsum += w; Vbulk += Vcell
                sx += w*coord[1][i]; sy += w*coord[2][j]; sz += w*coord[3][k]; sT += w*T[i,j,k]
            end
        end
    end
    wsum == 0 && return false                   # no eruptible melt ⇒ nothing to do
    cx = sx/wsum

    if Erupt.overpressure
        # Physical ΔP_crit trigger: step the chamber ODE on the mush-mean state
        # of the eruptible region: MTK's native volume convention throughout
        # (no out-of-plane km³ lift — that lift exists only for the V_crit
        # comparison below, and Ve here must stay dimensionally consistent with
        # Ṁ_in, which is derived from the same convention as Dikes.InjectVol).
        isnan(Δt)      && error("erupt_magma!: Erupt.overpressure=true requires the `Δt` keyword")
        ρ === nothing  && error("erupt_magma!: Erupt.overpressure=true requires the `ρ` density-callback keyword")
        T_mush_K   = sT/wsum + 273.15
        ϕ_mush     = Ve/Vbulk
        z_centroid = sz/wsum
        Erupt.chamber.P_lith = Erupt.ρ_crust*9.81*abs(z_centroid)
        V_out = step_overpressure!(Erupt.chamber, Erupt, T_mush_K, ϕ_mush, ρ, Ve, Ṁ_in, Δt)
        V_out <= 0 && return false               # chamber charged but did not drain
        η       = clamp(V_out/Ve, 0.0, 1.0)
        V_erupt = V_out
    else
        # (1→3D) In 2D, lift the per-unit-depth eruptible volume to a true 3D
        # volume by assuming a Gaussian out-of-plane (y) melt profile: the
        # effective out-of-plane length is √(2π)·σ, with σ the melt-weighted
        # horizontal half-width of the eruptible region (floored at the in-plane
        # spacing so even a single cell gets a sensible thickness). This keeps
        # `Ve` in km³ — directly comparable to `V_crit` and the (3D) injected
        # volume. No-op in 3D, where `Ve` is already a volume.
        if dim == 2 && Erupt.out_of_plane_3D
            σx  = sqrt(max(sx2/wsum - cx^2, 0.0))
            Ve *= sqrt(2π) * max(σx, Δ[1])
        end

        Ve < Erupt.V_crit && return false        # trigger on the (3D) eruptible volume
        η       = Erupt.erupt_efficiency
        V_erupt = η * Ve
    end

    # Freeze the tracers caught up in the eruption — a random fraction `η` of the
    # melt-rich (eruptible-cell) tracers — *before* any deflation advection below,
    # so they keep their eruption-instant position and Tt-path. Their preserved
    # history is the erupted "zircon cargo". The final history point is recorded
    # in Myr to match the tracer time convention used by the drivers (see
    # `update_Tvec!`).
    freeze_erupted_tracers!(Tracers, Grid, mask, η, time/SecYear*1e-6)

    # Deflation field, captured from the *pre-withdrawal* melt so the host rock
    # sinks by exactly the void the withdrawal opens below it. Model selected by
    # Erupt.deflation_model: :column (exact per-column, no lateral coupling) or
    # :local_mogi (smoothed, volume-exact, cutoff-bounded Mogi superposition).
    Vel = if !Erupt.deflate
        nothing
    elseif Erupt.deflation_model === :column
        _column_subsidence_velocity(ϕ, mask, Δ[end], η, dim)
    elseif Erupt.deflation_model === :local_mogi
        _local_mogi_deflation_velocity(ϕ, mask, coord, Δ, Vcell, η, dim;
                                       cutoff_factor=Erupt.deflation_cutoff_factor,
                                       max_window_cells=Erupt.deflation_max_window_cells)
    else
        error("erupt_magma!: unknown Erupt.deflation_model = $(Erupt.deflation_model) (expected :column or :local_mogi)")
    end

    # (3) withdraw a fraction η of the mobile melt from each eruptible cell:
    #     ϕ → (1-η)·ϕ, realized thermally by cooling the cell by ΔT = η·ϕ/(dϕ/dT)
    #     (the local linearized drop that removes η·ϕ of melt). Booked volume
    #     (below) = Σ η·ϕ·Vcell = η·Ve, exactly the melt removed here (η is fixed
    #     `erupt_efficiency` for the kinematic trigger, or the emergent
    #     drained-volume fraction for the physical trigger — set above).
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

    # (4) optional volume-conserving chamber deflation: advect the host rock,
    #     free surface, and phase field by the column-subsidence field. The
    #     surface-subsidence integral equals the booked erupted volume η·Ve.
    if Erupt.deflate
        _advect_hostrock!(T, Tracers, coord, _grid_full(coord), Vel)
        if !isnothing(z_surf)
            advect_surface!(z_surf, Vel[end], Grid)
        end
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

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

        pt = InjectSills.new_point_inside_sill(sill)   # Point{N, Float64}

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
    H   = sill.H.val           # maximum opening thickness [m]

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
    # Number of pseudo-time steps (keeps displacement < 0.5 * min_dx)
    # ------------------------------------------------------------------
    Spacing = [Grid[i][2] - Grid[i][1] for i in 1:dim]
    d       = minimum(Spacing) * 0.5
    nsteps  = max(ceil(Int, H / d), 2)
    dt      = 1.0 / nsteps

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
    # Injected volume (numeric SI value from the GeoUnit)
    # ------------------------------------------------------------------
    W_val = sill.W.val   # radius (InjectSills stores W as the half-width/radius)
    H_val = sill.H.val
    InjectedVolume = 4/3 * π * W_val^2 * (H_val/2)   # oblate spheroid volume [m³]

    # ------------------------------------------------------------------
    # Optionally advect a plotting polygon
    # ------------------------------------------------------------------
    if !isempty(dike_poly)
        advect_dike_polygon!(dike_poly, Grid, Velocity)
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

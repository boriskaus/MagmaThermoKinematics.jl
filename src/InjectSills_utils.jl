# InjectSills_utils.jl
#
# Sill emplacement on top of the InjectSills.jl API:
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

The sill's center and orientation are encoded in the `sill` object itself;
no external rotation is required here.
"""
function add_dike(Tfield, Tr, Grid, sill::InjectSills.AbstractSill, T_in::Real, Phase_in::Integer, nTr_dike::Integer)

    dim = length(Grid)

    # ------------------------------------------------------------------
    # 1.  Set temperature inside the sill
    # ------------------------------------------------------------------
    _launch!(_fill_sill!, Tfield, size(Tfield), Tfield, Tuple(Grid), sill, T_in)   # ponytail: sills with array fields (FiniteEllipsoidalCavity) need Adapt for a GPU

    # ------------------------------------------------------------------
    # 2.  Seed new tracers inside the sill
    # ------------------------------------------------------------------
    for _ in 1:nTr_dike

        pt = InjectSills.new_point_inside_sill(sill)   # Point{N, Float64}

        number = isassigned(Tr, 1) ? Tr.num[end] + 1 : 1

        FT = isassigned(Tr, 1) ? eltype(Tr[1].time_vec) : Float32
        coord      = [Float64(pt[i]) for i in 1:dim]  # Vector{Float64}
        new_tracer = Tracer{FT}(num=number, coord=coord, T=Float64(T_in), Phase=Int64(Phase_in))

        if !isassigned(Tr, 1)
            Tr = StructArray([new_tracer])
        else
            push!(Tr, new_tracer)
        end
    end

    return Tfield, Tr
end

@kernel function _fill_sill!(T, Grid, sill::InjectSills.AbstractSill{N}, T_in) where {N}
    I = @index(Global, Cartesian)
    if InjectSills.inside(InjectSills.Point{N,Float64}(map(getindex, Grid, Tuple(I))), sill)
        T[I] = T_in
    end
end

"Move the points `(P[1][i], …, P[N][i])` to `x + u(x)`, with `u` the displacement of `sill`, clamped to `Grid`."
function displace_points!(P, sill::InjectSills.AbstractSill{N}, Grid) where {N}
    for i in eachindex(P[1])
        x = InjectSills.Point{N,Float64}(ntuple(d -> P[d][i], Val(N)))
        u = InjectSills.hostrock_displacement(sill, x)
        for d in 1:N
            P[d][i] = clamp(x[d] + u[d], first(Grid[d]), last(Grid[d]))
        end
    end
    return P
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
- `T`                   – temperature array [°C], mutated in-place; any backend
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
`(Tracers, Tnew, InjectedVolume, dike_poly, Velocity)`, where `InjectedVolume` is
the equivalent 3D volume of `sill` in m³ (`InjectSills.volume`).

## Algorithm
The temperature field is advected by the displacement field of
`InjectSills.hostrock_displacement!` over `nsteps` pseudo-time steps, so that
the displacement per step stays below `0.5 * min(dx, dz)`. With the default
RK2/linear scheme this runs on the backend of `T`; the other schemes need a
CPU `Array`. Existing tracers and `dike_poly` move from `x` to `x + u(x)`,
with `u` evaluated at their own positions. `add_dike` then sets `T = T_in`
inside the sill and seeds the new tracers.

The displacement field (= velocity for pseudo-time `dt_total = 1`) is
obtained directly from the sill object, which already encodes the center and
orientation of the intrusion — no external rotation is needed.
"""
function inject_sills(Tracers, T::AbstractArray, Grid,
                      sill::InjectSills.AbstractSill,
                      T_in::Real, Phase_in::Integer, nTr_dike::Integer;
                      AdvectionMethod="RK2", InterpolationMethod="Linear",
                      dike_poly=[])

    dim = length(Grid)
    H   = sill.H.val           # maximum opening thickness [m]

    # ------------------------------------------------------------------
    # Number of pseudo-time steps (keeps displacement < 0.5 * min_dx)
    # ------------------------------------------------------------------
    Spacing = [Grid[i][2] - Grid[i][1] for i in 1:dim]
    d       = minimum(Spacing) * 0.5
    nsteps  = max(ceil(Int, H / d), 2)
    dt      = 1.0 / nsteps

    # ------------------------------------------------------------------
    # Displacement field (= velocity for pseudo-time dt_total = 1.0) on
    # the backend of T; hostrock_displacement! handles centering + rotation.
    # ------------------------------------------------------------------
    backend  = get_backend(T)
    GridFull = ntuple(dim) do k
        X = KernelAbstractions.allocate(backend, Float64, size(T))
        X .= reshape(Grid[k], ntuple(j -> j == k ? length(Grid[k]) : 1, dim))
    end
    Velocity = InjectSills.hostrock_displacement!(map(similar, GridFull), sill, GridFull)

    # ------------------------------------------------------------------
    # Pseudo-timestep advection of T: open the sill gradually
    # ------------------------------------------------------------------
    if AdvectionMethod == "RK2" && InterpolationMethod == "Linear"
        buf = similar(T)
        src, dst = T, buf
        for _ in 1:nsteps
            AdvectTemperature!(dst, src, Grid, Velocity, dt)
            src, dst = dst, src
        end
        src === T || copyto!(T, src)
    else
        T isa Array || throw(ArgumentError("inject_sills: AdvectionMethod=\"$AdvectionMethod\", InterpolationMethod=\"$InterpolationMethod\" needs a CPU Array; use RK2/Linear for $(typeof(T))"))
        for _ in 1:nsteps
            T .= AdvectTemperature(T, Grid, GridFull, Velocity, dt, AdvectionMethod, InterpolationMethod)
        end
    end

    # ------------------------------------------------------------------
    # Move existing tracers and the plotting polygon with the host rock
    # ------------------------------------------------------------------
    if isassigned(Tracers, 1)
        coord = Tracers.coord
        P = ntuple(k -> getindex.(coord, k), dim)
        displace_points!(P, sill, Grid)
        for (c, i) in zip(coord, eachindex(coord))
            c .= getindex.(P, i)
        end
    end
    isempty(dike_poly) || displace_points!(dike_poly, sill, Grid)

    # ------------------------------------------------------------------
    # Set T = T_in inside the sill and seed new tracers
    # ------------------------------------------------------------------
    Tnew, Tracers = add_dike(T, Tracers, Grid, sill, T_in, Phase_in, nTr_dike)

    # ------------------------------------------------------------------
    # Injected volume [m³]: equivalent 3D volume of the sill type
    # ------------------------------------------------------------------
    InjectedVolume = ustrip(uconvert(m^3, InjectSills.volume(sill)))

    return Tracers, Tnew, InjectedVolume, dike_poly, Velocity
end

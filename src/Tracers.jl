# This includes routines that deal with tracers

"""
    Structure that has information about the tracers

    General form:
        Tracer(num, coord, T)

        with:

            num:   number of the tracer (integer)

            coord: coordinates of the tracer
                    2D - [x; z]
                    3D - [x; y; z]

            T:          Temperature of the tracer [Celsius]
            Phase:      Phase of tracer
            Phi:        Melt fraction of tracer
            time_vec:   Vector with time
            T_vec :     Vector with temperature values
"""
mutable struct Tracer{FT <: AbstractFloat}
    num::Int64
    coord::Vector{Float64}
    T::Float64
    Phase::Int64
    Phi::Float64
    time_vec::Vector{FT}
    T_vec::Vector{FT}
end

# Default constructor: Tracer(coord=[...]) or Tracer{Float32}(coord=[...])
function Tracer{FT}(;
        num::Int64 = 0, coord::Vector{Float64}, T::Float64 = 900.0,
        Phase::Int64 = 1, Phi::Float64 = 0.0,
        time_vec::Vector{FT} = FT[], T_vec::Vector{FT} = FT[]
    ) where {FT <: AbstractFloat}
    return Tracer{FT}(num, coord, T, Phase, Phi, time_vec, T_vec)
end
# Convenience constructor without type parameter: defaults to Float32
Tracer(; kwargs...) = Tracer{Float32}(; kwargs...)
#    Chemistry   ::  Vector{Float64} = []    # Could @ some stage hold the evolving chemistry of the magma


"""
    UpdateTracers_T_ϕ!(Tracers, Grid::Tuple, T, Phi);

In-place function that interpolates `T` & `Phi`, defined on the `Grid`, to `Tracers`.

- Tracers:  StructArray that contains tracers, where we want to update
- Grid:     Regular grid on which the parameters to be interpolated are defined
                2D - (X,Z)
                3D - (X,Y,Z)
- T:  `T` field that is defined on the grid, to be interpolated to tracers
- Phi:  `Phi` field that is defined on the grid, to be interpolated to tracers

Note that we employ linear interpolation using custom functions
"""
function UpdateTracers_T_ϕ!(Tracers, Grid::Tuple, T::AbstractArray{_T, dim}, Phi::AbstractArray{_T, dim}) where {_T, dim}
    isassigned(Tracers, 1) || return nothing     # only if the Tracers StructArray is non-empty
    interpolate_to_tracers!((Tracers.T, Tracers.Phi), (T, Phi), Tracers.coord, Grid)
    return nothing
end

"""
    UpdateTracers_Field!(Tracers::StructVector{TRACERS}, Grid::GridData{_T,dim}, Field::AbstractArray{_T,dim}, FieldName::Symbol);

In-place, non-allocating, function that interpolates `Field`, defined on the `Grid`, to the field `FieldName` on `Tracers`.

- `Tracers`:    StructVector that contains tracers, where we want to update the properties. Each tracer should at least contain the fields `coord` (coordinates) and `FieldName`.
- `Grid``:      Grid structure that describes the coordinates
- `Field`:      The 2D or 3D field
- `FieldName``: Symbol of the name of the field on each of the Tracers

Note that we employ linear interpolation using custom functions; the grid may be refined.
"""
function UpdateTracers_Field!(Tracers::StructVector{TRACERS}, Grid::GridData{_T, dim}, Field::AbstractArray{_T, dim}, FieldName::Symbol) where {TRACERS, _T, dim}
    isassigned(Tracers, 1) || return nothing     # only if the Tracers StructArray is non-empty
    interpolate_to_tracers!((getproperty(Tracers, FieldName),), (Field,), Tracers.coord, Grid.coord1D)
    return nothing
end

# For every tracer, clamp its coordinates (in place) to the grid box and set `vals[k][iT]` to the multilinear
# interpolation of `fields[k]` at that point. `coords` are the 1D coordinate vectors of the grid (ranges or vectors).
function interpolate_to_tracers!(vals::Tuple, fields::Tuple, coord, coords::Tuple)
    for iT in eachindex(coord, vals...)
        pt = coord[iT]
        for d in eachindex(coords)
            pt[d] = clamp(pt[d], first(coords[d]), last(coords[d]))
        end
        p = ntuple(d -> _fractional_index(coords[d], pt[d]), length(coords))
        map((v, F) -> v[iT] = _lerp(F, p), vals, fields)
    end
    return nothing
end

"""
    PhaseRatioFromTracers!(PhaseRatio::AbstractArray, Grid::GridData, Tracers; InterpolationMethod="Constant", BackgroundPhase=nothing, ReturnNumTracers=false)

This computes the PhaseRatio from the `Tracers` on the gridpoints described by `Grid`. The `PhaseRatio` is a matrix that has one dimension more than the size of the grid
and, after calling this function, at every point we will have the fraction of that phase that is present in the grid.


optional Parameters:

- InterpolationMethod:    Interpolation method used to go from Tracers ->  Grid
    "Constant"          -   All particles within a distance [dx,dy,dz] around the grid point
    "DistanceWeighted"  -   Particles closer to the grid point have a stronger weight.
                                            This follows what is described in:
                                                Duretz, T., May, D.A., Gerya, T.V., Tackley, P.J., 2011. Discretization errors and
                                                free surface stabilization in the finite difference and marker-in-cell method for applied geodynamics:
                                                A numerical study: Geochem. Geophys. Geosyst. 12, https://doi.org/10.1029/2011GC00356

- BackgroundPhase:       The background phase (used for places that don't have cells, nor surrounding cells )
- ReturnNumTracers:      Return the number of tracers on every grid cell (default=false)
"""
function PhaseRatioFromTracers!(PhaseRatio::AbstractArray, Grid::GridData{_T, dim}, Tracers; InterpolationMethod = "Constant", ReturnNumTracers = false, BackgroundPhase = nothing) where {_T, dim}

    numPhases = maximum(Tracers.Phase)
    if !isnothing(BackgroundPhase)
        #   if numPhases<BackgroundPhase; numPhases=BackgroundPhase; end
    end

    if size(PhaseRatio)[1:dim] != (Grid.N...,)
        error("Size of PhaseRatio array inconsistent with input grid")
    end
    if size(PhaseRatio)[dim + 1] < numPhases
        error("Size of last dimension of PhaseRatio is too small")
    end
    x = Grid.coord1D[1]
    if dim == 2
        y = Grid.coord1D[2]
    end
    if dim == 3
        z = Grid.coord1D[3]
    end

    # Initialize Phase Ratio
    PhaseRatio .= 0.0
    if !isnothing(BackgroundPhase)
        if dim == 1
            PhaseRatio[:, BackgroundPhase] .= 1.0
        elseif dim == 2
            PhaseRatio[:, :, BackgroundPhase] .= 1.0
        elseif dim == 3
            PhaseRatio[:, :, :, BackgroundPhase] .= 1.0
        end
    end

    NumTracers = zeros(Int64, Grid.N...)    # Tracks # of tracers around every point
    idx = zeros(Int64, dim)          # pre-allocate index
    pt_near = zeros(_T, dim)             # coordinate of nearest point
    dist = zeros(_T, dim)             # normalize distance point -> nearest grid point

    if isassigned(Tracers, 1)                # only if the Tracers StructArray is non-empty

        for iT in 1:length(Tracers)
            Trac = Tracers[iT]
            pt = Trac.coord
            phase = Trac.Phase

            # correct point for bounds:
            for i in 1:dim
                if pt[i] < Grid.min[i]
                    pt[i] = Grid.min[i]
                end
                if pt[i] > Grid.max[i]
                    pt[i] = Grid.max[i]
                end
            end

            # find Cartesian index of nearest point on grid
            if Grid.ConstantΔ
                idx .= round.(Int64, (pt .- Grid.min) ./ Grid.Δ) .+ 1
            else                                    # refined grid: the fractional index is linear in x within a cell, so the nearest point is the rounded one
                for i in 1:dim
                    p = _fractional_index(Grid.coord1D[i], pt[i])
                    idx[i] = round(Int64, p)
                    dist[i] = abs(p - idx[i])       # distance to the nearest point, relative to the spacing on the side of the tracer
                end
            end
            if dim == 1
                I = CartesianIndex(idx[1])
                Iphase = CartesianIndex(idx[1], phase)
            elseif dim == 2
                I = CartesianIndex(idx[1], idx[2])
                Iphase = CartesianIndex(idx[1], idx[2], phase)
            elseif dim == 3
                I = CartesianIndex(idx[1], idx[2], idx[3])
                Iphase = CartesianIndex(idx[1], idx[2], idx[3], phase)
            end
            if InterpolationMethod == "DistanceWeighted"
                if Grid.ConstantΔ
                    pt_near = Tuple(I) .* Grid.Δ .- Grid.Δ .+ Grid.min      # coordinates of nearest point
                    dist .= abs.((pt .- pt_near) ./ Grid.Δ)      # distance of tracers to regular grid point (normalized over Δ)
                end
                Weight = prod(1 .- 2 .* dist)               # weight of point: 1 at the node, 0 at the cell face
            elseif InterpolationMethod == "Constant"
                Weight = 1.0
            end


            NumTracers[I] += 1            # Keep track of number of phases
            PhaseRatio[Iphase] += Weight       # Weight @ every point

        end
    end

    # If we have a BG phase set, remove what we set @ the beginning
    if !isnothing(BackgroundPhase)
        for I in CartesianIndices(NumTracers)
            if NumTracers[I] > 0
                Iph = CartesianIndex((Tuple(I)..., BackgroundPhase))
                PhaseRatio[Iph] = PhaseRatio[Iph] - 1.0         # subtract the value we added @ the beginning
            end
        end
    end

    # normalize
    PhaseRatioSum = sum(PhaseRatio, dims = dim + 1)
    for I in CartesianIndices(PhaseRatio)
        Isum = (Tuple(I)[1:dim]..., 1)
        PhaseRatio[I] = PhaseRatio[I] / PhaseRatioSum[Isum...]
    end

    if !isnothing(ReturnNumTracers)
        return NumTracers
    else
        return nothing
    end
end


"""
    PhaseFromTracers!(Phases::AbstractArray, Grid::GridData, Tracers; InterpolationMethod="Constant", BackgroundPhase=nothing)

This computes the `Phases` from the `Tracers` on the gridpoints described by `Grid`. The `Phases` is a matrix with integers that indicates the dominant phase at that point


optional Parameters:

- InterpolationMethod:    Interpolation method used to go from Tracers ->  Grid
    "Constant"          -   All particles within a distance [dx,dy,dz] around the grid point
    "DistanceWeighted"  -   Particles closer to the grid point have a stronger weight.
                                            This follows what is described in:
                                                Duretz, T., May, D.A., Gerya, T.V., Tackley, P.J., 2011. Discretization errors and
                                                free surface stabilization in the finite difference and marker-in-cell method for applied geodynamics:
                                                A numerical study: Geochem. Geophys. Geosyst. 12, https://doi.org/10.1029/2011GC00356

- BackgroundPhase:       The background phase (used for places that don't have cells, nor surrounding cells )
- ReturnNumTracers:      Return the number of tracers on every grid cell (default=false)
"""
function PhasesFromTracers!(Phases::AbstractArray, Grid::GridData{_T, dim}, Tracers; InterpolationMethod = "Constant", BackgroundPhase = nothing, ReturnNumTracers = nothing) where {_T, dim}

    maxPhase = maximum(Tracers.Phase)
    PhaseRatio = zeros((Grid.N..., maxPhase)...)

    # Compute tracers
    NumTracers = PhaseRatioFromTracers!(PhaseRatio, Grid, Tracers, InterpolationMethod = InterpolationMethod, BackgroundPhase = BackgroundPhase, ReturnNumTracers = ReturnNumTracers)

    for I in CartesianIndices(Phases)
        id = Tuple(I)
        maxPhase = argmax(@view PhaseRatio[id..., :])
        Phases[I] = maxPhase
    end

    return NumTracers
end


"""
        AdvectTracers!(Tracers, Grid, Velocity, dt, Method="RK2");

        Advects [Tracers] for one timestep (dt) using the [Velocity] defined on the points [Grid].

        Method: can be "Euler","RK2" or "RK4", for 1th, 2nd or 4th order explicit advection scheme, respectively.
"""
function AdvectTracers!(Tracers, Grid, Velocity, dt, Method = "RK2")
    # Advect tracers forward in time & interpolate T on them

    dim = length(Grid)
    coord = reduce(hcat, Tracers.coord)'     # extract array with coordinates of tracers

    x = coord[:, 1]
    z = coord[:, end]
    if dim == 2
        Points_irregular = (x, z)
    else
        y = coord[:, 2]
        Points_irregular = (x, y, z)
    end

    # Correct coordinates (to stay withoin bounds of models)
    CorrectBounds!(Points_irregular, Grid)

    # Advect
    Points_new = AdvectPoints(Points_irregular, Grid, Velocity, dt, Method, "Linear")      # Advect tracers

    # function to assign properties
    function testnoalloc_2D(sarr, val)
        for (Tracer, x, z) in zip(LazyRows(sarr), val[1], val[2])
            Tracer.coord = [x;z]
        end
        return
    end

    function testnoalloc_3D(sarr, val)
        for (Tracer, x, y, z) in zip(LazyRows(sarr), val[1], val[2], val[3])
            Tracer.coord = [x; y; z]
        end
        return
    end

    return if dim == 2
        testnoalloc_2D(Tracers, Points_new)
    else
        testnoalloc_3D(Tracers, Points_new)
    end

end

"""
    update_Tvec!(Tracers::StructArray, time)

Updates temperature & time vector on every tracer.

Tracers that are first sampled in the same call share one `time_vec` object (it is pushed
once per call), so the time history costs one vector per sampling call instead of one per
tracer. Treat `time_vec` as read-only.
"""
function update_Tvec!(Tracers::StructArray, time_val::Float64)
    isassigned(Tracers, 1) || return Tracers
    time_vecs, T_vecs, T = Tracers.time_vec, Tracers.T_vec, Tracers.T
    shared = nothing    # time_vec handed to every tracer sampled for the first time in this call
    for iT in eachindex(time_vecs, T_vecs, T)
        tv, Tv = time_vecs[iT], T_vecs[iT]
        if isempty(tv) && isempty(Tv)
            shared === nothing && (shared = tv)
            tv = time_vecs[iT] = shared
        end
        push!(Tv, T[iT])
        length(tv) < length(Tv) && push!(tv, time_val)
        length(tv) == length(Tv) || error("Tracer $iT: time_vec (length $(length(tv))) and T_vec (length $(length(Tv))) are out of sync")
    end
    return Tracers
end

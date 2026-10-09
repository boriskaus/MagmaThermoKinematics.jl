module Grid
# This creates a computational grid
import Base: show
import GeophysicalModelGenerator: CartData

export GridData, CreateGrid

# `R` and `RC` are the concrete range types of the coordinate fields.
# `StepRangeLen{FT}` alone leaves three of the range's four parameters free, so
# every read of `coord1D` would return an abstractly typed range and every
# operation on it would dispatch dynamically. Construct through `CreateGrid`,
# which fills them in.
struct GridData{FT, D, R <: StepRangeLen{FT}, RC <: StepRangeLen{FT}}
    ConstantΔ::Bool                         # Constant spacing (true in all cases for now)
    N::NTuple{D, Int}                # Number of grid points in every direction
    Δ::NTuple{D, FT}                 # (constant) spacing in every direction
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

Spacing is assumed to be constant

Note: since this is mostly for Solid Earth geoscience applications, in 2D the second dimension is called z (vertical)

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
        extent = nothing
    )

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

    # Generate 1D coordinate arrays of vertexes in all directions
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


"""
    Grid = CreateGrid(d::CartData; m_to_km::Bool=true)
Creates a (regularly spaced) grid that can be used within MTK, from a 2D or 3D `CartData` object (generated within the GeophysicalModelGenerator)
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
        " grid spacing Δ: $(g.Δ) \n"
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

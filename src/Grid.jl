module Grid
# This creates a computational grid
import Base: show 

export GridData, CreateGrid, GridArray, GridArray!

struct GridData{FT, D}
    ConstantΔ   :: Bool                         # Constant spacing (true in all cases for now)
    N           :: NTuple{D,Int}                # Number of grid points in every direction
    Δ           :: NTuple{D,FT}                 # (constant) spacing in every direction
    L           :: NTuple{D,FT}                 # Domain size
    min         :: NTuple{D,FT}                 # start of the grid in every direction 
    max         :: NTuple{D,FT}                 # end of the grid in every direction 
    coord1D     :: NTuple{D,StepRangeLen{FT}}   # Tuple with 1D vectors in all directions
    coord1D_cen :: NTuple{D,StepRangeLen{FT}}   # Tuple with 1D vectors of center points in all directions
end   


"""

Creates a 1D, 2D or 3D grid of given size. Grid can be created by defining the size and either the `extent` (length) of the grid in all directions, or by defining start & end points 

Spacing is assumed to be constant

Note: since this is mostly for Solid Earth geoscience applications, the second dimension is called z (vertical)

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
    size=(),
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
    dim =   length(N)   
    
    # Specify domain by length in every direction
    if !isnothing(extent)
        x,y,z = nothing, nothing, nothing
        x = (0., extent[1])
        if dim>1
            z =  (-extent[2], 0.0)       # vertical direction (negative)
        end
        if dim>2
            y = (0., extent[3])
        end
    end

    FT = typeof(x[1])
    if      dim==1
        L = (x[2] - x[1],)
        X₁= (x[1], )
    elseif  dim==2
        L = (x[2] - x[1], z[2] - z[1])
        X₁= (x[1], z[1])
    else
        L = (x[2] - x[1], z[2] - z[1], y[2] - y[1])
        X₁= (x[1], z[1], y[1])
    end
    Xₙ  = X₁ .+ L  
    Δ   = L ./ (N .- 1)       

    # Generate 1D coordinate arrays of vertexes in all directions
    coord1D=()
    for idim=1:dim
        coord1D  = (coord1D...,   range(X₁[idim], Xₙ[idim]; length = N[idim]  ))
    end
    
    # Generate 1D coordinate arrays centers in all directionbs
    coord1D_cen=()
    for idim=1:dim
        coord1D_cen  = (coord1D_cen...,   range(X₁[idim]+Δ[idim]/2, Xₙ[idim]-Δ[idim]/2; length = N[idim]-1  ))
    end
    
    ConstantΔ   = true;
    return GridData(ConstantΔ,N,Δ,L,X₁,Xₙ,coord1D, coord1D_cen)

end

"""
    X,Z = GridArray(x::StepRangeLen,z::StepRangeLen)

Creates 2D coordinate arrays from 1D vectors with coordinates in 1th & 2nd dimension.
Usually employed in combination with `GridData` 

"""
function GridArray(x::StepRangeLen,z::StepRangeLen)
    Nx, Nz = length(x), length(z)
    
    X,Z = zeros(Nx,Nz), zeros(Nx,Nz);
    for i in CartesianIndices(X)
        X[i] = x[i[1]]
        Z[i] = z[i[2]]
    end
    
    return X,Z
end

"""
    X,Y,Z = GridArray(x::StepRangeLen,y::StepRangeLen,z::StepRangeLen)

Creates 3D coordinate arrays from 1D vectors with coordinates in 1th & 2nd dimension.
Usually employed in combination with `GridData` 

"""
function GridArray(x::StepRangeLen, y::StepRangeLen, z::StepRangeLen)
    Nx, Ny, Nz = length(x), length(y), length(z)
    
    X, Y, Z = zeros(Nx,Ny,Nz),zeros(Nx,Ny,Nz),zeros(Nx,Ny,Nz);
    for i in CartesianIndices(X)
        X[i] = x[i[1]]
        Y[i] = y[i[2]]
        Z[i] = z[i[3]]
    end
    
    return X,Y,Z
end


"""
    GridArray!(X::AbstractArray,Z::AbstractArray,x::StepRangeLen,z::StepRangeLen)

In place setting of 3D coordinate arrays from 1D vectors.
Usually employed in combination with `GridData`. 

"""
function GridArray!( X::AbstractArray,Z::AbstractArray,
                    x::StepRangeLen,z::StepRangeLen)
    for i in CartesianIndices(X)
        X[i] = x[i[1]]
        Z[i] = z[i[2]]
    end
    
    return nothing
end

"""
    GridArray!(X::AbstractArray,Y::AbstractArray, Z::AbstractArray,x::StepRangeLen,z::StepRangeLen)

In place setting of 3D coordinate arrays from 1D vectors.
Usually employed in combination with `GridData`. 

"""
function GridArray!( X::AbstractArray,Y::AbstractArray,Z::AbstractArray,
                    x::StepRangeLen,z::StepRangeLen)

    for i in CartesianIndices(X)
        X[i] = x[i[1]]
        Y[i] = y[i[2]]
        Z[i] = z[i[3]]
    end
    
    return nothing
end

# view grid object
function show(io::IO, g::GridData{FT, DIM}) where {FT, DIM}
  
    print(io, "Grid{$FT, $DIM} \n",
              "           size: $(g.N) \n",
              "         length: $(g.L) \n",
              "         domain: $(domain_string(g)) \n",
              " grid spacing Δ: $(g.Δ) \n")

end

# nice printing of grid
function domain_string(grid::GridData{FT, DIM}) where {FT, DIM}
    
    xₗ, xᵣ = grid.coord1D[1][1], grid.coord1D[1][end]
    if DIM>1
        yₗ, yᵣ = grid.coord1D[2][1], grid.coord1D[2][end]
    end
    if DIM>2
        zₗ, zᵣ = grid.coord1D[3][1], grid.coord1D[3][end]
    end
    if DIM==1
        return "x ∈ [$xₗ, $xᵣ]"
    elseif DIM==2
        return "x ∈ [$xₗ, $xᵣ], z ∈ [$yₗ, $yᵣ]"
    elseif DIM==3
        return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ], z ∈ [$zₗ, $zᵣ]"
    end
end


end
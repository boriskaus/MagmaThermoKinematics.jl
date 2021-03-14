"""
    RegularRectilinearGrid{FT, TX, TY, TZ, R} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

A rectilinear grid with with constant grid spacings `Δx`, `Δy`, and `Δz` between cell centers
and cell faces, elements of type `FT`, topology `{TX, TY, TZ}`, and coordinate ranges
of type `R`.
"""
# NOTE: we can later add info here about the local & global extent of the grid, the halo, the neigboring processors and the type of calculation (CPU/MPI-CPU/GPU)
struct RegularRectilinearGrid{FT, TX, TY, TZ, R} <: AbstractRectilinearGrid{FT, TX, TY, TZ}
    # Number of grid points in (x,y,z).
    Nx :: Int
    Ny :: Int
    Nz :: Int
    # Domain size [m].
    Lx :: FT
    Ly :: FT
    Lz :: FT
    # Grid spacing [m].
    Δx :: FT
    Δy :: FT
    Δz :: FT
    # Range of coordinates at the centers of the cells.
    xC :: R
    yC :: R
    zC :: R
    # Range of grid coordinates at the faces of the cells.
    xF :: R
    yF :: R
    zF :: R
end

"""
    RegularRectilinearGrid([FT=Float64]; size,
                         extent = nothing, x = nothing, y = nothing, z = nothing)

Creates a `RegularRectilinearGrid` with `size = (Nx, Ny, Nz)` grid cells. 
Note that temperature is defined at the center of these grid cells, whereas velocities
are defined at the faces

Keyword arguments
=================

- `size` (required): A tuple prescribing the number of grid points in non-`Flat` directions.
                     `size` is a 3-tuple for 3D models, and a 2-tuple for 2D models.

- `extent`: A tuple prescribing the physical extent of the grid in non-`Flat` directions.
            The origin for three-dimensional domains is the lower left corner `(0, 0, -Lz)`.

- `x`, `y`, and `z`: Each of `x, y, z` are 2-tuples that specify the end points of the domain
                     in their respect directions. Scalar values may be used in `Flat` directions.

*Note*: _Either_ `extent`, or all of `x`, `y`, and `z` must be specified.

The physical extent of the domain can be specified via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(0, 10)` or via
the `extent` argument, e.g. `extent=(Lx, Ly, Lz)` which specifies the extent of each dimension
in which case 0 ≤ x ≤ Lx, 0 ≤ y ≤ Ly, and -Lz ≤ z ≤ 0.

Constants are stored using floating point values of type `FT`. By default this is `Float64`.
Make sure to specify the desired `FT` if not using `Float64`.

Grid properties
===============

- `(Nx, Ny, Nz)::Int`: Number of physical points in the (x, y, z)-direction

- `(Lx, Ly, Lz)::FT`: Physical extent of the grid in the (x, y, z)-direction

- `(Δx, Δy, Δz)::FT`: Center width in the (x, y, z)-direction

- `(xC, yC, zC)`: (x, y, z) coordinates of cell centers.

- `(xF, yF, zF)`: (x, y, z) coordinates of cell faces.

Examples
========

* A default grid with Float64 type:

```jldoctest
julia> using MagmaThermoKinematics

julia> grid = RegularRectilinearGrid(size=(32, 32, 32), extent=(1, 2, 3))
RegularRectilinearGrid{Float64, Bounded, Bounded, Bounded}
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 2.0], z ∈ [-3.0, 0.0]
                 topology: (Bounded, Bounded, Bounded)
  resolution (Nx, Ny, Nz): (32, 32, 32)
grid spacing (Δx, Δy, Δz): (0.03125, 0.0625, 0.09375)
```


"""
function RegularRectilinearGrid(FT=Float64;
                                  size,
                                     x = nothing, y = nothing, z = nothing,
                                extent = nothing
                              )

    TX, TY, TZ  =   (Bounded, Bounded, Bounded)         # default for now; we may want to extend that later

    # Add y dimension 
    if length(size)==2  
        size    =   (size[1], 1, size[2])
        TY      =   Flat2D
    end          

    Lx, Ly, Lz, x, y, z = validate_regular_grid_domain(TX, TY, TZ, FT, extent, x, y, z)

    # Unpacking
    Nx, Ny, Nz = N = size
                 L = (Lx, Ly, Lz)
    Δx, Δy, Δz = Δ = L ./ N
                X₁ = (x[1], y[1], z[1])

    # Face-node limits in x, y, z
    xF₋, yF₋, zF₋ = XF₋ = @. X₁ 
    xF₊, yF₊, zF₊ = XF₊ = @. XF₋ + L              

    # Center-node limits in x, y, z
    xC₋, yC₋, zC₋ = XC₋ = @. XF₋ + Δ / 2
    xC₊, yC₊, zC₊ = XC₊ = @. XC₋ + L - Δ

    # Generate coordinate arrays
    xF = range(xF₋, FT(xF₊); length = Nx+1)
    yF = range(yF₋, yF₊; length = Ny+1)
    zF = range(zF₋, zF₊; length = Nz+1)

    xC = range(xC₋, xC₊; length = Nx)
    yC = range(yC₋, yC₊; length = Ny)
    zC = range(zC₋, zC₊; length = Nz)


    return RegularRectilinearGrid{FT, TX, TY, TZ, typeof(xC)}(
        Nx, Ny, Nz, Lx, Ly, Lz, Δx, Δy, Δz, xC, yC, zC, xF, yF, zF)
end


function validate_regular_grid_domain(TX, TY, TZ, FT, extent, x, y, z)

    # Find domain endpoints or domain extent, depending on user input:
    if !isnothing(extent) # the user has specified an extent!

        (!isnothing(x) || !isnothing(y) || !isnothing(z)) &&
            throw(ArgumentError("Cannot specify both 'extent' and 'x, y, z' keyword arguments."))

        
        if length(extent)==2 
            extent = (extent[1],0, extent[2])   # 2D case
        end

        Lx, Ly, Lz = extent 

        # Default domain:
        x = (0, Lx)
        y = (0, Ly)
        z = (-Lz, 0)

    else # isnothing(extent) === true implies that user has not specified a length
        if isnothing(y)
            y = (0,0)
        end
        (isnothing(x) || isnothing(z)) &&
            throw(ArgumentError("You need to specify both 'x, z' or 'extent' keyword arguments.")) 

        Lx = x[2] - x[1]
        Ly = y[2] - y[1]
        Lz = z[2] - z[1]
    end

    return FT(Lx), FT(Ly), FT(Lz), FT.(x), FT.(y), FT.(z)
end



function domain_string(grid)
    xₗ, xᵣ = grid.xF[1], grid.xF[end]  
    yₗ, yᵣ = grid.yF[1], grid.yF[end]  
    zₗ, zᵣ = grid.zF[1], grid.zF[end]  
    return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ], z ∈ [$zₗ, $zᵣ]"
end

function show(io::IO, g::RegularRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RegularRectilinearGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "grid spacing (Δx, Δy, Δz): ", (g.Δx, g.Δy, g.Δz))
end
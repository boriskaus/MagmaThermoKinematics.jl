module Grids

# Grid infrastructure. Note that much of this is taken from Oceananigans

#####
##### Place-holder functions
#####



function fields end

export
    Center, Face,
    AbstractTopology, Periodic, Bounded, Flat2D, topology,
    AbstractGrid,
    AbstractRectilinearGrid, RegularRectilinearGrid

import Base: size, length, eltype, show

#####
##### Abstract types
#####

"""
    AbstractField{X, Y, Z, A, G}

Abstract supertype for fields located at `(X, Y, Z)` with data stored in a container
of type `A`. The field is defined on a grid `G`.
"""
abstract type AbstractField{X, Y, Z, A, G} end

"""
    Center

A type describing the location at the center of a grid cell.
"""
struct Center end

"""
	Face

A type describing the location at the face of a grid cell.
"""
struct Face end

"""
    AbstractTopology

Abstract supertype for grid topologies.
"""
abstract type AbstractTopology end

"""
    Periodic

Grid topology for periodic dimensions.
"""
struct Periodic <: AbstractTopology end

"""
    Bounded

Grid topology for bounded dimensions. These could be wall-bounded dimensions
or dimensions
"""
struct Bounded <: AbstractTopology end

"""
    Flat2D

Grid topology for flat (2D) dimensions, generally with one grid point, along which the solution
is uniform and does not vary.
"""
struct Flat2D <: AbstractTopology end

"""
    AbstractGrid{FT, TX, TY, TZ}

Abstract supertype for grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractGrid{FT, TX, TY, TZ} end

"""
    AbstractRectilinearGrid{FT, TX, TY, TZ}

Abstract supertype for rectilinear grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractRectilinearGrid{FT, TX, TY, TZ} <: AbstractGrid{FT, TX, TY, TZ} end


Base.eltype(::AbstractGrid{FT}) where FT = FT
Base.size(grid::AbstractGrid) = (grid.Nx, grid.Ny, grid.Nz)
Base.length(grid::AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)


#topology(::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = (TX, TY, TZ)
#topology(grid, dim) = topology(grid)[dim]

include("regular_rectilinear_grid.jl")

end

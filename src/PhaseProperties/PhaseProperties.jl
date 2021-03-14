module PhaseProperties

# defines structs to hold phase properyies

using StructArrays

export
    ThermalPhaseProp, ThermalProp, StokesProp, PhaseProp
   
import Base: eltype, show

#####
##### Abstract types
#####



"""
PhaseProp

Abstract supertype for phase properties
"""
abstract type PhaseProp end

"""
ThermalProp

Thermal phase properties
"""
struct ThermalProp <: PhaseProp end


"""
StokesProp

Stokes phase properties
"""
struct StokesProp <: PhaseProp end

"""
    AbstractPhaseProp{FT}

Abstract supertype for grids with elements of type `FT` and Phase Properties of type `PT`.
"""
abstract type AbstractPhaseProp{FT, PT} end

"""
    AbstractThermalPhaseProp{FT}

Abstract supertype for thermal phase properties with elements of type `FT`.
"""
abstract type AbstractThermalPhaseProp{FT, PT} <: AbstractPhaseProp{FT, PT} end


"""
    AbstractStokesPhaseProp{FT}

Abstract supertype for stokes phase properties with elements of type `FT`.
"""
abstract type AbstractStokesPhaseProp{FT, PT} <: AbstractPhaseProp{FT, PT} end

Base.eltype(::AbstractPhaseProp{FT}) where FT = FT

include("thermal_phase_props.jl")

end

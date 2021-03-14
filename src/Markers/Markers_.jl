module Markers_

export Markers, Marker, MarkerTemp #, update_particle_properties!

using StructArrays

import Base: size, length, show

abstract type AbstractMarker end

# Marker with x,y,z coordinates
struct Marker{T} <: AbstractMarker
    x :: T
    y :: T
    z :: T
end

# Marker that have a phase and temperature 
# (note that you can define additional marker structures within your own code)
struct MarkerTemp{T, I} <: AbstractMarker
    x       ::  T
    y       ::  T
    z       ::  T
    T       ::  T
    phase   ::  I    
end

# Overall tracers structure 
struct Markers{P, T, D, Π}
        properties :: P
    tracked_fields :: T
          dynamics :: D
        parameters :: Π
end

@inline no_dynamics(args...) = nothing  # for now, we define nothing; yet later we 

"""
    Tracers(; x, y, z, dynamics=no_dynamics, parameters=nothing)
Construct some `Tracers` that can be used within a simulation. The tracers will have initial locations
`x`, `y`, and `z`. `dynamics` is a function of `(tracers, model, Δt)` that is called prior to advecting tracers.
`parameters` can be accessed inside the `dynamics` function.
"""
function Markers(; x, y, z, tracked_fields::NamedTuple=NamedTuple(), dynamics=no_dynamics, parameters=nothing)
    size(x) == size(y) == size(z) ||
        throw(ArgumentError("x, y, z must all have the same size!"))

    (ndims(x) == 1 && ndims(y) == 1 && ndims(z) == 1) ||
        throw(ArgumentError("x, y, z must have dimension 1 but ndims=($(ndims(x)), $(ndims(y)), $(ndims(z)))"))

    markers = StructArray{Marker}((x, y, z))

    return Markers(markers; tracked_fields, dynamics, parameters)
    
end

"""
    Markers(; x, y, z, T, Phase, dynamics=no_dynamics, parameters=nothing)
Construct some `Markers` that can be used within a simulation. The markers will have initial locations
`x`, `y`, and `z`, temperature `T` and phase number `Phase` (Int64). `dynamics` is a function of `(markers, model, Δt)` that is called prior to advecting markers.
`parameters` can be accessed inside the `dynamics` function.
"""
function Markers(x, y, z, T, Phase; dynamics=no_dynamics, parameters=nothing)
    size(x) == size(y) == size(z) == size(T) == size(Phase) ||
        throw(ArgumentError("x, y, z, T, Phase must all have the same size!"))

    (ndims(x) == 1 && ndims(y) == 1 && ndims(z) == 1 && ndims(T) == 1 && ndims(Phase) == 1) ||
        throw(ArgumentError("x, y, z, T, Phase must have dimension 1 but ndims=($(ndims(x)), $(ndims(y)), $(ndims(z)), $(ndims(T)), $(ndims(Phase)) )"))

    markers = StructArray{MarkerTemp}((x, y, z, T, Phase))
    
    tracked_fields=(T=:T,);
    
    return Markers(markers; tracked_fields, dynamics, parameters)
    
end

"""
    Markers(markers::StructArray; tracked_fields::NamedTuple=NamedTuple(), dynamics=no_dynamics)
Construct `Markers` that can be passed to a model. The `markers` should be a `StructArray`
and can contain custom fields.x
A number of `tracked_fields` may be passed in as a `NamedTuple` of fields. Each particle will track the value of each
field. Each tracked field must have a corresponding particle property. So if `T` is a tracked field, then `T` must also
be a custom tracer property.
`dynamics` is a function of `(tracers, model, Δt)` that is called prior to advecting tracers.
`parameters` can be accessed inside the `dynamics` function.
"""
function Markers(markers::StructArray; tracked_fields::NamedTuple=NamedTuple(),
                             dynamics=no_dynamics, parameters=nothing)

    for (field_name, tracked_field) in pairs(tracked_fields)
        field_name in propertynames(markers) ||
            throw(ArgumentError("$field_name is a tracked field but $(eltype(markers)) has no $field_name field! " *
                                "You might have to define your own marker type."))
    end
    
    return Markers(markers, tracked_fields, dynamics, parameters)

end

size(markers::Markers)      =   size(markers.properties)
length(markers::Markers)    =   length(markers.properties)

function Base.show(io::IO, markers_struct::Markers)
    markers = markers_struct.properties
    properties = propertynames(markers)
    fields = markers_struct.tracked_fields
    print(io, "$(length(markers)) Markers with\n",
        "├── $(length(markers)) properties: $properties\n",
        "└── $(length(markers)) tracked fields: $(propertynames(fields))")
end

#include("update_particle_properties.jl")

end # module
"""
    fields = add_field(fields::NamedTuple, name::Symbol, newfield)

Adds a new field to the fields NamedTuple.
"""
function add_field(fields::NamedTuple, name::Symbol, newfield)
    return merge(fields, NamedTuple{(name,)}((newfield,)))
end

"""
    Arrays = CreateArrays(SizeNames::AbstractDict; backend=CPU(), FloatType=Float64)

Allocate the requested arrays on the KernelAbstractions `backend` (`CPU()`, or
e.g. `CUDABackend()` once CUDA.jl is loaded),
with element type `FloatType`, and initialize them with the requested values.
Returns a NamedTuple that contains all created arrays.
"""
function CreateArrays(SizeNames::AbstractDict; backend=CPU(), FloatType=Float64)
    arrays_out = NamedTuple()

    for (sz, arrays) in pairs(SizeNames)
        for (name, value) in pairs(arrays)
            data = KernelAbstractions.allocate(backend, FloatType, sz...)
            fill!(data, value)
            arrays_out = add_field(arrays_out, name, data)
        end
    end

    return arrays_out
end

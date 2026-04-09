"""
    fields = add_field(fields::NamedTuple, name::Symbol, newfield)

Adds a new field to the fields NamedTuple.
"""
function add_field(fields::NamedTuple, name::Symbol, newfield)
    return merge(fields, NamedTuple{(name,)}((newfield,)))
end

"""
    Arrays = CreateArrays(SizeNames::AbstractDict)

Initializes ParallelStencil arrays with the requested sizes and values.
Returns a NamedTuple that contains all created arrays.
"""
function CreateArrays(SizeNames::AbstractDict)
    arrays_out = NamedTuple()

    for (sz, arrays) in pairs(SizeNames)
        for (name, value) in pairs(arrays)
            data = @ones(sz...) * value
            arrays_out = add_field(arrays_out, name, data)
        end
    end

    return arrays_out
end

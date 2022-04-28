module Fields
using MagmaThermoKinematics
#environment!(:cpu, Float64, 2) 

using ParallelStencil
using CUDA

# Some helping routines that simplifies creating fields and work arrays 
export CreateArrays

"""


This initializes ParallelStencil arrays with the sizes you need for the calculations.
The arrays are initialized to a constant value (indicated by a Named Tuple) and the routine
returns a NamedTuple that contains all arrays

Example
====
```julia
julia> Arrays = CreateArrays(Dict( (100,100)=>(A=0,B=1,C=0), 
                                   (101,100)=>(E=0,))) 
(A = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], B = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], C = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], E = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])

julia> size(Arrays.E)
(101, 100)
```
"""
function CreateArrays(SizeNames::Dict )

    Arrays = NamedTuple()
    array_sizes = collect(keys(SizeNames))
    for sz in array_sizes
        arrays = get(SizeNames,sz,0)
        array_names = keys(arrays)
        for i = 1:length(arrays)
            data    = @ones(sz...)*arrays[i]     # initialize ParallelStencil array with correct size
            Arrays  = add_field(Arrays,array_names[i], data)
        end
    end

    return Arrays
end



"""
    fields = add_field(fields::NamedTuple, name::Symbol, newfield)

Adds a new field to the `fields` NamedTuple
"""
function add_field(fields::NamedTuple, name::Symbol, newfield)
    
    fields = merge(fields, NamedTuple{(name,)}( (newfield,)) );
    
    return fields
end





end
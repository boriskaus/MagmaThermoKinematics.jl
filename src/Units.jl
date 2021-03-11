module Units

export SecYear, second, yr, kyr, myr, meter, km, km³ 

#####
##### Convenient definitions
#####

"""
    second

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `Δt = 1second`.
"""
const second = 1.0

"""
    SecYear

A `Float64` constant equal to 365.25 * 24 * 3600. 
"""
const SecYear = 365.25*24*3600*second


"""
    year

A `Float64` constant equal to 365.25 * 24 * 3600 seconds. Useful for increasing the clarity of scripts, e.g. `Δt = 7yr`.
"""
const yr = 365.25*24*3600*second

"""
    kyr

A `Float64` constant equal to 1000`year`. Useful for increasing the clarity of scripts, e.g. `Δt = 10kyr`.
"""
const kyr = 1000yr

"""
    myr

A `Float64` constant equal to 10^6`year`. Useful for increasing the clarity of scripts, e.g. `Δt = 1myr`.
"""
const myr = 1000kyr

"""
    meter

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `L = 1meter`.
"""
const meter = 1.0

"""
    km

A `Float64` constant equal to 1000`m`. Useful for increasing the clarity of scripts, e.g. `L = 50km`.
"""
const km = 1000meter

"""
    km³

A `Float64` constant equal to 1000`m³`. Useful for increasing the clarity of scripts, e.g. `Vol = 50km³`.
"""
const km³ = 1e9*meter

end

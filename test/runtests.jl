using MagmaThermoKinematics, ParallelTestRunner

testsuite = find_tests(@__DIR__)
parsed_args = ParallelTestRunner.parse_args(copy(ARGS))

# Group tests by dimensionality, then run them in one ParallelTestRunner session.
testsuite_2d = Pair{String, Expr}[]
testsuite_3d = Pair{String, Expr}[]
testsuite_other = Pair{String, Expr}[]

for (name, expr) in testsuite
    if occursin("2D", name)
        push!(testsuite_2d, name => expr)
    elseif occursin("3D", name)
        push!(testsuite_3d, name => expr)
    else
        push!(testsuite_other, name => expr)
    end
end

testsuite_grouped = Dict{String, Expr}()
for (name, expr) in vcat(testsuite_2d, testsuite_3d, testsuite_other)
    testsuite_grouped[name] = expr
end

try
    ParallelTestRunner.runtests(MagmaThermoKinematics, parsed_args; testsuite = testsuite_grouped)
finally
    foreach(f -> rm(joinpath(@__DIR__, f)), filter(endswith(".png"), readdir(@__DIR__)))
end

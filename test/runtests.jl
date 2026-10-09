using MagmaThermoKinematics, ParallelTestRunner

testsuite = find_tests(@__DIR__)
delete!(testsuite, "run_example")     # helper script run by test_examples

try
    ParallelTestRunner.runtests(MagmaThermoKinematics, ARGS; testsuite)
finally
    foreach(f -> rm(joinpath(@__DIR__, f)), filter(endswith(".png"), readdir(@__DIR__)))
end

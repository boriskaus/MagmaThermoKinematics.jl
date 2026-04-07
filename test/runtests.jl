using MagmaThermoKinematics, ParallelTestRunner

testsuite = find_tests(@__DIR__)

try
    ParallelTestRunner.runtests(MagmaThermoKinematics, ARGS; testsuite)
finally
    foreach(f -> rm(joinpath(@__DIR__, f)), filter(endswith(".png"), readdir(@__DIR__)))
end

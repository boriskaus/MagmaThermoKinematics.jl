using MagmaThermoKinematics, ParallelTestRunner

function test_worker(name)
    if name == "test_MTK_GMG_3D"
        return addworker()
    else
        return nothing
    end
end

testsuite = find_tests(@__DIR__)

try
    ParallelTestRunner.runtests(MagmaThermoKinematics, ARGS; testsuite, test_worker)
finally
    foreach(f -> rm(joinpath(@__DIR__, f)), filter(endswith(".png"), readdir(@__DIR__)))
end

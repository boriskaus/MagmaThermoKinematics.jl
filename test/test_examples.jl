# Runs every `examples/MTK_GMG_*.jl` script for a few time steps, each in its own
# process. Opt-in (MTK_TEST_EXAMPLES=1): the scripts run in the package's own
# environment; the Lanin scripts, which need GMT and network access, are skipped.
using Test

if get(ENV, "MTK_TEST_EXAMPLES", "0") == "1"
    @testset "examples" begin
        dir = normpath(joinpath(@__DIR__, "..", "examples"))
        runner = joinpath(@__DIR__, "run_example.jl")
        scripts = filter(f -> startswith(f, "MTK_GMG_") && endswith(f, ".jl") && !occursin("Lanin", f), readdir(dir))
        @testset "$f" for f in scripts
            # Scripts write their output to the working directory.
            cmd = Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(dir)) $runner $(joinpath(dir, f))`; dir=mktempdir())
            out = IOBuffer()
            ok = success(pipeline(cmd; stdout=out, stderr=out))
            ok || print(String(take!(out)))
            @test ok
        end
    end
end

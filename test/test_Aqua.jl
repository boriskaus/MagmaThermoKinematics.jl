using Test
using MagmaThermoKinematics
using Aqua

@testset "Aqua" begin
    # `undefined_exports=false`: `@reexport using GeoParams` and `@reexport using
    # InjectSills` pull in names those packages themselves export without
    # defining; not fixable from this package.
    #
    # `stale_deps` ignore list: `TimerOutputs`, `MAT`, `CairoMakie`, and `Plots`
    # are used only by `examples/`, `docs/`, and a few tests, never by `src/`;
    # kept in `[deps]` so `julia --project=.` can run those scripts directly.
    #
    # `persistent_tasks=false`: the check instantiates every dependency in a
    # fresh environment and fails on GeoParams' dependency `InternedStrings`,
    # which ships without a `Project.toml`.
    Aqua.test_all(MagmaThermoKinematics;
        undefined_exports=false,
        persistent_tasks=false,
        stale_deps=(ignore=[:TimerOutputs, :MAT, :CairoMakie, :Plots],))
end

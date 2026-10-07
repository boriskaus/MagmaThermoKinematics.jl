using Test
using MagmaThermoKinematics
using Aqua

@testset "Aqua" begin
    # `undefined_exports=false`: `@reexport using GeoParams` and `@reexport using
    # InjectSills` pull in names those packages themselves export without
    # defining; not fixable from this package.
    #
    # `persistent_tasks=false`: the check instantiates every dependency in a
    # fresh environment and fails on GeoParams' dependency `InternedStrings`,
    # which ships without a `Project.toml`.
    Aqua.test_all(MagmaThermoKinematics;
        undefined_exports=false,
        persistent_tasks=false)
end

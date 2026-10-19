using Test
using MagmaThermoKinematics

@testset "sticky air BC" begin

    @testset "2D" begin
        T = fill(500.0, 8, 6)
        Phases = ones(Int64, 8, 6)
        Phases[:, 5:6] .= 0                 # two rows of air on top

        bc_sticky_air!(T, Phases, 0, 10.0)

        @test all(T[:, 5:6] .== 10.0)       # air is held at Tsurface
        @test all(T[:, 1:4] .== 500.0)      # the rock is untouched
    end

    @testset "3D" begin
        T = fill(500.0, 4, 5, 6)
        Phases = ones(Int64, 4, 5, 6)
        Phases[:, :, 6] .= 0

        bc_sticky_air!(T, Phases, 0, -5.0)

        @test all(T[:, :, 6] .== -5.0)
        @test all(T[:, :, 1:5] .== 500.0)
    end

    @testset "follows an irregular air/rock boundary" begin
        T = fill(300.0, 5, 5)
        Phases = ones(Int64, 5, 5)
        Phases[1, 3:5] .= 0                 # a topographic step: air reaches deeper in column 1
        Phases[2:5, 4:5] .= 0

        bc_sticky_air!(T, Phases, 0, 0.0)

        @test all(T[Phases .== 0] .== 0.0)
        @test all(T[Phases .== 1] .== 300.0)
    end

    @testset "respects the AirPhase number" begin
        T = fill(100.0, 4, 4)
        Phases = fill(7, 4, 4)
        Phases[:, 4] .= 2

        bc_sticky_air!(T, Phases, 2, 25.0)   # phase 2 is the air here

        @test all(T[:, 4] .== 25.0)
        @test all(T[:, 1:3] .== 100.0)
    end

    @testset "element type is preserved" begin
        T = fill(500.0f0, 4, 4)
        Phases = ones(Int64, 4, 4)
        Phases[:, 4] .= 0

        bc_sticky_air!(T, Phases, 0, 0.0f0)

        @test eltype(T) === Float32
        @test all(T[:, 4] .== 0.0f0)
    end

    @testset "NumParam carries the options" begin
        Num = NumParam(sticky_air_BC = true, AirPhase = 3, Tsurface_Celsius = 12.0)
        @test Num.sticky_air_BC
        @test Num.AirPhase == 3
        @test Num.Tsurface_Celsius == 12.0

        @test !NumParam().sticky_air_BC          # off by default
        @test NumParam().AirPhase == 0
    end

end

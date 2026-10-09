using Test, LinearAlgebra, SpecialFunctions, Random
using InjectSills

using MagmaThermoKinematics
using GeophysicalModelGenerator, GeoParams

const rng = Random.seed!(1234);     # same seed such that we can reproduce results

# Import a few routines, so we can overwrite them below
import MagmaThermoKinematics.MTK_GMG

@testset "MTK_GMG_2D" begin
    #=
    function MTK_GMG.MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
        println("$(Num.it), Time=$(round(Num.time/SecYear)) yrs; max(T) = $(round(maximum(Arrays.Tnew)))")
        return nothing
    end
=#
    # Test setup
    println("===============================================")
    println("Testing the MTK - GMG integration")
    println("===============================================")

    # These are the final simulations for the ZASSy paper, but done @ a lower resolution
    Num = NumParam( #Nx=269*1, Nz=269*1,
        Nx = 65 * 1, Nz = 65 * 1,
        SimName = "MTK_GMG_2D_Geneva", axisymmetric = false,
        #maxTime_Myrs=1.5,
        maxTime_Myrs = 0.025,
        fac_dt = 0.2, ω = 0.5, verbose = false,
        flux_bottom_BC = false, flux_bottom = 0, deactivate_La_at_depth = false,
        Geotherm = 30 / 1.0e3, TrackTracersOnGrid = true,
        SaveOutput_steps = 100000, CreateFig_steps = 100000, plot_tracers = false, advect_polygon = true,
        FigTitle = "Geneva Models, Geotherm 30/km"
    )

    Sill_params = SillParams(
        sill = CylindricalDikeTopAccretion(Center = Point2(0.0, -7.0e3) * m, W = 20.0e3 * m, H = 74.6269 * m),
        InjectionInterval_year = 5000,       # flux= 14.9e-6 km3/km2/yr
        nTr_dike = 300 * 1
    )

    MatParam = (
        SetMaterialParams(
            Name = "Rock & partial melt", Phase = 1,
            Density = ConstantDensity(ρ = 2700kg / m^3),
            LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
            #LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
            Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
            #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
            HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
            Melting = SmoothMelting(MeltingParam_4thOrder())
        ),      # Marxer & Ulmer melting
        # Melting = MeltingParam_Caricchi()),                     # Caricchi melting
        # add more parameters here, in case you have >1 phase in the model
    )

    # Call the main code with the specified material parameters
    Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params)  # start the main code

    @test sum(Arrays.Tnew) / prod(size(Arrays.Tnew)) ≈ 296.4607300089425  rtol = 1.0e-4
    @test sum(time_props.MeltFraction) ≈ 0.0  rtol = 1.0e-5
    @test !isempty(Dikes.sill_poly)


    # -----------------------------


    Topo_cart = load_GMG(normpath(joinpath(@__DIR__, "..", "examples", "Topo_cart")))       # Note: Laacher seee is around [10,20]

    # Create 3D grid of the region
    X, Y, Z = xyz_grid(-23:0.1:23, -19:0.1:19, -20:0.1:5)
    Data_set3D = CartData(X, Y, Z, (Phases = zeros(Int64, size(X)), Temp = zeros(size(X))))        # 3D dataset

    # Create 2D cross-section
    Nx = Num.Nx   # resolution in x
    Nz = Num.Nz
    Data_2D = cross_section(Data_set3D, Start = (-20, 4), End = (20, 4), dims = (Nx, Nz))
    Data_2D = addfield(Data_2D, "FlatCrossSection", flatten_cross_section(Data_2D))
    Data_2D = addfield(Data_2D, "Phases", Int64.(Data_2D.fields.Phases))

    # Intersect with topography
    Below = below_surface(Data_2D, Topo_cart)
    Data_2D.fields.Phases[Below] .= 1

    # Set Moho
    ind = findall(Data_2D.z.val .< -30.0)
    Data_2D.fields.Phases[ind] .= 2

    # Set T:
    gradient = 30
    Data_2D.fields.Temp .= -Data_2D.z.val * gradient
    ind = findall(Data_2D.fields.Temp .< 10.0)
    Data_2D.fields.Temp[ind] .= 10.0

    # Set thermal anomaly
    x_c, z_c, r = -10, -15, 2.5
    Volume = 4 / 3 * pi * r^3 # equivalent 3D volume of the anomaly [km^3]
    ind = findall((Data_2D.x.val .- x_c) .^ 2 .+ (Data_2D.z.val .- z_c) .^ 2 .< r^2)
    Data_2D.fields.Temp[ind] .= 800.0

    """
    Randomly change orientation and location of a dike
    """
    function MTK_GMG.MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::SillParameters, Num::NumericalParameters)
        if mod(Num.it, 10) == 0
            cen = (Grid.max .+ Grid.min) ./ 2 .+ 0 * rand(rng, -0.5:1.0e-3:0.5, 2) .* [Dikes.W_ran; Dikes.H_ran]     # Randomly vary center of dike
            if cen[end] < -15.0e3
                Angle_rand = 0 * rand(rng, 80.0:0.1:100.0)                                              # Orientation: near-vertical @ depth
            else
                Angle_rand = 0 * rand(rng, -10.0:0.1:10.0)
            end

            if hasproperty(Dikes, :sill) && !isnothing(Dikes.sill)
                Dikes.sill = InjectSills.update_abstractsill(
                    Dikes.sill;
                    Center = Point2(cen[1], cen[2]) * m,
                    Angle = Vec1(Angle_rand) * NoUnits
                )
            else
                Dikes.Center = cen
                Dikes.Angle = [Angle_rand]
            end
        end
        return nothing
    end

    # Define numerical parameters
    Num = NumParam(
        SimName = "Unzen1", axisymmetric = false,
        maxTime_Myrs = 0.005,
        fac_dt = 0.2, ω = 0.5, verbose = false,
        SaveOutput_steps = 10000, CreateFig_steps = 1000, plot_tracers = false, advect_polygon = false
    )

    # dike parameters
    Sill_params = SillParams(
        sill = EllipticalIntrusion(Center = Point2(0.0, -7.0e3) * m, W = 5.0e3 * m, H = 250 * m),
        InjectionInterval_year = 1000,       # flux= 14.9e-6 km3/km2/yr
        nTr_dike = 300 * 1,
        H_ran = 5000, W_ran = 5000,
        SillPhase = 3, BackgroundPhase = 1,
    )

    # Define parameters for the different phases
    MatParam = (
        SetMaterialParams(
            Name = "Air", Phase = 0,
            Density = ConstantDensity(ρ = 2700kg / m^3),
            LatentHeat = ConstantLatentHeat(Q_L = 0.0J / kg),
            Conductivity = ConstantConductivity(k = 3Watt / K / m),          # in case we use constant k
            HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
            Melting = SmoothMelting(MeltingParam_4thOrder())
        ),          # Marxer & Ulmer melting

        SetMaterialParams(
            Name = "Crust", Phase = 1,
            Density = ConstantDensity(ρ = 2700kg / m^3),
            LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
            Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
            #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
            HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
            Melting = SmoothMelting(MeltingParam_4thOrder())
        ),      # Marxer & Ulmer melting

        SetMaterialParams(
            Name = "Mantle", Phase = 2,
            Density = ConstantDensity(ρ = 2700kg / m^3),
            LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
            Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
            HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K)
        ),

        SetMaterialParams(
            Name = "Dikes", Phase = 3,
            Density = ConstantDensity(ρ = 2700kg / m^3),
            LatentHeat = ConstantLatentHeat(Q_L = 3.13e5J / kg),
            #     Conductivity = ConstantConductivity(k=3.3Watt/K/m),          # in case we use constant k
            Conductivity = T_Conductivity_Whittington_parameterised(),   # T-dependent k
            #Conductivity = T_Conductivity_Whittington(),                 # T-dependent k
            HeatCapacity = ConstantHeatCapacity(Cp = 1000J / kg / K),
            Melting = SmoothMelting(MeltingParam_4thOrder())
        ),      # Marxer & Ulmer melting

    )


    # Call the main code with the specified material parameters
    Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params, CartData_input = Data_2D)  # start the main code

    @test sum(Arrays.Tnew) / prod(size(Arrays.Tnew)) ≈ 251.58206620240594  rtol = 1.0e-4
    @test sum(time_props.MeltFraction) ≈ 0.2238128607809668 rtol = 1.0e-5

    # Random sill placement moves the sill within the randomization zone
    Num.AddRandomSills, Num.it = true, 0
    MTK_GMG.MTK_update_ArraysStructs!(Arrays, Grid, Dikes, Num, MatParam)
    cen = [c.val for c in Dikes.sill.Center]
    @test Dikes.sill isa EllipticalIntrusion
    @test all(abs.(cen .- (Grid.max .+ Grid.min) ./ 2) .<= [Dikes.W_ran, Dikes.H_ran] ./ 2)

    # remove directory created by this test
    rm("MTK_GMG_2D_Geneva", recursive = true, force = true)
    rm("Unzen1", recursive = true, force = true)

end

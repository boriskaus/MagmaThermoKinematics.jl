using Test, LinearAlgebra, SpecialFunctions, Random
using MagmaThermoKinematics

Random.seed!(1234);     # such that we can reproduce results

const CreatePlots = false      # easy way to deactivate plotting throughout
if CreatePlots
    using CairoMakie: Figure, Axis, DataAspect, lines!, scatter!, heatmap!, axislegend, save
end

function Diffusion_SteadyState2D(Setup = "Constant_Zdirection")
    # steady state diffusion in x and z-direction for constant and variable K

    # Model parameters
    if Setup == "Constant_Zdirection" || Setup == "VariableK_Zdirection"
        W, H = 3, 30                               # Width, Height in km
        k_rock1 = 3
        k_rock2 = 1.5
        Nx, Nz = 10, 100                             # resolution
        Tbot = 600
        H1 = 20
        GeoT = Tbot / H
    elseif Setup == "Constant_Xdirection" || Setup == "VariableK_Xdirection"
        W, H = 40, 3                               # Width, Height in km
        k_rock1 = 3
        k_rock2 = 1.5
        Nx, Nz = 100, 10                             # resolution
        Tbot = 600
        W1 = 20
        GeoT = Tbot / W
    else
        error("unknown setup")
    end

    SecYear = 365.25 * 24 * 3600
    ρ = 2800                                # Density
    cp = 1050                                # Heat capacity
    L = 350.0e3                               # Latent heat J/kg/K
    dx = W / (Nx - 1) * 1.0e3; dz = H * 1.0e3 / (Nz - 1)     # grid size [m]
    κ = k_rock1 ./ (ρ * cp)                      # thermal diffusivity
    dt = min(dx^2, dz^2) ./ κ / 10              # stable timestep (required for explicit FD)

    # Array initializations (1 - main arrays on which we can initialize properties)
    T = zeros(Nx, Nz)
    K = fill(Float64(k_rock1), Nx, Nz)
    Rho = fill(Float64(ρ), Nx, Nz)
    Hs = zeros(Nx, Nz)
    Hl = fill(Float64(L), Nx, Nz)
    Cp = fill(Float64(cp), Nx, Nz)
    dPhi_dt = zeros(Nx, Nz)

    # Work array initialization
    Tnew = zeros(Nx, Nz)                                             # thermal solver
    X, Xc, Z = zeros(Nx, Nz), zeros(Nx - 1, Nz - 1), zeros(Nx, Nz)    # 2D gridpoints

    # Set up model geometry & initial T structure
    x, z = 0:dx:(W * 1.0e3), (-H * 1.0e3):dz:(-H * 1.0e3 + (Nz - 1) * dz)
    X, Z = ones(Nz)' .* x, z' .* ones(Nx)                              # 2D coordinate grids
    Xc = (X[2:Nx, :] + X[1:(Nx - 1), :]) / 2.0
    Grid, Spacing = (X, Z), (dx, dz)
    T .= Tbot                                              # initial (linear) temperature profile

    if Setup == "Constant_Zdirection" || Setup == "VariableK_Zdirection"
        # bottom BC
        T[1, :] .= 0
    else
        T[:, 1] .= 0
    end


    if Setup == "VariableK_Zdirection"
        # variable k
        K[Z .< -H1 * 1.0e3] .= k_rock2


    elseif Setup == "VariableK_Xdirection"
        # variable k
        K[X .< W1 * 1.0e3] .= k_rock2
    end


    time, time_kyrs, dike_inj = 0.0, 0.0, 0.0
    err = 100
    it = 0
    while (err > 1.0e-10) & (it < 1.0e6)

        it += 1
        # Perform a diffusion step
        diffusion_step!(Tnew, T, K, Rho, Cp, Hs, Hl, dt, (dx, dz), dPhi_dt)
        if Setup == "Constant_Zdirection" || Setup == "VariableK_Zdirection"
            # diffusion in z-direction
            bc_zero_flux!(Tnew, 1)                                                                      # set lateral boundary conditions (flux-free)
            Tnew[:, 1] .= Tbot; Tnew[:, end] .= 0.0                                                     # bottom & top temperature (constant)

        else
            # diffusion in x-direction
            bc_zero_flux!(Tnew, 2)                                                                      # set lateral boundary conditions (flux-free)
            Tnew[1, :] .= 0; Tnew[end, :] .= Tbot                                                     # bottom & top temperature (constant)
        end

        err = maximum(abs.(Tnew - T))

        T, Tnew = Tnew, T                                                                 # Update temperature
        time, time_kyrs = time + dt, time / SecYear / 1.0e3                                             # Keep track of evolved time

        if mod(it, 10000) == 0  # Visualisation
            #    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")
        end

    end

    x_km, z_km = x ./ 1.0e3, z ./ 1.0e3


    # compute analytical solution
    if Setup == "Constant_Zdirection"
        Tanal = -z_km .* GeoT
        Tnum = T[1, :]
        fname = "Diffusion_2D_SS_constantK_Z"

    elseif Setup == "VariableK_Zdirection"
        # 1D steady steate analytical solution for variable K is given by the folliwing balance equations
        #   k1*dT/dz|_1                     =   k2*dT/dz|_2     (heat flux)
        #   dT/dz|_1 * H1 + dT/dz|_2 * H2 =   Tbot            (assuming Ttop=0)
        #   H1 + H2                         =   H               (total thickness)
        #
        # substitute eq. 1 into eq 2 to eliminate dT/dz|_2
        #  dT/dz|_1 * H_1 + dT/dz|_1 * k1/k2*H2 =   Tbot
        #  dT/dz|_1 = Tbot/(H1 + k1/k2*H2)

        H2 = H - H1
        dTdz1 = Tbot / (H1 + k_rock1 / k_rock2 * H2)
        dTdz2 = k_rock1 / k_rock2 .* dTdz1

        Tanal = zeros(size(z_km));              Tanal2 = copy(Tanal)
        Tanal .= -z_km[:] .* dTdz1
        Tanal2 .= -z_km .* dTdz2 .- dTdz1 * H1
        Tanal[z_km .< -H1] .= Tanal2[z_km .< -H1]

        Tnum = T[1, :]
        fname = "Diffusion_2D_SS_variableK_Z"

    elseif Setup == "Constant_Xdirection"
        Tanal = x_km .* GeoT
        Tnum = T[:, 1]
        fname = "Diffusion_2D_SS_constantK_X"

    elseif Setup == "VariableK_Xdirection"
        W2 = W - W1
        dTdx1 = Tbot / (W1 + k_rock1 / k_rock2 * W2)
        dTdx2 = k_rock1 / k_rock2 .* dTdx1

        Tanal = zeros(size(x_km));              Tanal2 = copy(Tanal)
        Tanal .= x_km .* dTdx2
        Tanal2 .= x_km[:] .* dTdx1 .+ dTdx1 * W2
        Tanal[x_km .> W1] .= Tanal2[x_km .> W1]

        Tnum = T[:, 1]
        fname = "Diffusion_2D_SS_variableK_X"
    end

    if Setup == "Constant_Zdirection" || Setup == "VariableK_Zdirection"
        if CreatePlots
            # create plot
            fig = Figure()
            ax = Axis(fig[1, 1], xlabel = "Temperature [C]", ylabel = "Depth [km]")
            lines!(ax, Tanal, z_km, label = "Analytics")
            scatter!(ax, Tnum, z_km, markersize = 4, label = "Numerics")
            axislegend(ax)
        end
        error = norm(Array(T[1, :]) .- Tanal, 2)
    else
        if CreatePlots
            # create plot
            fig = Figure()
            ax = Axis(fig[1, 1], xlabel = "Width [km]", ylabel = "Temperature [C]")
            lines!(ax, x_km, Tanal, label = "Analytics")
            scatter!(ax, x_km, Tnum, markersize = 4, label = "Numerics")
            axislegend(ax)
        end
        error = norm(Array(T[:, 1]) .- Tanal, 2)
    end
    if CreatePlots
        save("$(fname).png", fig)
    end


    return error         # return error

end # end of steady state diffusion test

function Diffusion_Halfspace2D()
    # Halfspace cooling example

    # Model parameters
    W, H = 3, 300                              # Width, Height in km
    k_rock1 = 3
    Nx, Nz = 10, 100                             # resolution
    Tbot = 1200
    SecYear = 365.25 * 24 * 3600
    CoolingAge = 30.0e6 * SecYear                        # thermal cooling age
    ρ = 2800                                # Density
    cp = 1050                                # Heat capacity
    L = 350.0e3                               # Latent heat J/kg/K
    dx = W / (Nx - 1) * 1.0e3; dz = H * 1.0e3 / (Nz - 1)     # grid size [m]
    κ = k_rock1 ./ (ρ * cp)                      # thermal diffusivity
    dt = min(dx^2, dz^2) ./ κ / 10              # stable timestep (required for explicit FD)

    numTime = ceil(CoolingAge / dt)
    dt = CoolingAge / numTime
    nt = Int(numTime)

    # Array initializations (1 - main arrays on which we can initialize properties)
    T = fill(Float64(Tbot), Nx, Nz)
    K = fill(Float64(k_rock1), Nx, Nz)
    Rho = fill(Float64(ρ), Nx, Nz)
    Cp = fill(Float64(cp), Nx, Nz)
    Hs = zeros(Nx, Nz)
    Hl = zeros(Nx, Nz) * L
    dPhi_dt = zeros(Nx, Nz)

    # Work array initialization
    Tnew = zeros(Nx, Nz)                                             # thermal solver
    X, Xc, Z = zeros(Nx, Nz), zeros(Nx - 1, Nz - 1), zeros(Nx, Nz)    # 2D gridpoints

    # Set up model geometry & initial T structure
    x, z = 0:dx:(W * 1.0e3), (-H * 1.0e3):dz:(-H * 1.0e3 + (Nz - 1) * dz)
    X, Z = ones(Nz)' .* x, z' .* ones(Nx)                              # 2D coordinate grids
    Xc = (X[2:Nx, :] + X[1:(Nx - 1), :]) / 2.0
    Grid, Spacing = (X, Z), (dx, dz)
    T .= Tbot                                              # initial (linear) temperature profile

    T[:, end] .= 0      # top BC
    Tnew .= T

    time, time_kyrs, dike_inj = 0.0, 0.0, 0.0
    err = 100
    it = 0
    for it in 1:nt

        # Perform a diffusion step
        diffusion_step!(Tnew, T, K, Rho, Cp, Hs, Hl, dt, (dx, dz), dPhi_dt)

        # diffusion in z-direction
        bc_zero_flux!(Tnew, 1)                                                                      # set lateral boundary conditions (flux-free)
        Tnew[:, 1] .= Tbot; Tnew[:, end] .= 0.0                                                     # bottom & top temperature (constant)


        T, Tnew = Tnew, T                                                                 # Update temperature
        time, time_kyrs = time + dt, time / SecYear / 1.0e3                                             # Keep track of evolved time

        if mod(it, 1000) == 0  # print progress
            #    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")
        end

    end

    x_km, z_km = x ./ 1.0e3, z ./ 1.0e3


    # compute analytical solution
    Tanal = -z_km

    Tanal = (0 - Tbot) .* erfc.((abs.(z_km) .* 1.0e3) ./ (2 * sqrt(κ * CoolingAge))) .+ Tbot


    Tnum = T[1, :]
    fname = "Diffusion_2D_Halfspace"

    if CreatePlots
        # create plot
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Temperature [C]", ylabel = "Depth [km]")
        lines!(ax, Tanal, z_km, label = "Analytics")
        scatter!(ax, Tnum, z_km, markersize = 4, label = "Numerics")
        axislegend(ax)

        save("$(fname).png", fig)
    end

    error = norm(Array(T[1, :]) .- Tanal, 2)
    return error         # return error

end # end of halfspace cooling test


function Diffusion_Gaussian2D(Setup = "2D")
    # Gaussian diffusion test in 2D

    # Model parameters
    W, H = 300, 300                            # Width, Height in km
    k_rock1 = 3
    Nx, Nz = 100, 100                            # resolution
    Tbot = 0
    SecYear = 365.25 * 24 * 3600
    σ = 15.0e3                                # halfwidth of gaussian
    Tmax = 1000                                 # max. of gaussian
    TotalTime = 3.0e6 * SecYear                         # thermal cooling age
    ρ = 2800                                # Density
    cp = 1050                                # Heat capacity
    L = 350.0e3                               # Latent heat J/kg/K
    dx = W / (Nx - 1) * 1.0e3; dz = H * 1.0e3 / (Nz - 1)     # grid size [m]
    κ = k_rock1 ./ (ρ * cp)                     # thermal diffusivity
    dt = min(dx^2, dz^2) ./ κ / 10               # stable timestep (required for explicit FD)

    numTime = ceil(TotalTime / dt)
    dt = TotalTime / numTime / 100
    nt = Int(numTime)

    # Array initializations (1 - main arrays on which we can initialize properties)
    T = fill(Float64(Tbot), Nx, Nz)
    K = fill(Float64(k_rock1), Nx, Nz)
    Rho = fill(Float64(ρ), Nx, Nz)
    Cp = fill(Float64(cp), Nx, Nz)
    dPhi_dt = zeros(Nx, Nz)
    Hs = zeros(Nx, Nz)
    Hl = zeros(Nx, Nz)

    # Work array initialization
    Tnew = zeros(Nx, Nz)                                             # thermal solver
    X, Xc, Z = zeros(Nx, Nz), zeros(Nx - 1, Nz - 1), zeros(Nx, Nz)    # 2D gridpoints

    # Set up model geometry & initial T structure
    x, z = (-W / 2 * 1.0e3):dx:(W / 2 * 1.0e3), (-H / 2 * 1.0e3):dz:(-H / 2 * 1.0e3 + (Nz - 1) * dz)
    X, Z = ones(Nz)' .* x, z' .* ones(Nx)                              # 2D coordinate grids
    Xc = ((X[2:Nx, :] + X[1:(Nx - 1), :]) / 2.0)
    Grid, Spacing = (X, Z), (dx, dz)

    if Setup == "2D"
        T .= (Tmax .* exp.(-(X .^ 2 .+ Z .^ 2) ./ (σ^2)))                      # initial gaussian profile
    elseif Setup == "Axisymmetric"
        T .= (Tmax .* exp.(-(X .^ 2 .+ Z .^ 2) ./ (σ^2)))                      # initial gaussian profile
    else
        error("Unknown setup")
    end

    Tnew .= T

    #mkpath("viz2D_out")                            # directory for animation frames

    time, time_kyrs = 0.0, 0.0
    err = 100
    it = 0
    nt = 500
    for it in 1:nt

        # Perform a diffusion step
        if Setup == "2D"
            diffusion_step!(Tnew, T, K, Rho, Cp, Hs, Hl, dt, (dx, dz), dPhi_dt)
        elseif Setup == "Axisymmetric"
            diffusion_step!(Tnew, T, K, Rho, Cp, Hs, Hl, dt, (dx, dz), dPhi_dt; R = X)
        end

        # diffusion in z-direction
        bc_zero_flux!(Tnew, 1)                                                                      # set lateral boundary conditions (flux-free)
        Tnew[:, 1] .= Tbot; Tnew[:, end] .= 0.0                                                     # bottom & top temperature (constant)


        T, Tnew = Tnew, T                                                                 # Update temperature
        time, time_kyrs = time + dt, time / SecYear / 1.0e3                                             # Keep track of evolved time

        if mod(it, 50) == 0  # print progress
            #    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")

            #    x_km, z_km  =   x./1e3, z./1e3;
            #    fig = Figure()
            #    heatmap!(Axis(fig[1,1], title="Temperature, $(round(time_kyrs, digits=2)) kyrs", aspect=DataAspect()), x_km, z_km, T, colormap=:inferno)
            #    save("viz2D_out/Diffusion2D_$(it).png", fig)
        end

    end

    x_km, z_km = x ./ 1.0e3, z ./ 1.0e3


    # compute analytical solution

    if Setup == "2D"
        Tanal = Tmax ./ (1 + 4 * time * κ / σ^2)^(2 / 2) .* exp.(-(X .^ 2 .+ Z .^ 2) ./ (σ^2 + 4 * time * κ))                      # initial gaussian profile
        fname = "Diffusion_2D_Gaussian"

    elseif Setup == "Axisymmetric"
        # Axisymmetric is like 3D, where Y=X. Hence we can use the 3D solution, which is
        Tanal = Tmax ./ ((1 + 4 * time * κ / σ^2)^(3 / 2)) .* exp.(-(X .^ 2 .+ Z .^ 2) ./ (σ^2 + 4 * time * κ))                      # initial gaussian profile
        fname = "Diffusion_Axisymmetric_Gaussian"
    end
    Terror = Array(T) - Array(Tanal)

    if CreatePlots
        # create plot
        fig = Figure()
        heatmap!(Axis(fig[1, 1], title = "T error 2D $(round(time_kyrs / 1.0e3, digits = 2)) Myrs", aspect = DataAspect()), x_km, z_km, Terror, colormap = :inferno)
        save("$(fname).png", fig)
    end

    error = norm(Array(Terror[:]), 2)

    return error         # return error

end # end of gaussian diffusion test

# Create a range of 2D diffusion tests which calls the routines above
@testset "2D steady state diffusion" begin
    @test Diffusion_SteadyState2D("VariableK_Zdirection") ≈ 8.749073037094778  atol = 1.0e-5
    @test Diffusion_SteadyState2D("Constant_Zdirection") ≈ 1.0e-5   atol = 1.0e-5
    @test Diffusion_SteadyState2D("VariableK_Xdirection") ≈ 2.041013962386462  atol = 1.0e-5
    @test Diffusion_SteadyState2D("Constant_Xdirection") ≈ 1.0268559132246036e-5   atol = 1.0e-5
end;

@testset "2D halfspace cooling" begin
    err = Diffusion_Halfspace2D()
    @test  err ≈ 0.6339708451156041 atol = 1.0e-5
end;
@testset "2D Gaussian diffusion" begin
    @test Diffusion_Gaussian2D("2D") ≈ 5.229954229551127 atol = 1.0e-5
    @test Diffusion_Gaussian2D("Axisymmetric") ≈ 10.587520589916926 atol = 1.0e-5
end;

@testset "lithostatic pressure" begin
    ρ, g, Δz = 2700.0, 9.81, 100.0
    for dims in ((5, 21), (4, 3, 21))
        Rho = fill(ρ, dims)
        P = fill(NaN, dims)
        lithostatic_pressure!(P, Rho, g, Δz)
        H = Δz * (dims[end] - 1)
        @test all(==(0), selectdim(P, ndims(P), dims[end]))
        @test all(isapprox(ρ * g * H; rtol = 1.0e-12), selectdim(P, ndims(P), 1))
    end
    @test_throws "P and Rho must match" lithostatic_pressure!(zeros(3, 4), zeros(3, 5), g, Δz)
end;

@testset "no melt or latent heat below deactivationDepth" begin
    Mat = (
        SetMaterialParams(
            Phase = 1, Density = ConstantDensity(), LatentHeat = ConstantLatentHeat(),
            Conductivity = ConstantConductivity(), HeatCapacity = ConstantHeatCapacity(),
            Melting = SmoothMelting(MeltingParam_4thOrder())
        ),
    )
    N = (9, 21)
    Grid = CreateGrid(size = N, extent = (10.0e3, 20.0e3))
    names = (:T, :T_K, :Tnew, :T_it_old, :Tupdate, :Kc, :Rho, :Cp, :Hr, :Hl, :ϕ, :dϕdT, :R, :Z, :P)
    Arrays = CreateArrays(Dict(N => NamedTuple{names}(ntuple(_ -> 0, length(names)))))
    GridArray!(Arrays.R, Arrays.Z, Grid)
    Arrays.T .= 900.0                   # partially molten everywhere
    Arrays.Tnew .= Arrays.T
    Phases = ones(Int64, N)
    Num = Numeric_params(; deactivate_La_at_depth = true, deactivationDepth = -10.0e3)
    Nonlinear_Diffusion_step!(Arrays, Mat, Phases, Grid, 1.0e8, Num)
    deep = Arrays.Z .< -10.0e3
    @test all(iszero, Arrays.ϕ[deep])
    @test all(iszero, Arrays.dϕdT[deep])
    @test all(>(0), Arrays.ϕ[.!deep])
end;

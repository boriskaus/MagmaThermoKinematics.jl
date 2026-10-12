using Test, LinearAlgebra, SpecialFunctions, Random
using MagmaThermoKinematics

const CreatePlots = false      # easy way to deactivate plotting throughout
if CreatePlots
    using CairoMakie: Figure, Axis, DataAspect, heatmap!, save
end


function Diffusion_Gaussian3D(Setup = "3D")
    # Test the

    # Model parameters
    W, L, H = 300.0, 300.0, 300.0                                # Width, Length,    Height in km
    k_rock1 = 3
    Nx, Ny, Nz = 100, 100, 100                                   # resolution
    Tbot = 0
    SecYear = 365.25 * 24 * 3600
    σ = 15.0e3                                            # halfwidth of gaussian
    Tmax = 1000                                            # max. of gaussian
    TotalTime = 3.0e6 * SecYear                                     # thermal cooling age
    ρ = 2800                                            # Density
    cp = 1050                                            # Heat capacity
    La = 350.0e3                                           # Latent heat J/kg/K
    dx, dy, dz = W / (Nx - 1) * 1.0e3, L / (Ny - 1) * 1.0e3, H * 1.0e3 / (Nz - 1)        # grid size [m]
    κ = k_rock1 ./ (ρ * cp)                                 # thermal diffusivity
    dt = min(dx^2, dy^2, dz^2) ./ κ / 2                        # stable timestep (required for explicit FD)

    numTime = ceil(TotalTime / dt)
    dt = TotalTime / numTime / 20
    nt = Int(numTime)

    # Array initializations (1 - main arrays on which we can initialize properties)
    T = fill(Float64(Tbot), Nx, Ny, Nz)
    K = fill(Float64(k_rock1), Nx, Ny, Nz)
    Rho = fill(Float64(ρ), Nx, Ny, Nz)
    Cp = fill(Float64(cp), Nx, Ny, Nz)
    dPhi_dt = zeros(Nx, Ny, Nz)
    Hs = zeros(Nx, Ny, Nz)
    Hl = fill(Float64(La), Nx, Ny, Nz)

    # Work array initialization
    Tnew = zeros(Nx, Ny, Nz)                                                         # thermal solver
    X, Y, Z = zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz)                             # 3D gridpoints


    # Set up model geometry & initial T structure
    x, y, z = (-W / 2 * 1.0e3):dx:(-W / 2 * 1.0e3 + (Nx - 1) * dx), (-L / 2 * 1.0e3):dy:(-L / 2 * 1.0e3 + (Ny - 1) * dy), (-H / 2 * 1.0e3):dz:(-H / 2 * 1.0e3 + (Nz - 1) * dz)
    coords = collect(Iterators.product(x, y, z))                               # generate coordinates from 1D coordinate vectors
    X, Y, Z = (x -> x[1]).(coords), (x -> x[2]).(coords), (x -> x[3]).(coords)      # transfer coords to 3D arrays
    Grid, Spacing = (X, Y, Z), (dx, dy, dz)
    T .= (Tmax .* exp.(-((X .^ 2 .+ Y .^ 2 .+ Z .^ 2) ./ (σ^2))))                  # initial gaussian profile
    Tnew .= T


    #mkpath("viz2D_out")                            # directory for animation frames

    time, time_kyrs = 0.0, 0.0
    err = 100
    it = 0
    nt = 500
    for it in 1:nt

        # Perform a diffusion step
        if Setup == "3D"
            diffusion_step!(Tnew, T, K, Rho, Cp, Hs, Hl, dt, (dx, dy, dz), dPhi_dt)
        end

        # diffusion in z-direction
        bc_zero_flux!(Tnew, 1)                                                            # set lateral boundary conditions (flux-free)
        bc_zero_flux!(Tnew, 2)                                                            # set lateral boundary conditions (flux-free)

        Tnew[:, :, 1] .= Tbot; Tnew[:, :, end] .= 0.0                                                   # bottom & top temperature (constant)


        T, Tnew = Tnew, T                                                                 # Update temperature
        time, time_kyrs = time + dt, time / SecYear / 1.0e3                                             # Keep track of evolved time

        if mod(it, 100) == 0  # print progress
            # println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")
            #    x_km, z_km  =   x./1e3, z./1e3;
            #    fig = Figure()
            #    #heatmap!(Axis(fig[1,1], title="Temperature, $(round(time_kyrs, digits=2)) kyrs", aspect=DataAspect()), x_km, z_km, T[:,Int(Ny/2),:], colormap=:inferno)
            #    heatmap!(Axis(fig[1,1], title="Temperature, $(round(time_kyrs, digits=2)) kyrs", aspect=DataAspect()), y./1e3, z_km, T[Int(Nx/2),:,:], colormap=:inferno)
            #
            #    save("viz2D_out/Diffusion3D_$(it).png", fig)
        end

    end

    x_km, z_km = x ./ 1.0e3, z ./ 1.0e3


    # compute analytical solution

    if Setup == "3D"
        Tanal = Tmax ./ (1 + 4 * time * κ / σ^2)^(3 / 2) .* exp.(-(X .^ 2 .+ Y .^ 2 .+ Z .^ 2) ./ (σ^2 + 4 * time * κ))                      # initial gaussian profile
        fname = "Diffusion_3D_Gaussian"
    end
    Terror = Array(T) - Tanal

    Tslice = T[:, Int(Ny / 2), :]
    Tanal1 = Tanal[:, Int(Ny / 2), :]
    Terror1 = Terror[:, Int(Ny / 2), :]

    if CreatePlots
        # create plot
        fig = Figure(size = (1500, 450))
        heatmap!(Axis(fig[1, 1], title = "T  3D $(round(time_kyrs / 1.0e3, digits = 2)) Myrs", aspect = DataAspect()), x_km, z_km, Tslice, colormap = :inferno)
        heatmap!(Axis(fig[1, 2], title = "T anal 3D $(round(time_kyrs / 1.0e3, digits = 2)) Myrs", aspect = DataAspect()), x_km, z_km, Tanal1, colormap = :inferno)
        heatmap!(Axis(fig[1, 3], title = "T error 3D $(round(time_kyrs / 1.0e3, digits = 2)) Myrs", aspect = DataAspect()), x_km, z_km, Terror1, colormap = :inferno)
        save("$(fname).png", fig)
    end

    error = norm(Terror[:], 2) / length(Terror[:])

    return error         # return error
end # end of gaussian diffusion test

# Create a range of diffusion tests which calls the routines above
@testset "3D Gaussian diffusion" begin
    @test Diffusion_Gaussian3D("3D") ≈ 1.74e-5 atol = 1.0e-4
end;

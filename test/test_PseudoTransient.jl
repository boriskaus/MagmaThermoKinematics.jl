using Test, LinearAlgebra
using MagmaThermoKinematics

# Arrays of a small 2D model with constant or variable properties for the implicit (accelerated pseudo-transient) step
function pt_arrays(dims...; K = 3.0, ρ = 2700.0, cp = 1000.0, L = 0.0, Kfun = nothing, Hr = 0.0)
    z() = zeros(dims...)
    A = (;
        T = z(), T_it_old = z(), Tnew = z(), Kc = fill(K, dims...), Rho = fill(ρ, dims...), Cp = fill(cp, dims...),
        Hl = fill(L, dims...), dϕdT = z(), ϕ = z(), Hr = fill(Hr, dims...),
        pt_C = z(), pt_b = z(), pt_phiold = z(), pt_res = z(), pt_Tk = z(), pt_Tkm1 = z(), pt_Tn = z(),
    )
    isnothing(Kfun) || (A.Kc .= [Kfun(Tuple(I)...) for I in CartesianIndices(A.Kc)])
    return A
end

# N-dimensional dense backward-Euler system (same discretisation as the solver; optional cell radii `R` for the axisymmetric case):
# zero flux on the lateral faces (all but the last dimension), fixed T on the first and last layer of the last dimension.
function pt_dense_nd(A, dt, Δ; R = nothing)
    dims = size(A.T); N = length(dims); li = LinearIndices(dims); CI = CartesianIndices(dims)
    M = zeros(prod(dims), prod(dims)); b = zeros(prod(dims))
    unit(d) = CartesianIndex(ntuple(k -> Int(k == d), N))
    for I in CI
        r = li[I]; t = Tuple(I)
        if t[N] == 1 || t[N] == dims[N]
            M[r, r] = 1; b[r] = A.T[I]
        elseif any(d -> t[d] == 1, 1:(N - 1))
            d = findfirst(d -> t[d] == 1, 1:(N - 1)); M[r, r] = 1; M[r, li[I + unit(d)]] = -1
        elseif any(d -> t[d] == dims[d], 1:(N - 1))
            d = findfirst(d -> t[d] == dims[d], 1:(N - 1)); M[r, r] = 1; M[r, li[I - unit(d)]] = -1
        else
            C = A.Rho[I] * (A.Cp[I] + A.Hl[I] * A.dϕdT[I])
            M[r, r] = C / dt
            b[r] = (A.Rho[I] * A.Cp[I] * A.T[I] + A.Rho[I] * A.Hl[I] * (A.dϕdT[I] * A.T_it_old[I] - A.ϕ[I] + A.pt_phiold[I])) / dt + A.Hr[I]
            for d in 1:N, s in (1, -1)
                J = I + s * unit(d)
                kf = (A.Kc[I] + A.Kc[J]) / 2
                w = (R !== nothing && d == 1) ? (R[I] + R[J]) / 2 / R[I] : 1.0       # cylindrical face weight in the radial direction
                M[r, r] += w * kf / Δ[d]^2; M[r, li[J]] -= w * kf / Δ[d]^2
            end
        end
    end
    return reshape(M \ b, dims)
end

# Dense backward-Euler system for the same discretisation (zero flux on the lateral faces, fixed T top and bottom)
function pt_dense_solution(A, dt, Δ)
    Nx, Nz = size(A.T); idx(i, j) = i + (j - 1) * Nx
    M = zeros(Nx * Nz, Nx * Nz); b = zeros(Nx * Nz)
    for j in 1:Nz, i in 1:Nx
        r = idx(i, j)
        if j == 1 || j == Nz
            M[r, r] = 1; b[r] = A.T[i, j]
        elseif i == 1
            M[r, r] = 1; M[r, idx(2, j)] = -1
        elseif i == Nx
            M[r, r] = 1; M[r, idx(Nx - 1, j)] = -1
        else
            C = A.Rho[i, j] * (A.Cp[i, j] + A.Hl[i, j] * A.dϕdT[i, j])
            M[r, r] = C / dt
            b[r] = (A.Rho[i, j] * A.Cp[i, j] * A.T[i, j] + A.Rho[i, j] * A.Hl[i, j] * (A.dϕdT[i, j] * A.T_it_old[i, j] - A.ϕ[i, j] + A.pt_phiold[i, j])) / dt + A.Hr[i, j]
            for (di, dj, h) in ((1, 0, Δ[1]), (-1, 0, Δ[1]), (0, 1, Δ[2]), (0, -1, Δ[2]))
                kf = (A.Kc[i, j] + A.Kc[i + di, j + dj]) / 2
                M[r, r] += kf / h^2; M[r, idx(i + di, j + dj)] -= kf / h^2
            end
        end
    end
    return reshape(M \ b, Nx, Nz)
end

@testset "Implicit pseudo-transient diffusion step" begin
    Nx, Nz = 12, 16; Δ = (100.0, 150.0)
    dt_explicit = min(Δ...)^2 / (3.0 / (2700.0 * 1000.0)) / 4

    @testset "agrees with a direct solve (constant and variable K, source, latent heat)" begin
        for (label, kw) in (("constant", (;)), ("variable K", (; Kfun = (i, j) -> 1.5 + 3.0 * (j > 8))), ("with source", (; Hr = 2.0e-6)))
            A = pt_arrays(Nx, Nz; kw...)
            A.T .= [300.0 + 40.0 * sin(i / 3) + 25.0 * j for i in 1:Nx, j in 1:Nz]
            if label == "with source"            # latent heat in a band; iterate and previous time level differ
                A.Hl .= 3.0e5; A.dϕdT[:, 6:10] .= 0.01; A.ϕ[:, 6:10] .= 0.3; A.pt_phiold[:, 6:10] .= 0.2
            end
            A.T_it_old .= A.T .+ (label == "with source" ? 3.0 : 0.0)
            Num = NumParam(Nx = Nx, Nz = Nz, solver = :implicit_enthalpy, apt_tol = 1.0e-10, apt_check = 5)
            for fac in (1.0, 50.0)
                dt = fac * dt_explicit
                iters = apt_diffusion_solve!(A, dt, Δ, nothing, Num)
                @test iters > 0
                @test A.Tnew ≈ pt_dense_solution(A, dt, Δ) rtol = 1.0e-7
            end
        end
    end

    @testset "steady state with a source term" begin
        Nx, Nz = 6, 81; Δ = (100.0, 100.0); K, S, H = 3.0, 4.0e-6, 8000.0     # fixed T = 0 at top and bottom
        A = pt_arrays(Nx, Nz; K = K, Hr = S)
        Num = NumParam(Nx = Nx, Nz = Nz, solver = :apt, apt_tol = 1.0e-9, apt_check = 5)
        dt = 1.0e4 * (Δ[1]^2 / (K / (2700.0 * 1000.0)) / 4)                     # 10⁴ × the explicit limit
        for step in 1:12                                                        # decays like (1/(1 + dt/τ))^steps, τ = H²/(π²κ)
            A.T_it_old .= A.T
            apt_diffusion_solve!(A, dt, Δ, nothing, Num)
            A.T .= A.Tnew
        end
        z = range(0, H, length = Nz)
        Tanal = S / (2K) .* z .* (H .- z)                                       # -K T'' = S, T(0) = T(H) = 0
        @test maximum(abs, A.T[3, :] .- Tanal) < 1.0e-3 * maximum(Tanal)
    end

    @testset "iteration count grows slower than the time step" begin
        A = pt_arrays(40, 60); Num = NumParam(Nx = 40, Nz = 60, solver = :apt, apt_tol = 1.0e-8)
        A.T .= [500.0 + 10.0 * sin(i / 5) for i in 1:40, j in 1:60]; A.T[:, [1, 60]] .= 400.0
        n_dt = Dict{Float64, Int}()
        for fac in (10.0, 1000.0)
            A.T_it_old .= A.T
            n_dt[fac] = apt_diffusion_solve!(A, fac * (100.0^2 / (3.0 / 2.7e6) / 4), (100.0, 100.0), nothing, Num)
        end
        @test n_dt[1000.0] < 100 * n_dt[10.0]      # sub-linear: 100× the time step needs far fewer than 100× the iterations
        @test n_dt[1000.0] < 1000                  # an explicit scheme needs 1000 steps for the same time
    end

    @testset "explicit and implicit steps agree for small time steps" begin
        A = pt_arrays(20, 24); Num = NumParam(Nx = 20, Nz = 24, solver = :apt, apt_tol = 1.0e-10)
        A.T .= [300.0 + 80.0 * exp(-((i - 10)^2 + (j - 12)^2) / 20) for i in 1:20, j in 1:24]
        bc_zero_flux!(A.T, 1); A.T_it_old .= A.T; Δ = (50.0, 50.0); dt = 0.005 * (50.0^2 / (3.0 / 2.7e6) / 4)
        Texp = copy(A.Tnew); diffusion_step!(Texp, A.T, A.Kc, A.Rho, A.Cp, A.Hr, A.Hl, dt, Δ, A.dϕdT)
        apt_diffusion_solve!(A, dt, Δ, nothing, Num)
        @test maximum(abs, (A.Tnew .- A.T)[2:end-1, 2:end-1] .- (Texp .- A.T)[2:end-1, 2:end-1]) < 1.0e-2 * maximum(abs, (Texp .- A.T)[2:end-1, 2:end-1])
    end

    @testset "solver option" begin
        @test NumParam().solver === :explicit                                   # the existing explicit step stays the default
        @test MagmaThermoKinematics._use_implicit(NumParam(solver = :implicit_enthalpy))
        @test MagmaThermoKinematics._use_implicit(NumParam(solver = :apt))
        @test !MagmaThermoKinematics._use_implicit(NumParam())
        @test_throws ArgumentError MagmaThermoKinematics._use_implicit(NumParam(solver = :unknown))
    end

    @testset "3D and axisymmetric: agree with a direct solve" begin
        Δ3 = (120.0, 90.0, 150.0); dt_e = minimum(Δ3)^2 / (3.0 / 2.7e6) / 6
        A = pt_arrays(5, 6, 8; Kfun = (i, j, k) -> 1.5 + 3.0 * (k > 4), Hr = 1.0e-6)
        A.T .= [300.0 + 20.0 * sin(i + j / 2) + 30.0 * k for i in 1:5, j in 1:6, k in 1:8]; A.Hl .= 2.5e5; A.dϕdT[:, :, 3:5] .= 0.012; A.ϕ[:, :, 3:5] .= 0.4; A.pt_phiold[:, :, 3:5] .= 0.3
        A.T_it_old .= A.T .+ 2.0
        Num = NumParam(Nx = 5, Ny = 6, Nz = 8, solver = :apt, apt_tol = 1.0e-10, apt_check = 5)
        for fac in (1.0, 40.0)
            apt_diffusion_solve!(A, fac * dt_e, Δ3, nothing, Num)
            @test A.Tnew ≈ pt_dense_nd(A, fac * dt_e, Δ3) rtol = 1.0e-7
        end
        # axisymmetric 2D: first dimension is the radius
        Δ2 = (100.0, 150.0); R = [100.0 * (i - 0.5) for i in 1:7, j in 1:9]
        B = pt_arrays(7, 9; Kfun = (i, j) -> 2.0 + (j > 5), Hr = 3.0e-6)
        B.T .= [400.0 + 10.0 * i + 15.0 * j for i in 1:7, j in 1:9]; B.T_it_old .= B.T
        NumB = NumParam(Nx = 7, Nz = 9, solver = :apt, apt_tol = 1.0e-10, apt_check = 5)
        apt_diffusion_solve!(B, 30.0 * 100.0^2 / (3.0 / 2.7e6) / 4, Δ2, R, NumB)
        @test B.Tnew ≈ pt_dense_nd(B, 30.0 * 100.0^2 / (3.0 / 2.7e6) / 4, Δ2; R = R) rtol = 1.0e-7
    end

    @testset "3D steady state with a source term" begin
        Nx, Ny, Nz = 4, 4, 61; Δ = (100.0, 100.0, 100.0); K, S, H = 3.0, 4.0e-6, 6000.0
        A = pt_arrays(Nx, Ny, Nz; K = K, Hr = S)
        Num = NumParam(Nx = Nx, Ny = Ny, Nz = Nz, solver = :apt, apt_tol = 1.0e-9, apt_check = 5)
        dt = 1.0e4 * (Δ[1]^2 / (K / (2700.0 * 1000.0)) / 6)
        for step in 1:12
            A.T_it_old .= A.T
            apt_diffusion_solve!(A, dt, Δ, nothing, Num)
            A.T .= A.Tnew
        end
        z = range(0, H, length = Nz); Tanal = S / (2K) .* z .* (H .- z)
        @test maximum(abs, A.T[2, 2, :] .- Tanal) < 1.0e-3 * maximum(Tanal)
    end
end

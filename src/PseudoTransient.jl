# Implicit (backward-Euler) diffusion step solved with an accelerated pseudo-transient iteration.
#
# The step solves the enthalpy balance  [E(T) - E(T_old)]/dt = ∇⋅(K ∇T) + H + s (E* - E(T))  with E = ρ (cp T + Hl ϕ(T)),
# where `s` (`Arrays.Qrelax`, 1/s, default 0) and `E*` (`Arrays.Qenth`, J/m³) define an optional relaxation term. The outer
# (Picard/Newton) loop linearises E around the current iterate T_k, which gives the linear problem A T = b with
#     A = f C - ∇⋅(K ∇·),   f = 1/dt + s,   C = ρ (cp + Hl ∂ϕ/∂T),
#     b = f ρ Hl (ϕ'(T_k) T_k - ϕ(T_k)) + ρ (cp T_old + Hl ϕ(T_old))/dt + s E* + H.
# A is symmetric positive definite.
# The system is solved with a Jacobi-preconditioned second-order (damped) pseudo-transient iteration
#     T_{k+1} = T_k + α D⁻¹ (b - A T_k) + β (T_k - T_{k-1}),
# whose parameters α, β are the Chebyshev-optimal ones for a spectrum [m, M] of D⁻¹A (D = diag(A)).
# Because D⁻¹A = I - D⁻¹(D - A) and D - A has non-negative entries, the spectrum lies in [m, 2 - m] with
# m = min_i (C_i/dt)/D_i; these safe bounds are used. The iteration is matrix-free (one stencil sweep per
# iteration, no global reductions except the occasional convergence check), so it runs on any
# KernelAbstractions backend. The number of iterations grows with the square root of the time step
# (in units of the explicit stability limit) instead of linearly.

"`true` if `Num.solver` selects the implicit enthalpy step (`:implicit_enthalpy`, alias `:apt`), `false` for the explicit step (`:explicit`, the default)."
function _use_implicit(Num)
    solver = _opt(Num, :solver, :explicit)
    solver === :explicit && return false
    solver in (:implicit_enthalpy, :apt) && return true
    throw(ArgumentError("Num.solver must be :explicit or :implicit_enthalpy, got :$(solver)"))
end

"`C[I] = f ρ (cp + Hl ∂ϕ/∂T)` with `f = 1/dt + s`: the diagonal mass term of the implicit problem."
@kernel function _apt_ceff!(C, Rho, Cp, Hl, dϕdT, Qrelax, dt)
    I = @index(Global, Cartesian)
    f = inv(dt) + Qrelax[I]
    C[I] = f * Rho[I] * (Cp[I] + Hl[I] * dϕdT[I])
end

"Right-hand side `b` of the linearised implicit problem (see the header)."
@kernel function _apt_rhs!(b, Told, Tit, Rho, Cp, Hl, dϕdT, ϕ, ϕold, H, Qrelax, Qenth, dt)
    I = @index(Global, Cartesian)
    f = inv(dt) + Qrelax[I]
    b[I] = f * Rho[I] * Hl[I] * (dϕdT[I] * Tit[I] - ϕ[I]) + Rho[I] * (Cp[I] * Told[I] + Hl[I] * ϕold[I]) / dt + Qrelax[I] * Qenth[I] + H[I]
end

"Coefficient of `T[I]` in `∇⋅(K ∇T)` along the step `e`, i.e. `-∂(∇q)/∂T[I]` for one dimension (see `_∂q`)."
@inline _apt_diag(K, I, e, Δ::Number, R) = _rinv(R, I) * (_rface(R, I, I + e) * (K[I] + K[I + e]) + _rface(R, I - e, I) * (K[I - e] + K[I])) / (2 * Δ^2)
@inline function _apt_diag(K, I, e, h::AbstractVector, R)
    i = _along(I, e)
    hp, hm = h[i], h[i - 1]
    return _rinv(R, I) * (_rface(R, I, I + e) * (K[I] + K[I + e]) / (2 * hp) + _rface(R, I - e, I) * (K[I - e] + K[I]) / (2 * hm)) / ((hp + hm) / 2)
end

@inline function _apt_diag_total(K, I, T, Δ, R)
    d = _apt_diag(K, I, _unit(1, T), Δ[1], R)
    for k in 2:ndims(T)
        d += _apt_diag(K, I, _unit(k, T), Δ[k], nothing)
    end
    return d
end

"`ratio[I] = Cdt / D` on interior cells (`Cdt = C[I]` the mass term, `D = Cdt + Σ face conductances/Δ²`), the cell-wise lower bound of the spectrum of D⁻¹A."
@kernel function _apt_ratio!(ratio, C, K, T, Δ, R)
    I0 = @index(Global, Cartesian)
    I = I0 + oneunit(I0)
    Cdt = C[I]
    ratio[I] = Cdt / (Cdt + _apt_diag_total(K, I, T, Δ, R))
end

"One pseudo-transient (heavy-ball) update of the interior cells; also stores the preconditioned residual `D⁻¹(b - A T_k)` in `res`."
@kernel function _apt_update!(Tn, Tk, Tkm1, b, C, K, res, Δ, R, α, β)
    I0 = @index(Global, Cartesian)
    I = I0 + oneunit(I0)
    ∇q = _∂q(Tk, K, I, _unit(1, Tk), Δ[1], R)
    for d in 2:ndims(Tk)
        ∇q += _∂q(Tk, K, I, _unit(d, Tk), Δ[d], nothing)
    end
    Cdt = C[I]
    r = b[I] - Cdt * Tk[I] + ∇q                           # b - A T_k
    D = Cdt + _apt_diag_total(K, I, Tk, Δ, R)
    z = r / D
    res[I] = z
    Tn[I] = Tk[I] + α * z + β * (Tk[I] - Tkm1[I])
end

"Interior index ranges of `A` (all but the outermost cell layer in every dimension)."
@inline _apt_inner(A) = ntuple(d -> 2:(size(A, d) - 1), Val(ndims(A)))

"""
    apt_diffusion_solve!(Arrays, dt, Δ, R, Num)

Solve one implicit diffusion step (grid spacing `Δ` as for [`diffusion_step!`](@ref)) for the frozen properties in `Arrays` (`Kc`, `Rho`, `Cp`, `Hl`, `dϕdT`, `Hr`)
starting from the guess `Arrays.T_it_old` (the properties are those at this iterate, with melt fraction `Arrays.ϕ`), with `Arrays.T` and `Arrays.pt_phiold` the temperature and melt fraction of the previous time level, and store the result in `Arrays.Tnew`.
Boundary conditions are those of the explicit step: zero flux on the lateral faces, isothermal (`T` of the previous
time level) or a prescribed flux on the bottom.

Returns the number of pseudo-transient iterations used. Throws an error if `Num.apt_tol` (relative to `max(|T|, 1)`) is not
reached within `Num.apt_max_iter` iterations.
"""
function apt_diffusion_solve!(Arrays, dt, Δ, R, Num)
    FT = eltype(Arrays.T)
    N = ndims(Arrays.T)
    C, res, b = Arrays.pt_C, Arrays.pt_res, Arrays.pt_b
    Tk, Tkm1, Tn = Arrays.pt_Tk, Arrays.pt_Tkm1, Arrays.pt_Tn
    inner = _apt_inner(Arrays.T)
    Δbot = _bottom_spacing(Δ[N])                   # on the host, for the bottom flux condition

    _launch!(_apt_ceff!, Arrays.T, size(Arrays.T), C, Arrays.Rho, Arrays.Cp, Arrays.Hl, Arrays.dϕdT, Arrays.Qrelax, dt)
    _launch!(_apt_rhs!, Arrays.T, size(Arrays.T), b, Arrays.T, Arrays.T_it_old, Arrays.Rho, Arrays.Cp, Arrays.Hl, Arrays.dϕdT, Arrays.ϕ, Arrays.pt_phiold, Arrays.Hr, Arrays.Qrelax, Arrays.Qenth, dt)

    # spectrum bounds of D⁻¹A (see the header) and the Chebyshev parameters
    _launch!(_apt_ratio!, Arrays.T, size(Arrays.T) .- 2, res, C, Arrays.Kc, Arrays.T, Δ, R)
    m = max(FT(minimum(view(res, inner...))), eps(FT))
    M = FT(2) - m
    α = FT(4) / (sqrt(M) + sqrt(m))^2
    β = ((sqrt(M) - sqrt(m)) / (sqrt(M) + sqrt(m)))^2
    res .= 0

    # initial guess with consistent boundary values
    Tk .= Arrays.T_it_old
    for d in 1:(N - 1)
        bc_zero_flux!(Tk, d)
    end
    Tkm1 .= Tk
    Tn .= Tk
    tol = FT(Num.apt_tol) * max(maximum(abs, Tk), one(FT))      # relative to the largest |T| (at least 1 K)
    βk = zero(FT)                                  # plain preconditioned Richardson step for the first iteration
    err, iter = Inf, 0
    while iter < Num.apt_max_iter
        iter += 1
        _launch!(_apt_update!, Tk, size(Tk) .- 2, Tn, Tk, Tkm1, b, C, Arrays.Kc, res, Δ, R, α, βk)
        for d in 1:(N - 1)
            bc_zero_flux!(Tn, d)
        end
        if Num.flux_bottom_BC
            bc_z_bottom_flux!(Tn, Arrays.Kc, Δbot, FT(Num.flux_bottom))
        else
            bc_T!(Tn, Arrays.T)
        end
        Tkm1, Tk, Tn = Tk, Tn, Tkm1                # rotate the three work arrays
        βk = β
        if mod(iter, Num.apt_check) == 0
            err = maximum(abs, res) / m           # estimate of the remaining temperature error, from the smallest eigenvalue
            err <= tol && break
        end
    end
    err <= tol || error("$(N)D pseudo-transient iterations did not converge after $(iter) iterations (error estimate=$(err), tolerance=$(tol)); increase Num.apt_max_iter or reduce Δt")
    Arrays.Tnew .= Tk
    return iter
end

# Nonlinear heat diffusion, written once for 2D and 3D. Kernels run on the
# KernelAbstractions backend of the arrays they receive, and the dimension
# follows `ndims` of the arrays. The vertical direction is always the last one.

"""
    Numeric_params

Parameters that control the nonlinear diffusion solver.
"""
@with_kw struct Numeric_params
    ω::Float64 = 0.5             # relaxation parameter for nonlinear iterations
    max_iter::Int64 = 1500            # max. number of nonlinear iterations
    verbose::Bool = false           # print info?
    convergence::Float64 = 1.0e-4            # nonlinear convergence criteria
    axisymmetric::Bool = false           # Axisymmetric or 2D?
    flux_bottom_BC::Bool = false           # Flux bottom BC?
    flux_bottom::Float64 = 0.0             # flux @ bottom, in case flux_bottom_BC=true
    deactivate_La_at_depth::Bool = false           # no latent heat and melt below `deactivationDepth`?
    deactivationDepth::Float64 = -15.0e3           # depth [m] below which latent heat and melt are switched off
end

"Launch the KernelAbstractions kernel `kernel!` over `ndrange` on the backend of `A`."
function _launch!(kernel!, A, ndrange, args...)
    kernel!(get_backend(A))(args...; ndrange)
    return nothing
end

"""
    GridArray!(X, Z, Grid)
    GridArray!(X, Y, Z, Grid)

Fill coordinate arrays from `Grid`: `X` and `Y` with the horizontal
coordinates, `Z` with the vertical one.
"""
GridArray!(X::AbstractArray, Z::AbstractArray, Grid::GridData) = (_coordinate!(X, Grid, 1); _coordinate!(Z, Grid, 2))
GridArray!(X::AbstractArray, Y::AbstractArray, Z::AbstractArray, Grid::GridData) =
    (_coordinate!(X, Grid, 1); _coordinate!(Y, Grid, 2); _coordinate!(Z, Grid, 3))

"Fill `A` with the grid coordinate along dimension `d`."
function _coordinate!(A, Grid, d)
    c = Grid.coord1D[d]
    cA = similar(A, length(c))
    copyto!(cA, collect(eltype(A), c))
    A .= reshape(cA, ntuple(i -> i == d ? length(c) : 1, ndims(A)))
    return nothing
end

"""
    diffusion_step!(Tnew, T, K, Rho, Cp, H, Hl, dt, Δ, dϕdT; R=nothing)

Explicit update of the interior cells of `Tnew` for
`ρ (cp + Hl ∂ϕ/∂T) ∂T/∂t = ∇⋅(K ∇T) + H`, with face conductivities the mean of
the two neighboring cells. `Δ` holds the grid spacing per dimension. With the
cell radii `R`, the first dimension is radial (2D axisymmetric).
"""
function diffusion_step!(Tnew, T, K, Rho, Cp, H, Hl, dt, Δ, dϕdT; R = nothing)
    axes(Tnew) == axes(T) || throw(DimensionMismatch("Tnew and T must match: $(axes(Tnew)) vs $(axes(T))"))
    _launch!(_diffusion_step!, T, size(T) .- 2, Tnew, T, K, Rho, Cp, H, Hl, dt, Tuple(Δ), dϕdT, R)
    return nothing
end

@kernel function _diffusion_step!(Tnew, T, K, Rho, Cp, H, Hl, dt, Δ, dϕdT, R)
    I0 = @index(Global, Cartesian)
    I = I0 + oneunit(I0)                    # interior cells only
    ∇q = _∂q(T, K, I, _unit(1, T), Δ[1], R)
    for d in 2:ndims(T)
        ∇q += _∂q(T, K, I, _unit(d, T), Δ[d], nothing)
    end
    Tnew[I] = T[I] + dt / (Rho[I] * (Cp[I] + Hl[I] * dϕdT[I])) * (∇q + H[I])
end

"Unit step along dimension `d` of `A`."
@inline _unit(d, A) = CartesianIndex(ntuple(i -> Int(i == d), Val(ndims(A))))

"Contribution of the step `e` to `∇⋅(K ∇T)`; with radii `R`, the cylindrical (1/r) ∂(r K ∂T/∂r)/∂r."
@inline function _∂q(T, K, I, e, Δ, ::Nothing)
    qp = (K[I] + K[I + e]) / 2 * (T[I + e] - T[I]) / Δ
    qm = (K[I - e] + K[I]) / 2 * (T[I] - T[I - e]) / Δ
    return (qp - qm) / Δ
end
@inline function _∂q(T, K, I, e, Δ, R)
    qp = (R[I] + R[I + e]) / 2 * ((K[I] + K[I + e]) / 2) * (T[I + e] - T[I]) / Δ
    qm = (R[I - e] + R[I]) / 2 * ((K[I - e] + K[I]) / 2) * (T[I] - T[I - e]) / Δ
    return inv(R[I]) * (qp - qm) / Δ
end

"""
    bc_zero_flux!(T, d)

Zero-flux boundaries on both faces of `T` normal to dimension `d`.
"""
function bc_zero_flux!(T, d)
    Base.require_one_based_indexing(T)
    n = size(T, d)
    selectdim(T, d, 1) .= selectdim(T, d, 2)
    selectdim(T, d, n) .= selectdim(T, d, n - 1)
    return nothing
end

"""
    bc_T!(Tnew, T)

Isothermal top and bottom: `Tnew` takes the values of `T` on both vertical faces.
"""
function bc_T!(Tnew, T)
    Base.require_one_based_indexing(Tnew, T)
    N, n = ndims(T), size(T, ndims(T))
    selectdim(Tnew, N, 1) .= selectdim(T, N, 1)
    selectdim(Tnew, N, n) .= selectdim(T, N, n)
    return nothing
end

"""
    bc_z_bottom_flux!(T, K, dz, q_z)

Heat flux `q_z` through the bottom face.
"""
function bc_z_bottom_flux!(T, K, dz, q_z)
    Base.require_one_based_indexing(T, K)
    N = ndims(T)
    selectdim(T, N, 1) .= selectdim(T, N, 2) .+ q_z * dz ./ selectdim(K, N, 1)
    return nothing
end

"""
    _squared_l2_distance(A, B)

`Σ (A[i] - B[i])²`, without materializing the difference: one fused kernel on
GPU arrays, a linear-index loop on CPU (`vec` avoids the slow Cartesian iteration).
"""
_squared_l2_distance(A, B) = sum(abs2, Broadcast.instantiate(Broadcast.broadcasted(-, vec(A), vec(B))))

"""
    compute_phase_param!(A, fn, MatParam::Tuple, Phases, args)

Set every cell of `A` to `fn` (a GeoParams `compute_…` function, e.g.
`compute_density`) evaluated with the material of that cell's phase. `args` is a
NamedTuple of arrays shaped like `A` (e.g. `(; T, P)`), read at each cell.
`Phases` may hold any integer type (e.g. `Int32` on GPUs).
"""
function compute_phase_param!(A, fn::F, MatParam::Tuple, Phases, args::NamedTuple) where {F}
    eltype(Phases) <: Integer || throw(ArgumentError("Phases must hold integer phase numbers, got $(eltype(Phases))"))
    _launch!(_compute_phase_param!, A, size(A), A, fn, MatParam, Phases, args)
    return nothing
end

@kernel function _compute_phase_param!(A, fn::F, MatParam, Phases, args) where {F}
    I = @index(Global, Cartesian)
    argsI = NamedTuple{keys(args)}(map(a -> a[I], values(args)))
    A[I] = fn(MatParam, Int64(Phases[I]), argsI)   # GeoParams' phase lookup takes an Int64
end

"""
    Nonlinear_Diffusion_step!(Arrays, Mat_tup, Phases, Grid, dt, Num = Numeric_params())

Performs a single, nonlinear, diffusion step during which temperature dependent
properties (density, heat capacity, conductivity) are updated. The dimension
follows the arrays; with `Num.axisymmetric` a 2D model is axisymmetric, with
the cell radii in `Arrays.R`.

Throws an error if the Picard iterations do not converge within `Num.max_iter`.
"""
function Nonlinear_Diffusion_step!(Arrays, Mat_tup::Tuple, Phases, Grid, dt, Num = Numeric_params())
    # Scalars are converted to the state's element type, so a Float32 state
    # runs Float32 arithmetic in every kernel.
    FT = eltype(Arrays.T)
    dt, T₀, ω, flux_bottom = FT(dt), FT(273.15), FT(Num.ω), FT(Num.flux_bottom)
    Δ = FT.(Tuple(Grid.Δ))
    N = ndims(Arrays.T)
    R = Num.axisymmetric ? Arrays.R : nothing

    @. Arrays.T_K = Arrays.T + T₀
    Arrays.T_it_old .= Arrays.T
    args1 = haskey(Arrays, :index) ? (; T = Arrays.T_K, P = Arrays.P, index = Arrays.index) : (; T = Arrays.T_K, P = Arrays.P)
    compute_phase_param!(Arrays.Hr, compute_radioactive_heat, Mat_tup, Phases, (; z = -Arrays.Z))   # independent of T
    err, iter = 1.0, 1
    while err > Num.convergence && iter < Num.max_iter
        compute_phase_param!(Arrays.ϕ, compute_meltfraction, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.dϕdT, compute_dϕdT, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Rho, compute_density, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Cp, compute_heatcapacity, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Kc, compute_conductivity, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Hl, compute_latent_heat, Mat_tup, Phases, args1)

        if Num.deactivate_La_at_depth       # no latent heat and melt below `deactivationDepth`
            minZ = FT(Num.deactivationDepth)
            @. Arrays.dϕdT = ifelse(Arrays.Z < minZ, zero(FT), Arrays.dϕdT)
            @. Arrays.ϕ = ifelse(Arrays.Z < minZ, zero(FT), Arrays.ϕ)
        end

        diffusion_step!(Arrays.Tnew, Arrays.T, Arrays.Kc, Arrays.Rho, Arrays.Cp, Arrays.Hr, Arrays.Hl, dt, Δ, Arrays.dϕdT; R)
        for d in 1:(N - 1)                      # flux-free lateral boundaries
            bc_zero_flux!(Arrays.Tnew, d)
        end
        if Num.flux_bottom_BC
            bc_z_bottom_flux!(Arrays.Tnew, Arrays.Kc, Δ[N], flux_bottom)
        else
            bc_T!(Arrays.Tnew, Arrays.T)    # isothermal top and bottom
        end

        # Relaxed Picard iteration for the T used by the (nonlinear) material properties
        @. Arrays.Tupdate = ω * Arrays.Tnew + (one(FT) - ω) * Arrays.T_it_old
        @. Arrays.T_K = Arrays.Tupdate + T₀     # all GeoParams routines expect T in K

        err = sqrt(_squared_l2_distance(Arrays.Tnew, Arrays.T_it_old)) / maximum(Arrays.Tnew)
        Num.verbose && println("  Nonlinear iteration $(iter), error=$(err)")
        Arrays.T_it_old .= Arrays.Tupdate
        iter += 1
    end
    (isfinite(err) && err <= Num.convergence) ||
        error("$(N)D nonlinear diffusion did not converge after $(iter - 1) iterations (error=$(err), tolerance=$(Num.convergence)); reduce Δt or the relaxation parameter Num.ω=$(Num.ω) [0-1]")
    Num.verbose && println("  ----")
    return nothing
end

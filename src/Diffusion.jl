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
    sticky_air_BC::Bool = false            # hold the air phase at `Tsurface_Celsius` (see `bc_sticky_air!`)
    AirPhase::Int64 = 0                # phase number of the air
    Tsurface_Celsius::Float64 = 0.0               # surface temperature [C]
    convection::Bool = false               # parameterised convection of melt-rich regions (see `compute_convective_nusselt!`); needs the implicit solver
    convection_phi::Float64 = 0.5          # melt fraction above which a region convects
    convection_Nu_max::Float64 = 100.0
    convection_coeffs::NTuple{3, Float64} = (0.0087, 0.135, -0.2943)
    convection_Ra_crit::Float64 = 1708.0
    convection_alpha::Float64 = 3.0e-5
    convection_viscosity::Any = nothing    # GeoParams creep law of the melt-rich regions (`nothing`: `DEFAULT_CONVECTION_VISCOSITY`)
    convection_strainrate::Float64 = 1.0e-12
    convection_boundary_layer::Bool = true  # the outermost cells of a convecting region keep Nu = 1 (conductive boundary layer)
    convection_z_split::Any = ()           # depths [m] separating chambers of different viscosity law (`convection_viscosity` a tuple of laws)
    out_of_plane::Bool = false             # planar 2D: add the out-of-plane heat loss (see `add_out_of_plane_loss!`)
    out_of_plane_factor::Float64 = 1.5
    out_of_plane_smooth::Int64 = 10
    out_of_plane_Lmin::Float64 = 1.0e3
end

_sticky_air(Num) = hasproperty(Num, :sticky_air_BC) ? Num.sticky_air_BC : false       # lightweight parameter structs have no `sticky_air_BC` field
_convection(Num) = hasproperty(Num, :convection) ? Num.convection : false        # lightweight parameter structs have no `convection` field

_out_of_plane(Num) = hasproperty(Num, :out_of_plane) ? Num.out_of_plane : false        # lightweight parameter structs have no `out_of_plane` field

"Launch the KernelAbstractions kernel `kernel!` over `ndrange` on the backend of `A`."
function _launch!(kernel!, A, ndrange, args...)
    kernel!(get_backend(A))(args...; ndrange)
    return nothing
end

"`v` as a vector with the element type and on the backend of `A` (`v` itself if it already is)."
function _on_backend(A, v::AbstractVector)
    (get_backend(v) == get_backend(A) && eltype(v) == eltype(A)) && return v
    b = similar(A, eltype(A), length(v))
    copyto!(b, eltype(A).(Array(v)))
    return b
end
_on_backend(A, v::AbstractRange) = v

"""
    _kernel_spacing(Grid, A)

Grid spacing per direction for the kernels, in the element type of `A`: numbers for a grid with constant spacing,
otherwise vectors with the distances between neighbouring points (`h[i] = x[i+1] - x[i]`) on the backend of `A`
for every direction.
"""
function _kernel_spacing(Grid, A)
    FT = eltype(A)
    Grid.ConstantΔ && return FT.(Tuple(Grid.Δ))
    return map(c -> _on_backend(A, FT.(diff(c))), Tuple(Grid.coord1D))
end

"""
    _cell_weights(Grid, A)

Widths of the grid cells per direction on the backend of `A`, shaped for broadcasting against `A` (their product is the
cell volume), or `nothing` for a grid with constant spacing.
"""
function _cell_weights(Grid, A)
    Grid.ConstantΔ && return nothing
    N = length(Grid.N)
    return ntuple(d -> reshape(_on_backend(A, cell_widths(Grid.coord1D[d])), ntuple(k -> k == d ? Grid.N[d] : 1, N)), N)
end

# spacing to the next point of index `i`: a number for constant spacing, `h[i]` otherwise
@inline _hp(h::Number, i) = h
@inline _hp(h::AbstractVector, i) = h[i]

# width of the control volume of point `k` of `n` (half the distance to both neighbours; the outermost points take the distance to their only neighbour)
@inline _cw(h::Number, k, n) = h
@inline _cw(h::AbstractVector, k, n) = k == 1 ? h[1] : k == n ? h[n - 1] : (h[k - 1] + h[k]) / 2

"Distance between the two lowest points (the spacing used by the bottom flux condition), on the host."
_bottom_spacing(h::Number) = h
_bottom_spacing(h::AbstractVector) = Array(view(h, 1:1))[1]

"Index of the Cartesian index `I` along the unit step `e`."
@inline _along(I, e) = sum(Tuple(I) .* Tuple(e))

"""
    lithostatic_pressure!(P, Rho, g, Δz)

Fill `P` with the lithostatic pressure in Pa (the unit GeoParams' `compute_*`
laws and phase-diagram lookups expect) for the density `Rho` [kg/m³], gravity
`g` [m/s²], and vertical spacing `Δz` [m]: a number, or for a refined grid the
vector of the distances between neighbouring points (`diff(z)`). The vertical
direction is the last one; `P` is zero at its last index (the surface) and
increases downward by the trapezoid rule.
"""
function lithostatic_pressure!(P, Rho, g, Δz)
    axes(P) == axes(Rho) || throw(DimensionMismatch("P and Rho must match: $(axes(P)) vs $(axes(Rho))"))
    gΔz = _scaled_spacing(P, g, Δz)
    _launch!(_lithostatic_pressure!, P, size(P)[1:(end - 1)], P, Rho, gΔz)
    return nothing
end

_scaled_spacing(P, g, Δz::Number) = eltype(P)(g) * eltype(P)(Δz)
function _scaled_spacing(P, g, Δz::AbstractVector)
    length(Δz) == size(P, ndims(P)) - 1 || throw(DimensionMismatch("Δz must hold the $(size(P, ndims(P)) - 1) vertical distances between the points, got $(length(Δz))"))
    return eltype(P)(g) .* _on_backend(P, Δz)
end

@kernel function _lithostatic_pressure!(P, Rho, gΔz)
    I = @index(Global, NTuple)
    n = size(P, ndims(P))
    P[I..., n] = zero(eltype(P))
    for k in (n - 1):-1:1
        P[I..., k] = P[I..., k + 1] + _hp(gΔz, k) * (Rho[I..., k] + Rho[I..., k + 1]) / 2
    end
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
the two neighboring cells. `Δ` holds the grid spacing per dimension: numbers, or
for a refined grid vectors with the distances between neighbouring points (on the
backend of `T`, see `_kernel_spacing`). With the cell radii `R`, the first
dimension is radial (2D axisymmetric).
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
# refined grid: the fluxes through the faces midway between the points, divided by the width of the control volume
@inline function _∂q(T, K, I, e, h::AbstractVector, ::Nothing)
    i = _along(I, e)
    hp, hm = h[i], h[i - 1]
    qp = (K[I] + K[I + e]) / 2 * (T[I + e] - T[I]) / hp
    qm = (K[I - e] + K[I]) / 2 * (T[I] - T[I - e]) / hm
    return (qp - qm) / ((hp + hm) / 2)
end
@inline function _∂q(T, K, I, e, h::AbstractVector, R)
    i = _along(I, e)
    hp, hm = h[i], h[i - 1]
    qp = (R[I] + R[I + e]) / 2 * ((K[I] + K[I + e]) / 2) * (T[I + e] - T[I]) / hp
    qm = (R[I - e] + R[I]) / 2 * ((K[I - e] + K[I]) / 2) * (T[I] - T[I - e]) / hm
    return inv(R[I]) * (qp - qm) / ((hp + hm) / 2)
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
    bc_sticky_air!(T, Phases, AirPhase, Tair)

Hold every cell of phase `AirPhase` at the temperature `Tair`.

The air above the topography is a numerical device that keeps the ground surface
at a prescribed temperature. Emulating that with a very high air conductivity
makes the air the fastest-diffusing phase of the model, and since the time step
is set by the largest diffusivity anywhere (`κ_time`), the whole domain is then
advanced with a time step that is one to three orders of magnitude smaller than
the rock requires. Prescribing `T` in the air instead imposes the same surface
condition directly and leaves `κ_time` to the rocks.

`Phases` is the current phase array, so the condition follows the surface when
sills inflate the model and displace the host rock.
"""
function bc_sticky_air!(T, Phases, AirPhase, Tair)
    _launch!(_bc_sticky_air!, T, size(T), T, Phases, AirPhase, Tair)
    return nothing
end

@kernel function _bc_sticky_air!(T, @Const(Phases), AirPhase, Tair)
    I = @index(Global, Cartesian)
    @inbounds if Phases[I] == AirPhase
        T[I] = Tair
    end
end

"""
    bc_z_bottom_flux!(T, K, dz, q_z)

Heat flux `q_z` through the bottom face; `dz` is the distance between the two lowest points.
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

# GeoParams' radioactive heat takes the depth `z` (positive downward); the model coordinate `Z` is negative below the surface.
radioactive_heat_at_Z(MatParam, phase, args) = compute_radioactive_heat(MatParam, phase, (; z = -args.Z))

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
    Δ = _kernel_spacing(Grid, Arrays.T)      # numbers, or vectors of the distances between the points of a refined grid
    N = ndims(Arrays.T)
    Δbot = _bottom_spacing(Δ[N])
    R = Num.axisymmetric ? Arrays.R : nothing
    g = compute_gravity(Mat_tup[1])

    @. Arrays.T_K = Arrays.T + T₀
    Arrays.T_it_old .= Arrays.T
    args1 = haskey(Arrays, :index) ? (; T = Arrays.T_K, P = Arrays.P, index = Arrays.index) : (; T = Arrays.T_K, P = Arrays.P)
    compute_phase_param!(Arrays.Hr, radioactive_heat_at_Z, Mat_tup, Phases, (; Z = Arrays.Z))   # independent of T
    haskey(Arrays, :Qsrc) && (Arrays.Hr .+= Arrays.Qsrc)         # volumetric source term [W/m³], set by the user
    if _convection(Num)                           # Nusselt numbers of the melt-rich regions from the state at the beginning of the step
        _use_implicit(Num) || throw(ArgumentError("convection needs the implicit solver (solver = :implicit_enthalpy): the enhanced conductivity is far above the explicit stability limit"))
        haskey(Arrays, :Nu) || throw(ArgumentError("convection needs the array `Arrays.Nu` (created by `MTK_GeoParams`)"))
        compute_phase_param!(Arrays.ϕ, compute_meltfraction, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Rho, compute_density, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Cp, compute_heatcapacity, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Kc, compute_conductivity, Mat_tup, Phases, args1)
        compute_convective_nusselt!(
            Arrays.Nu, Arrays.Kc, Arrays.T_K, Arrays.ϕ, Arrays.Rho, Arrays.Cp, Phases, Arrays.Z, Δ[N], something(Num.convection_viscosity, DEFAULT_CONVECTION_VISCOSITY); z_split = Num.convection_z_split, g, ϕ_crit = Num.convection_phi,
            α = Num.convection_alpha, coeffs = Num.convection_coeffs, Ra_crit = Num.convection_Ra_crit, Nu_max = Num.convection_Nu_max, strainrate = Num.convection_strainrate, boundary_layer = Num.convection_boundary_layer
        )
    end
    if _out_of_plane(Num)                                         # out-of-plane heat loss of a 2D section (lagged: T and Kc of the previous step)
        (N == 2 && !Num.axisymmetric) || throw(ArgumentError("out_of_plane is available for planar 2D models only"))
        haskey(Arrays, :Tbg) || throw(ArgumentError("out_of_plane needs the background temperature `Arrays.Tbg`"))
        widths = Grid.ConstantΔ ? (;) : (; dx = cell_widths(Grid.coord1D[1]), dz = cell_widths(Grid.coord1D[2]))     # weights of the moments on a refined grid
        add_out_of_plane_loss!(Arrays.Hr, Arrays.T, Arrays.Tbg, Arrays.Kc, Arrays.R, Phases; factor = Num.out_of_plane_factor, smooth = Num.out_of_plane_smooth, Lmin = Num.out_of_plane_Lmin, air_phase = hasproperty(Num, :AirPhase) ? Num.AirPhase : 0, widths...)
    end
    err, iter, apt_iter_total = 1.0, 1, 0
    while err > Num.convergence && iter < Num.max_iter
        compute_phase_param!(Arrays.ϕ, compute_meltfraction, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.dϕdT, compute_dϕdT, Mat_tup, Phases, args1)
        iter == 1 && haskey(Arrays, :pt_phiold) && (Arrays.pt_phiold .= Arrays.ϕ)     # melt fraction at the previous time level (the first iterate is T)
        compute_phase_param!(Arrays.Rho, compute_density, Mat_tup, Phases, args1)
        lithostatic_pressure!(Arrays.P, Arrays.Rho, g, Δ[N])
        compute_phase_param!(Arrays.Cp, compute_heatcapacity, Mat_tup, Phases, args1)
        compute_phase_param!(Arrays.Kc, compute_conductivity, Mat_tup, Phases, args1)
        _convection(Num) && (Arrays.Kc .*= Arrays.Nu)         # enhanced conductivity of the melt-rich regions (Nusselt numbers of the beginning of the step)
        compute_phase_param!(Arrays.Hl, compute_latent_heat, Mat_tup, Phases, args1)

        if Num.deactivate_La_at_depth       # no latent heat and melt below `deactivationDepth`
            minZ = FT(Num.deactivationDepth)
            @. Arrays.dϕdT = ifelse(Arrays.Z < minZ, zero(FT), Arrays.dϕdT)
            @. Arrays.ϕ = ifelse(Arrays.Z < minZ, zero(FT), Arrays.ϕ)
        end

        if _use_implicit(Num)
            n_apt = apt_diffusion_solve!(Arrays, dt, Δ, R, Num)    # implicit step, includes the boundary conditions
            apt_iter_total += n_apt
            Num.verbose && println("  pseudo-transient iterations: $(n_apt)")
        else
            Hheat = Arrays.Hr
            if haskey(Arrays, :Qrelax)               # relaxation term Qrelax (Qenth - E(T_k)), lagged at the iterate
                @. Arrays.Hq = Arrays.Hr + Arrays.Qrelax * (Arrays.Qenth - Arrays.Rho * (Arrays.Cp * Arrays.T_it_old + Arrays.Hl * Arrays.ϕ))
                Hheat = Arrays.Hq
            end
            diffusion_step!(Arrays.Tnew, Arrays.T, Arrays.Kc, Arrays.Rho, Arrays.Cp, Hheat, Arrays.Hl, dt, Δ, Arrays.dϕdT; R)
            for d in 1:(N - 1)                      # flux-free lateral boundaries
                bc_zero_flux!(Arrays.Tnew, d)
            end
            if Num.flux_bottom_BC
                bc_z_bottom_flux!(Arrays.Tnew, Arrays.Kc, Δbot, flux_bottom)
            else
                bc_T!(Arrays.Tnew, Arrays.T)    # isothermal top and bottom
            end
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
    _sticky_air(Num) && bc_sticky_air!(Arrays.Tnew, Phases, Num.AirPhase, FT(Num.Tsurface_Celsius))     # once per time step
    Num.verbose && println("  ----  $(iter - 1) nonlinear iterations, $(apt_iter_total) pseudo-transient iterations in total")
    return nothing
end

"""
Magma-chamber overpressure and eruption model for 2D and 3D simulations.

A chamber is the largest connected region of the grid with melt fraction `ϕ ≥ ϕ_erupt`.
Its pressure follows a lumped three-phase (melt + crystals + exsolved H₂O) mass balance
with host-rock elasticity and viscous shell relaxation; once the overpressure exceeds a
critical value and a dike can reach the surface, the chamber drains back to a relaxed
pressure. The magma volume lost (or gained, for injection) is turned into a host-rock
displacement field by summing Mogi point sources, one per grid cell, using the same
point-source solution as `InjectSills`.

The pressure model works on an equivalent column: the chamber has a mean thickness
`h = V/W` and a lateral radius `R`, where `V` is its volume and `W` its footprint (the
width in 2D, the projected area in 3D).
"""

# ---------------------------------------------------------------------------------------
# Parameters and state
# ---------------------------------------------------------------------------------------
"""
    EruptionParams(; kwargs...)

Parameters of the chamber/eruption model. Pressures in Pa, lengths in m, times in s.

- `ϕ_erupt`: minimum melt fraction of a mobile chamber cell
- `h_melt_min`: minimum melt thickness (`∫ϕ dz` equivalent) for a region to count as a chamber [m]
- `ΔP_crit`, `ΔP_relax`: overpressure at which the chamber drains, and the one it relaxes to
- `ϕ_g_crit`: gas volume fraction above which the chamber is gas-locked and cannot erupt
- `σ_barrier`: stress barrier that the dike has to overcome [Pa]
- `w_dike`, `κ_magma`: dike half-width [m] and magma thermal diffusivity [m²/s] (dike freezing)
- `μ_shear`: host-rock shear modulus [Pa]; `ν`: Poisson ratio used for the displacement field
- `A_visc`, `G_act`: Arrhenius constants of the crustal creep law
- `m_w`: total H₂O mass fraction of the magma
- `ρ_melt`, `ρ_x`, `crust`: densities as `(ρ0, α, β)` (reference density, expansivity, compressibility)
- `melt_viscosity`, `solubility`, `ρ_gas`: GeoParams laws
- `g`: gravity
"""
Base.@kwdef mutable struct EruptionParams
    ϕ_erupt::Float64 = 0.5
    h_melt_min::Float64 = 500.0
    ΔP_crit::Float64 = 20.0e6
    ϕ_g_crit::Float64 = 0.5
    ΔP_relax::Float64 = 0.0
    σ_barrier::Float64 = 10.0e6
    w_dike::Float64 = 2.0
    κ_magma::Float64 = 1.0e-6
    μ_shear::Float64 = 1.0e10
    ν::Float64 = 0.25
    A_visc::Float64 = 4.25e7
    G_act::Float64 = 141.0e3
    m_w::Float64 = 0.0
    ρ_melt::NTuple{3, Float64} = (2400.0, 3.0e-5, 1.0e-10)
    ρ_x::NTuple{3, Float64} = (2700.0, 3.0e-5, 1.5e-11)
    crust::NTuple{3, Float64} = (2700.0, 3.0e-5, 1.5e-11)
    melt_viscosity::Any = LinearMeltViscosity()
    solubility::Any = Liu2005_Solubility()
    ρ_gas::Any = RedlichKwong_Density()
    g::Float64 = 9.81
end

"State of one chamber: pressure, volume and mass inventories per unit footprint area."
Base.@kwdef mutable struct EruptionState
    P::Float64 = 0.0
    P_lith::Float64 = 0.0
    V::Float64 = 0.0            # chamber thickness [m]
    V_grid_prev::Float64 = 0.0
    M::Float64 = 0.0            # [kg/m²]
    M_H2O::Float64 = 0.0
    M_initial::Float64 = 0.0
    M_in::Float64 = 0.0
    M_out::Float64 = 0.0
    M_boundary::Float64 = 0.0
    mass_residual::Float64 = 0.0
    ϕ_g::Float64 = 0.0
    T_prev::Float64 = NaN
    ϕ_prev::Float64 = NaN
    h_erupt::Float64 = 0.0      # bulk thickness drained during the last step [m]
    h_pending::Float64 = 0.0
    pending_since::Float64 = NaN
    ρ_magma::Float64 = 0.0
    η_magma::Float64 = 0.0
    η_r::Float64 = 0.0
    id::Int = 0
    z_lo::Float64 = NaN
    z_hi::Float64 = NaN
    cells::Any = nothing        # cells of the chamber at the last step (Vector{CartesianIndex})
    init::Bool = false
end

"Reset every field except the generation counter `id`."
function reset_chamber!(state::EruptionState)
    fresh = EruptionState()
    for f in fieldnames(EruptionState)
        f === :id || setfield!(state, f, getfield(fresh, f))
    end
    return state
end

# ---------------------------------------------------------------------------------------
# Densities, H₂O partitioning, compliance, viscosity
# ---------------------------------------------------------------------------------------
_pt_density(c, P, T_K) = c[1] * (1 - c[2] * (T_K - 273.15) + c[3] * P)
melt_density(ep::EruptionParams, P, T_K) = _pt_density(ep.ρ_melt, P, T_K)
crystal_density(ep::EruptionParams, P, T_K) = _pt_density(ep.ρ_x, P, T_K)
crust_density(ep::EruptionParams, P, T_K) = _pt_density(ep.crust, P, T_K)

condensed_density(ep, P, T_K, ϕ) = ϕ * melt_density(ep, P, T_K) + (1 - ϕ) * crystal_density(ep, P, T_K)

"""
    water_gas_partition(P, T_K, ϕ_melt, ep; m_w) -> (m_diss, X_g, ρ_g)

Dissolved (per magma mass) and exsolved H₂O mass fractions and the gas density.
"""
function water_gas_partition(P, T_K, ϕ_melt, ep::EruptionParams; m_w = ep.m_w)
    P >= 0 || throw(DomainError(P, "magma pressure must be nonnegative"))
    m_eq = first(compute_dissolved(ep.solubility, (; P, T = T_K, X_co2 = 0.0)))
    ρ_m = melt_density(ep, P, T_K)
    x_melt = ϕ_melt * ρ_m / (ϕ_melt * ρ_m + (1 - ϕ_melt) * crystal_density(ep, P, T_K))
    m_diss = clamp(m_eq * x_melt, zero(m_eq), m_w)
    X_g = m_w - m_diss
    ρ_g = X_g > 0 ? compute_density(ep.ρ_gas, (; P, T = T_K)) : zero(P)
    X_g > 0 && !(ρ_g > 0) && throw(DomainError(ρ_g, "gas density law returned a nonpositive density"))
    return m_diss, X_g, ρ_g
end

"""
    mixture_density(P, T_K, ϕ_melt, ep; m_w) -> (ρ, ϕ_g)

Three-phase mixture density and gas volume fraction.
"""
function mixture_density(P, T_K, ϕ_melt, ep::EruptionParams; m_w = ep.m_w)
    ρ_c = condensed_density(ep, P, T_K, ϕ_melt)
    _, Xg, ρ_g = water_gas_partition(P, T_K, ϕ_melt, ep; m_w)
    Xg > 0 || return ρ_c, zero(ρ_c)
    Vg = Xg / ρ_g
    Vc = (1 - Xg) / ρ_c
    return 1 / (Vg + Vc), Vg / (Vg + Vc)
end

# 1/β_m = (1/ρ) ∂ρ/∂P at fixed T, ϕ (central difference)
function inv_magma_compressibility(P, T_K, ϕ, ep, m_w, ρ)
    δ = max(1.0e-6 * P, 1.0e3)
    ρp = first(mixture_density(P + δ, T_K, ϕ, ep; m_w))
    ρm = first(mixture_density(max(P - δ, 0.0), T_K, ϕ, ep; m_w))
    return (ρp - ρm) / (P + δ - max(P - δ, 0.0)) / ρ
end

"Host-rock compliance 1/β_r of a sill-like chamber of thickness `V` and radius `R` (spherical limit at aspect ratio 1)."
host_compliance(ep::EruptionParams, V, R) = 3 / (4 * ep.μ_shear * min(0.5 * V / R, 1.0))

crustal_viscosity(ep::EruptionParams, T_K) = ep.A_visc * exp(ep.G_act / (8.314 * T_K))

"""
    crustal_relaxation_viscosity(ep, T_mush_K, T_far_K; shell_ratio=11, n=64) -> η_r

Effective viscosity of the visco-elastic shell around the chamber (steady spherical conduction
profile between the chamber wall and the far field, Simpson integration), clamped to 1e17–1e24 Pa s.
"""
function crustal_relaxation_viscosity(ep::EruptionParams, T_mush_K, T_far_K; shell_ratio = 11.0, n = 64)
    T_far_K = min(T_far_K, T_mush_K)
    T_shell(s) = (T_mush_K * (shell_ratio - s) + shell_ratio * T_far_K * (s - 1)) / ((shell_ratio - 1) * s)
    integrand(s) = 3 * crustal_viscosity(ep, T_shell(s)) / s^4
    h = (shell_ratio - 1) / n
    total = integrand(1.0) + integrand(shell_ratio)
    for i in 1:(n - 1)
        total += (isodd(i) ? 4 : 2) * integrand(1 + i * h)
    end
    return clamp(total * h / 3, 1.0e17, 1.0e24)
end

# ---------------------------------------------------------------------------------------
# Dike ascent
# ---------------------------------------------------------------------------------------
"Distance a dike travels before freezing, positive root of `L² − Δρg·c·L − ΔP·c = 0`, `c = w⁴/(3ηκ)`."
function max_ascent_length(ΔP, Δρg, η, ep::EruptionParams)
    c = ep.w_dike^4 / (3 * η * ep.κ_magma)
    b = max(Δρg, zero(Δρg)) * c
    return 0.5 * (b + sqrt(b^2 + 4 * max(ΔP, zero(ΔP)) * c))
end

"""
    dike_ascends(ΔP, P_lith, ρ_magma, η_magma, depth, ep) -> Bool

A dike from a chamber at `depth` [m, positive] reaches the surface if the magma pressure surviving
the ascent exceeds `σ_barrier` and the chamber is not deeper than the dike's freezing length.
"""
function dike_ascends(ΔP, P_lith, ρ_magma, η_magma, depth, ep::EruptionParams)
    ΔP + P_lith - ρ_magma * ep.g * depth >= ep.σ_barrier || return false
    depth > 0 || return true
    return depth <= max_ascent_length(ΔP, (P_lith - ρ_magma * ep.g * depth) / depth, η_magma, ep)
end

# ---------------------------------------------------------------------------------------
# Overpressure ODE
# ---------------------------------------------------------------------------------------
function chamber_water_fraction(state::EruptionState, fallback)
    return iszero(state.M) ? fallback : state.M_H2O / state.M
end

# Newton-project (P, V) onto ρ(P) V(P) = M with V = V₀ exp((P-P₀)/β_r)
function project_mass_constraint!(state, ep, T_K, ϕ, m_w, inv_βr)
    P0, V0 = state.P, state.V
    P = P0
    res_prev = Inf
    ΔP = 0.0
    for it in 1:60
        V = V0 * exp((P - P0) * inv_βr)
        ρ = first(mixture_density(P, T_K, ϕ, ep; m_w))
        residual = ρ * V - state.M
        if abs(residual) <= 64eps(max(state.M, ρ * V, 1.0))
            state.P, state.V, state.mass_residual = P, V, residual
            return ρ
        end
        if abs(residual) > abs(res_prev) && it > 1     # overshoot: halve the previous step and retry
            P -= 0.5ΔP
            ΔP *= 0.5
            continue
        end
        res_prev = residual
        compliance = inv_βr + inv_magma_compressibility(P, T_K, ϕ, ep, m_w, ρ)
        compliance > 0 || throw(DomainError(compliance, "mass-constraint compliance must be positive"))
        ΔP = -residual / (ρ * V * compliance)
        P + ΔP > 0 || (ΔP = -0.5P)
        P += ΔP
    end
    throw(ErrorException("chamber mass projection did not converge in 60 Newton iterations"))
end

"""
    pending_withdrawal!(state, h_requested, h_available, Δ; time) -> h_realizable

Queue drained bulk thickness; return zero while the queued amount is at most `2Δ`, else the
resolvable thickness bounded by `h_available`. Debit it with [`commit_pending_withdrawal!`](@ref).
"""
function pending_withdrawal!(state::EruptionState, h_requested, h_available, Δ; time = NaN)
    if h_requested > 0 && state.h_pending == 0
        state.pending_since = time
    end
    state.h_pending += h_requested
    h_realizable = min(state.h_pending, h_available)
    return h_realizable > 2Δ ? h_realizable : 0.0
end

function commit_pending_withdrawal!(state::EruptionState, h_realized)
    state.h_pending -= h_realized
    if state.h_pending <= 64eps(max(h_realized, 1.0))
        state.h_pending = 0.0
        state.pending_since = NaN
    end
    return state
end

"""
    step_overpressure!(state, ep, T_mush_K, ϕ_mush, V_e, R, ȧ, Δt; depth, η_r)

Advance the chamber over one thermal step `Δt` [s]. `V_e` is the mean chamber thickness on the
grid [m], `R` its lateral radius [m] and `ȧ` the recharge rate per footprint area [m/s]. The
bulk thickness drained by eruptions during the step is returned in `state.h_erupt`.
"""
function step_overpressure!(state::EruptionState, ep::EruptionParams, T_mush_K, ϕ_mush, V_e, R, ȧ, Δt; depth, η_r)
    m_w = chamber_water_fraction(state, ep.m_w)
    ρ, ϕg = mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w)
    state.ϕ_g = ϕg
    state.ρ_magma = ρ
    state.η_magma = compute_viscosity_εII(ep.melt_viscosity, 1.0, (; T = T_mush_K))
    state.h_erupt = 0.0
    state.η_r = η_r
    if !state.init || V_e <= 0
        state.T_prev, state.ϕ_prev = T_mush_K, ϕ_mush
        state.V = state.V_grid_prev = V_e
        state.M = ρ * V_e
        state.M_H2O = ep.m_w * state.M
        state.M_initial = state.M
        state.M_in = state.M_out = state.M_boundary = state.mass_residual = 0.0
        state.init = true
        return state
    end

    # the grid contributes only its change (melting/freezing at the margins)
    ΔV_grid = V_e - state.V_grid_prev
    ΔM = ρ * ΔV_grid
    state.V += ΔV_grid
    state.M += ΔM
    state.M_H2O += m_w * ΔM
    state.M_boundary += ΔM
    state.V_grid_prev = V_e
    state.V > 0 || throw(DomainError(state.V, "chamber volume collapsed to zero or below"))

    inv_βm = inv_magma_compressibility(state.P, T_mush_K, ϕ_mush, ep, m_w, ρ)
    ρ_old = first(mixture_density(state.P, state.T_prev, state.ϕ_prev, ep; m_w))
    Ṁ_in = ȧ * melt_density(ep, state.P, T_mush_K)
    S = Ṁ_in / (ρ * state.V) - (ρ - ρ_old) / Δt / ρ
    invβ = host_compliance(ep, state.V, R) + inv_βm
    τ = η_r * invβ
    ΔP_eq = S * η_r
    ΔP_0 = state.P - state.P_lith
    excursion = abs(ΔP_eq - ΔP_0) * (-expm1(-Δt / τ))
    nsub = round(Int, clamp(ceil(excursion / (0.1 * ep.ΔP_crit)), 1.0, 10_000.0))
    dt_sub = Δt / nsub

    h_out = 0.0
    for _ in 1:nsub
        m_w = chamber_water_fraction(state, ep.m_w)
        ρ_sub = first(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w))
        inv_βm_sub = inv_magma_compressibility(state.P, T_mush_K, ϕ_mush, ep, m_w, ρ_sub)
        ρ_old_sub = first(mixture_density(state.P, state.T_prev, state.ϕ_prev, ep; m_w))
        Ṁ_in_sub = ȧ * melt_density(ep, state.P, T_mush_K)
        S_sub = Ṁ_in_sub / (ρ_sub * state.V) - (ρ_sub - ρ_old_sub) / Δt / ρ_sub
        inv_βr = host_compliance(ep, state.V, R)
        τ_sub = η_r * (inv_βr + inv_βm_sub)
        ΔP_eq_sub = S_sub * η_r
        decay = exp(-dt_sub / τ_sub)

        M_in = Ṁ_in_sub * dt_sub
        state.M += M_in
        state.M_H2O += ep.m_w * M_in
        state.M_in += M_in
        ΔP_start = state.P - state.P_lith
        ΔP = ΔP_eq_sub + (ΔP_start - ΔP_eq_sub) * decay
        ∫ΔP_dt = ΔP_eq_sub * dt_sub + (ΔP_start - ΔP_eq_sub) * τ_sub * (1 - decay)
        state.P = state.P_lith + ΔP
        state.V *= exp((ΔP - ΔP_start) * inv_βr + ∫ΔP_dt / η_r)
        ρ_sub = project_mass_constraint!(state, ep, T_mush_K, ϕ_mush, m_w, inv_βr)
        ΔP = state.P - state.P_lith
        ϕg_sub = last(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w))
        if ϕg_sub < ep.ϕ_g_crit && ΔP >= ep.ΔP_crit && dike_ascends(ΔP, state.P_lith, ρ_sub, state.η_magma, depth, ep)
            M_before = state.M
            state.P = state.P_lith + ep.ΔP_relax
            state.V *= exp((ep.ΔP_relax - ΔP) * inv_βr)
            M_after = first(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w)) * state.V
            M_out = M_before - M_after
            M_out >= 0 || throw(DomainError(M_out, "pressure relaxation would add mass during an eruption"))
            h_out += M_out / ρ_sub
            state.M = M_after
            state.M_H2O -= m_w * M_out
            state.M_out += M_out
        end
    end

    m_w = chamber_water_fraction(state, ep.m_w)
    ρ = project_mass_constraint!(state, ep, T_mush_K, ϕ_mush, m_w, host_compliance(ep, state.V, R))
    state.ρ_magma = ρ
    state.ϕ_g = last(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w))
    state.h_erupt = h_out
    state.T_prev, state.ϕ_prev = T_mush_K, ϕ_mush
    return state
end

# ---------------------------------------------------------------------------------------
# Chamber detection on a 2D or 3D grid
# ---------------------------------------------------------------------------------------
"""
    chamber_cells(ϕ, ϕ_erupt) -> Vector{CartesianIndex{N}}

Cells of the largest face-connected region with `ϕ ≥ ϕ_erupt` (by number of cells). Empty if no cell
qualifies; NaN counts as solid. A host `Array` is searched breadth-first; any other array (GPU) is
labelled on its own backend with [`chamber_labels!`](@ref) and only the labels are copied to the host.
"""
function chamber_cells(ϕ::Array{<:Real, N}, ϕ_erupt) where {N}
    mask = ϕ .>= ϕ_erupt
    seen = falses(size(ϕ))
    R = CartesianIndices(ϕ)
    offs = [CartesianIndex(ntuple(k -> k == d ? s : 0, N)) for d in 1:N for s in (-1, 1)]
    best = CartesianIndex{N}[]
    queue = CartesianIndex{N}[]
    for I0 in R
        (mask[I0] && !seen[I0]) || continue
        empty!(queue); push!(queue, I0); seen[I0] = true
        head = 1
        while head <= length(queue)
            I = queue[head]; head += 1
            for o in offs
                J = I + o
                (J in R && mask[J] && !seen[J]) || continue
                seen[J] = true; push!(queue, J)
            end
        end
        length(queue) > length(best) && (best = copy(queue))
    end
    return best
end

function chamber_cells(ϕ::AbstractArray{<:Real, N}, ϕ_erupt) where {N}
    lab = chamber_labels!(similar(ϕ, Int32), ϕ, ϕ_erupt)
    return largest_label_cells(Array(lab))
end

"Cells of the most frequent nonzero label (host array of labels)."
function largest_label_cells(lab::Array{Int32, N}) where {N}
    counts = Dict{Int32, Int}()
    for v in lab
        v == 0 || (counts[v] = get(counts, v, 0) + 1)
    end
    isempty(counts) && return CartesianIndex{N}[]
    best = argmax(counts)
    return findall(==(best), lab)
end

@kernel function _label_init!(lab, ϕ, ϕ_erupt)
    I = @index(Global, Cartesian)
    i = @index(Global, Linear)
    lab[I] = ϕ[I] >= ϕ_erupt ? Int32(i) : Int32(0)
end

@kernel function _label_hook!(lab, ::Val{N}) where {N}
    I = @index(Global, Cartesian)
    l = lab[I]
    if l != 0
        for d in 1:N, s in (-1, 1)
            J = CartesianIndex(ntuple(k -> k == d ? I[k] + s : I[k], Val(N)))
            if checkbounds(Bool, lab, J)
                lj = lab[J]
                (lj != 0 && lj < l) && (l = lj)
            end
        end
        lab[I] = l
    end
end

@kernel function _label_jump!(lab)
    i = @index(Global, Linear)
    l = lab[i]
    if l != 0
        r = lab[l]                      # label of the cell the label points to (labels are linear indices)
        while r != 0 && r != l
            l = r; r = lab[l]
        end
        lab[i] = l
    end
end

"""
    chamber_labels!(lab, ϕ, ϕ_erupt; max_iter=10_000) -> lab

Label the face-connected regions with `ϕ ≥ ϕ_erupt` on the backend of `ϕ`: every cell of a region ends
up with the smallest linear index of that region, other cells get 0 (`lab` is an `Int32` array of the
size of `ϕ`). Iterates neighbour-minimum hooking and pointer jumping until no label changes.
"""
function chamber_labels!(lab, ϕ::AbstractArray{<:Real, N}, ϕ_erupt; max_iter = 10_000) where {N}
    backend = get_backend(ϕ)
    _label_init!(backend)(lab, ϕ, eltype(ϕ)(ϕ_erupt); ndrange = size(ϕ))
    prev = similar(lab)
    for it in 1:max_iter
        copyto!(prev, lab)
        _label_hook!(backend)(lab, Val(N); ndrange = size(ϕ))
        _label_jump!(backend)(lab; ndrange = length(ϕ))
        KernelAbstractions.synchronize(backend)
        prev == lab && return lab
    end
    throw(ErrorException("chamber labelling did not converge in $max_iter iterations"))
end

"""
    chamber_geometry(cells, Δ, coords) -> (; V, W, h, R, scale, zc, z_lo, z_hi)

Equivalent column of a chamber: native volume `V` (m² in 2D, m³ in 3D), footprint `W` (m in 2D, m²
in 3D), mean thickness `h = V/W`, lateral radius `R`, the factor `scale` converting native volume to
the 3D-equivalent volume (`π R² h / V`), the volume-weighted centroid depth coordinate `zc` and the
vertical extent. `Δ` are the cell sizes and `coords` the 1D coordinate vectors (z last).
"""
function chamber_geometry(cells::Vector{CartesianIndex{N}}, Δ, coords) where {N}
    dV = prod(Δ)
    V = length(cells) * dV
    zc = sum(I -> coords[N][I[N]], cells) / length(cells)
    z_lo = minimum(I -> coords[N][I[N]], cells) - Δ[N] / 2
    z_hi = maximum(I -> coords[N][I[N]], cells) + Δ[N] / 2
    if N == 2
        cols = Set(I[1] for I in cells)
        W = length(cols) * Δ[1]
        R = W / 2
    else
        cols = Set((I[1], I[2]) for I in cells)
        W = length(cols) * Δ[1] * Δ[2]
        R = sqrt(W / π)
    end
    h = V / W
    scale = π * R^2 * h / V
    return (; V, W, h, R, scale, zc, z_lo, z_hi)
end

# ---------------------------------------------------------------------------------------
# Host-rock displacement from summed Mogi point sources
# ---------------------------------------------------------------------------------------
@kernel function _mogi_sum!(U, X, centers, strength, L, soft2, ::Val{N}) where {N}
    I = @index(Global, Cartesian)
    FT = eltype(centers)
    px = X[1][I]; pz = X[N][I]; py = N == 3 ? X[2][I] : zero(FT)
    ux = zero(FT); uy = zero(FT); uz = zero(FT)
    for s in 1:size(centers, 1)
        dx = px - centers[s, 1]; dz = pz - centers[s, N]; dy = N == 3 ? py - centers[s, 2] : zero(FT)
        r2 = soft2 + dx * dx + dy * dy + dz * dz
        Ls = L[s]
        # point source d/|d|³, or the same source spread along the out-of-plane direction over [-L, L]
        c = Ls > 0 ? strength[s] * 2 * Ls / (r2 * sqrt(r2 + Ls * Ls)) : strength[s] / (r2 * sqrt(r2))
        ux += c * dx; uy += c * dy; uz += c * dz
    end
    U[1][I] = ux
    U[N][I] = uz
    if N == 3
        U[2][I] = uy
    end
end

"""
    mogi_displacement!(U, X, centers, ΔV; L=zeros, ν=0.25, softening)

Displacement `U` (tuple of arrays, one per direction) at the node coordinates `X` (tuple of arrays)
due to Mogi point sources at `centers` (`Ns × N`) with volume changes `ΔV` (`Ns`; positive = inflation,
negative = withdrawal). Each source contributes `(1−ν) ΔV/π · d/|d|³`, which is the
`InjectSills.MogiSphere` solution written in terms of its volume change `ΔV = π ΔP r³/G`; the contributions
are summed (linear elasticity). In a 2D section a source with `L[s] > 0` is instead a line of point sources
along the out-of-plane direction over `[-L, L]`, with `ΔV` the volume change per unit length: the
integrated kernel is `2L · d/(|d|² √(|d|² + L²))`. The sum runs on the backend of `X[1]`; `softening` [m]
regularises the singularity at the sources' own nodes.
"""
function mogi_displacement!(U, X, centers, ΔV; L = zeros(length(ΔV)), ν = 0.25, softening)
    backend = get_backend(X[1])
    FT = eltype(X[1])
    c = KernelAbstractions.allocate(backend, FT, size(centers)); copyto!(c, FT.(centers))
    s = KernelAbstractions.allocate(backend, FT, length(ΔV)); copyto!(s, FT.((1 - ν) / π .* ΔV))
    l = KernelAbstractions.allocate(backend, FT, length(ΔV)); copyto!(l, FT.(L))
    _mogi_sum!(backend)(U, X, c, s, l, FT(softening)^2, Val(length(U)); ndrange = size(X[1]))
    KernelAbstractions.synchronize(backend)
    return U
end

"""
    cell_sources(cells, dV_native, factor, coords; coarsen=1) -> (centers, ΔV)

Point sources for a set of cells: one per cell, volume `factor · dV_native` each (a list of 3D-equivalent
volume changes can be formed by scaling `factor` with [`chamber_geometry`](@ref)'s `scale` and the
fractional volume change). With `coarsen > 1`, cells are binned in blocks of `coarsen` cells per direction
and each bin becomes one source at its volume-weighted centroid (cheaper for large 3D chambers).
"""
function cell_sources(cells::Vector{CartesianIndex{N}}, dV_native, factor, coords; coarsen = 1) where {N}
    if coarsen == 1
        centers = [coords[k][I[k]] for I in cells, k in 1:N]
        return centers, fill(factor * dV_native, length(cells))
    end
    bins = Dict{NTuple{N, Int}, Tuple{Vector{Float64}, Int}}()
    for I in cells
        key = ntuple(k -> (I[k] - 1) ÷ coarsen, Val(N))
        acc, n = get!(bins, key, (zeros(N), 0))
        for k in 1:N
            acc[k] += coords[k][I[k]]
        end
        bins[key] = (acc, n + 1)
    end
    centers = zeros(length(bins), N); ΔV = zeros(length(bins))
    for (i, (_, (acc, n))) in enumerate(bins)
        centers[i, :] .= acc ./ n
        ΔV[i] = factor * dV_native * n
    end
    return centers, ΔV
end

"""
    displace_temperature!(T, Tbuf, coords, U; maxfrac=0.5) -> nsub

Advect the temperature field `T` with the displacement field `U` [m] (semi-Lagrangian, in as many
sub-steps as needed to keep each below `maxfrac` of a cell). `Tbuf` is a work array of the same size.
"""
function displace_temperature!(T, Tbuf, coords, U; maxfrac = 0.5)
    Δmin = minimum(step.(Tuple(coords)))
    umax = maximum(maximum(abs, u) for u in U)
    nsub = max(1, ceil(Int, umax / (maxfrac * Δmin)))
    V = map(u -> u ./ nsub, Tuple(U))
    src, dst = T, Tbuf
    for _ in 1:nsub
        AdvectTemperature!(dst, src, Tuple(coords), V, 1.0)
        src, dst = dst, src
    end
    src === T || copyto!(T, src)
    return nsub
end

# ---------------------------------------------------------------------------------------
# One chamber step on the grid
# ---------------------------------------------------------------------------------------
"""
    chamber_step!(state, ep, ϕ, T, coords; recharge, Δt, time, T_background, margin=0.0) -> event

Advance the chamber of the arrays `ϕ` (melt fraction) and `T` [°C] (host or device) by `Δt` [s]. `coords` are the
1D coordinate vectors (`z` last, negative downward, free surface at `z = 0`), `recharge` the magma
volume injected into the chamber per second in native units (m²/s in 2D, m³/s in 3D) and
`T_background(depth)` the undisturbed geotherm [°C] at depth [m].

A drained thickness is queued until it exceeds `resolve` cells (the grid-resolvable withdrawal; the
displacement field is continuous, so values below 1 are admissible).

Returns `nothing` when no eruption is realised, otherwise a named tuple with the bulk thickness
`h_erupt`, the fractional volume change `f = h_erupt/h` of the chamber, the chamber `cells` and its
geometry `geo`. Remove the volume with [`erupt_sources`](@ref) and [`mogi_displacement!`](@ref).
"""
function chamber_step!(state::EruptionState, ep::EruptionParams, ϕ, T, coords; recharge, Δt, time, T_background, margin = 0.0, resolve = 2.0)
    N = ndims(ϕ)
    Δ = ntuple(k -> step(coords[k]), N)
    if ϕ isa Array
        cells = chamber_cells(ϕ, ep.ϕ_erupt)
        inmask = nothing
    else
        lab = chamber_labels!(similar(ϕ, Int32), ϕ, ep.ϕ_erupt)
        cells = largest_label_cells(Array(lab))
        inmask = isempty(cells) ? nothing : (lab .== Int32(LinearIndices(ϕ)[first(cells)]))
    end
    if isempty(cells)
        reset_chamber!(state)
        return nothing
    end
    geo = chamber_geometry(cells, Δ, coords)
    # sums over the chamber: on the host through the cell list, on the device through the label mask
    ϕsum = inmask === nothing ? sum(I -> ϕ[I], cells) : sum(ϕ .* inmask)
    h_melt = ϕsum * prod(Δ) / geo.W
    if h_melt < ep.h_melt_min
        reset_chamber!(state)
        return nothing
    end
    state.init && (geo.z_lo > state.z_hi || geo.z_hi < state.z_lo) && reset_chamber!(state)
    state.init || (state.id += 1)
    state.z_lo, state.z_hi = geo.z_lo, geo.z_hi
    state.cells = cells

    depth = -geo.zc
    P_lith = _lithostatic(ep, depth, T_background)
    if state.P_lith == 0.0 && state.P == 0.0
        state.P = P_lith
    else
        state.P += P_lith - state.P_lith
    end
    state.P_lith = P_lith

    T_mush_K = (inmask === nothing ? sum(I -> T[I], cells) : sum(T .* inmask)) / length(cells) + 273.15
    ϕ_mush = ϕsum / length(cells)
    η_r = crustal_relaxation_viscosity(ep, T_mush_K, T_background(depth) + 273.15)
    step_overpressure!(state, ep, T_mush_K, ϕ_mush, geo.h, geo.R, recharge / geo.W, Δt; depth, η_r)

    Δz = Δ[N]
    h_drained = state.h_erupt
    h_erupt = pending_withdrawal!(state, h_drained, geo.h, resolve * Δz / 2; time)
    near_boundary = (geo.z_lo - margin <= coords[N][1]) || (geo.z_hi + margin >= coords[N][end])
    (h_erupt > 0 && !near_boundary) || return nothing
    commit_pending_withdrawal!(state, h_erupt)
    return (; h_erupt, f = h_erupt / geo.h, cells, geo, chamber = state.id, trigger_time = state.pending_since, time)
end

# lithostatic pressure at `depth`: integrate the crustal density law over the geotherm
function _lithostatic(ep::EruptionParams, depth, T_background; n = 200)
    P = 0.0
    dz = depth / n
    for i in 1:n
        P += crust_density(ep, P, T_background((i - 0.5) * dz) + 273.15) * ep.g * dz
    end
    return P
end

"""
    erupt_sources(event, coords; coarsen=1) -> (centers, ΔV, L)

Mogi sources of an eruption event: every chamber cell loses the fraction `event.f` of its volume. In 3D
these are point sources with the cell volumes (negative). In 2D each cell is a line source over the
chamber's out-of-plane half-extent `L(x) = √(R² − (x − x_c)²)` (circular footprint of radius `R` around
the chamber centre), with the cell area as volume per unit length, so the sources integrate to the 3D
volume `f · π R² h`. Pass to [`mogi_displacement!`](@ref).
"""
function erupt_sources(event, coords; coarsen = 1)
    N = length(coords)
    Δ = ntuple(k -> step(coords[k]), N)
    centers, ΔV = cell_sources(event.cells, prod(Δ), -event.f, coords; coarsen)
    N == 3 && return centers, ΔV, zeros(length(ΔV))
    xc = sum(I -> coords[1][I[1]], event.cells) / length(event.cells)
    R = event.geo.R
    L = [sqrt(max(R^2 - (centers[s, 1] - xc)^2, 0.0)) for s in axes(centers, 1)]
    return centers, ΔV, L
end

# ---------------------------------------------------------------------------------------
# Apply an eruption to the grid and tracers
# ---------------------------------------------------------------------------------------
"""
    erupt!(T, Tbuf, Tracers, coords, event; ν=0.25, coarsen=1, softening=min(Δ), rng) -> (U, cargo)

Realise the eruption `event` of [`chamber_step!`](@ref): a Mogi sink of volume `f·dV` in every chamber
cell (3D-equivalent volumes) gives the host-rock displacement `U`, with which the temperature `T`
(any backend; `Tbuf` is a work array) and the tracers are advected. Tracers inside the chamber are
removed with probability `f` and returned as `cargo`. `softening` regularises the point sources.
"""
function erupt!(T, Tbuf, Tracers, coords, event; ν = 0.25, coarsen = 1, softening = nothing, rng = Random.default_rng())
    N = length(coords)
    Δ = ntuple(k -> step(coords[k]), N)
    softening === nothing && (softening = minimum(Δ))
    centers, ΔV, L = erupt_sources(event, coords; coarsen)
    backend = get_backend(T)
    X = ntuple(N) do k
        A = KernelAbstractions.allocate(backend, eltype(T), size(T))
        A .= reshape(coords[k], ntuple(j -> j == k ? length(coords[k]) : 1, N))
        A
    end
    U = map(similar, X)
    mogi_displacement!(U, X, centers, ΔV; L, ν, softening)
    displace_temperature!(T, Tbuf, coords, U)

    cargo = Tracers[Int[]]
    if isassigned(Tracers, 1)
        inside = Set(event.cells)
        sel = Int[]
        for (j, c) in enumerate(Tracers.coord)
            I = CartesianIndex(ntuple(k -> clamp(round(Int, (c[k] - first(coords[k])) / Δ[k]) + 1, 1, length(coords[k])), N))
            I in inside && rand(rng) < event.f && push!(sel, j)
        end
        cargo = Tracers[sel]
        deleteat!(Tracers, sel)
        isassigned(Tracers, 1) && AdvectTracers!(Tracers, Tuple(coords), map(Array, U), 1.0)
    end
    return U, cargo
end

# ---------------------------------------------------------------------------------------
# Free surface
# ---------------------------------------------------------------------------------------
"""
    move_free_surface!(Phases, Phases_init, T, Zsurf, coords, U; air=0, fill=1) -> Zsurf

Move the free surface (the boundary between the `air` phase and the rock below it) with the
displacement field `U` [m] (tuple of arrays, vertical component last). `Zsurf` holds the surface
elevation of every vertical column (size of the horizontal grid; `NaN` entries are initialised from the
phases, half a cell above the highest rock cell), and is shifted by the vertical displacement at the
surface, so displacements smaller than a cell accumulate. Cells above the surface become `air`,
cells below it that were air become `fill`; `Phases_init` is updated the same way so that the
initial-phase reset after injections keeps the moved surface, and `T` of cells turned into air takes
the air temperature above them. All arrays may live on any backend.
"""
function move_free_surface!(Phases, Phases_init, T, Zsurf, coords, U; air = 0, fill = 1)
    N = length(coords)
    z = coords[N]
    Δz = step(z)
    Ph = Array(Phases); Pi = Array(Phases_init); Th = Array(T); Uz = Array(U[N]); Zs = Array(Zsurf)
    nz = length(z)
    cols = CartesianIndices(ntuple(k -> axes(Ph, k), N - 1))
    for c in cols
        col(k) = CartesianIndex(Tuple(c)..., k)
        if isnan(Zs[c])
            ks = findfirst(k -> Ph[col(k)] == air, 1:nz)              # first air cell from below
            Zs[c] = ks === nothing ? z[end] + Δz / 2 : z[ks] - Δz / 2
        end
        zs = Zs[c]
        ks = clamp(round(Int, (zs - z[1]) / Δz) + 1, 1, nz)        # cell containing the surface
        u = Uz[col(ks)]
        zs_new = zs + u
        Zs[c] = zs_new
        for k in nz:-1:1                                            # top down, so a cell turned into air sees air above it
            I = col(k)
            above = z[k] > zs_new
            if above && Ph[I] != air
                Ph[I] = air; Pi[I] = air
                Th[I] = Th[col(min(k + 1, nz))]
            elseif !above && Ph[I] == air
                Ph[I] = fill; Pi[I] = fill
            end
        end
    end
    copyto!(Phases, Ph); copyto!(Phases_init, Pi); copyto!(T, Th); copyto!(Zsurf, Zs)
    return Zsurf
end

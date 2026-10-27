# Parameterised convection of melt-rich regions (GeoParams viscosity laws, `Nu`-`Ra` scaling).
#
# A body of low-crystallinity magma convects vigorously, which keeps its interior close to isothermal and moves heat to its boundaries much faster than
# conduction does. This is mimicked by an enhanced ("hyper-") conductivity `K_eff = Nu K` in every connected melt-rich region, with the Nusselt number
# from the Rayleigh number of the region, `Ra = ρ g α ΔT L³ / (κ η)`, with the high-Prandtl-number fit `log10 Nu = a₂ log10(Ra)² + a₁ log10(Ra) + a₀` of Grossmann & Lohse (2001), which
# Gerbault et al. (2025, Solid Earth 16, 1307) adopted for magma chambers and checked with their own CFD calculations, and `Nu = 1` below the critical Rayleigh number 1708:
#   - the region is a vertical run of cells with melt fraction ϕ ≥ ϕ_c in a column (not air); `L` is its thickness and `ΔT` the temperature contrast across it,
#   - `κ = K/(ρ cp)` and `ρ` are averaged over the run, `η` is the viscosity of a crystal-bearing melt at the mean temperature and melt fraction of the run:
#     a GeoParams viscosity law (default: Giordano et al. 2008 rhyolitic melt with the Costa et al. 2009 crystal correction).
# `Nu` is evaluated once per time step from the state at the beginning of the step (lagged), which keeps the nonlinear iteration of the step free of the on/off switching of the regions.
# `Nu` is bounded by `Nu_max`: once the interior is homogenised within a time step, a larger value changes nothing but makes the implicit solve more expensive.

"""
    convective_nusselt(ΔT, L, ρ, κ, η; g = 9.81, α = 3.0e-5, coeffs = (0.0087, 0.1350, -0.2943), Ra_crit = 1708.0, Nu_max = 100.0) -> Nu

Nusselt number of a convecting layer of thickness `L` [m] with temperature contrast `ΔT` [K], density `ρ`, thermal diffusivity `κ` [m²/s] and viscosity `η` [Pa s], with the
Rayleigh number `Ra = ρ g α ΔT L³/(κ η)`: `Nu = 1` below `Ra_crit`, and above it
`log10(Nu) = a₂ log10(Ra)² + a₁ log10(Ra) + a₀`, `coeffs = (a₂, a₁, a₀)`, bounded by `Nu_max`. The default coefficients are the fit to the experiments and simulations of
Grossmann & Lohse (2001) for Prandtl numbers above 10⁶ (where `Nu` no longer depends on the Prandtl number), which Gerbault et al. (2025, Solid Earth 16, 1307)
adopted for magma chambers and checked with their own CFD calculations;
the critical Rayleigh number is that of a layer with rigid boundaries (1708).
"""
function convective_nusselt(ΔT, L, ρ, κ, η; g = 9.81, α = 3.0e-5, coeffs = (0.0087, 0.135, -0.2943), Ra_crit = 1708.0, Nu_max = 100.0)
    Ra = ρ * oftype(ρ, g) * oftype(ρ, α) * max(ΔT, zero(ΔT)) * L^3 / (κ * η)
    Ra < oftype(Ra, Ra_crit) && return one(Ra)
    x = log10(Ra)
    Nu = exp10(oftype(Ra, coeffs[1]) * x^2 + oftype(Ra, coeffs[2]) * x + oftype(Ra, coeffs[3]))
    return min(max(one(Ra), Nu), oftype(Ra, Nu_max))
end

"""
Default viscosity law of the convecting regions: the hydrous rhyolitic melt of Giordano et al. (2008) (72 wt % SiO₂, 6 wt % H₂O) with the crystal correction of
Costa et al. (2009), i.e. a GeoParams creep law that is evaluated with the temperature and melt fraction of the region.
"""
const DEFAULT_CONVECTION_VISCOSITY = ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = (72.0, 0.3, 14.0, 2.0, 0.4, 1.5, 3.5, 3.5, 6.0)))

"Effective viscosity [Pa s] of the GeoParams creep law `law` at temperature `T` [K], melt fraction `ϕ` and the reference strain rate `ε` [1/s] (secant viscosity `τ/(2ε)`)."
@inline function convective_viscosity(law, T, ϕ, ε)
    return compute_τII(law, ε; T, ϕ) / (2 * ε)
end

@kernel function _convection_columns!(Nu_out, Kc, T, ϕ, Rho, Cp, Phases, Z, visc, splits, nz, ϕc, g, α, coeffs, Ra_crit, Nu_max, ε, Δz)
    I = @index(Global, Cartesian)
    FT = eltype(Kc)
    k = 1
    for kk in 1:nz
        Nu_out[I, kk] = one(FT)
    end
    while k <= nz
        if _is_melt(ϕ, Phases, I, k, ϕc)
            k2 = k
            while k2 < nz && _is_melt(ϕ, Phases, I, k2 + 1, ϕc)
                k2 += 1
            end
            n = k2 - k + 1
            Tmin = T[I, k]; Tmax = Tmin; Tsum = zero(FT); ϕsum = zero(FT); ρsum = zero(FT); κsum = zero(FT); zsum = zero(FT)
            for kk in k:k2
                Tkk = T[I, kk]
                Tmin = min(Tmin, Tkk); Tmax = max(Tmax, Tkk)
                Tsum += Tkk; zsum += Z[I, kk]; ϕsum += ϕ[I, kk]; ρsum += Rho[I, kk]; κsum += Kc[I, kk] / (Rho[I, kk] * Cp[I, kk])
            end
            η = _viscosity_at(visc, splits, zsum / n, Tsum / n, ϕsum / n, ε)
            Nu = convective_nusselt(Tmax - Tmin, _thickness(Δz, k, k2, nz), ρsum / n, κsum / n, η; g, α, coeffs, Ra_crit, Nu_max)
            for kk in k:k2
                Nu_out[I, kk] = Nu
            end
            k = k2 + 1
        else
            k += 1
        end
    end
end

# thickness of the run of cells k:k2 (of nz in the column): `n Δz` for constant spacing, the sum of the control-volume widths on a refined grid
@inline _thickness(Δz::Number, k, k2, nz) = (k2 - k + 1) * Δz
@inline function _thickness(h::AbstractVector, k, k2, nz)
    L = zero(eltype(h))
    for kk in k:k2
        L += _cw(h, kk, nz)
    end
    return L
end

# the viscosity law of a region follows from its mean depth: `laws[i]` applies above `splits[i]` (z, negative downward; the last law below all splits)
@inline _viscosity_at(laws::Tuple{Any}, ::Tuple{}, z, T, ϕ, ε) = convective_viscosity(laws[1], T, ϕ, ε)
@inline _viscosity_at(laws::Tuple, splits::Tuple, z, T, ϕ, ε) = z >= splits[1] ? convective_viscosity(laws[1], T, ϕ, ε) : _viscosity_at(Base.tail(laws), Base.tail(splits), z, T, ϕ, ε)

# The outermost cells of a convecting region keep Nu = 1: the solver takes the conductivity of a cell face as the arithmetic mean of the two cells, so a jump of Nu across a region boundary
# would let the cold wall conduct at about the average conductivity (Nu K / 2) instead of at its own. The cell layer at the boundary is the conductive thermal boundary layer of the region.
@kernel function _convection_boundary_layer!(Nu_out, Nu_in)
    I = @index(Global, Cartesian)
    FT = eltype(Nu_out)
    Nu_out[I] = Nu_in[I]
    if Nu_in[I] > one(FT)
        for d in 1:ndims(Nu_in), s in (-1, 1)
            J = I + s * _unit(d, Nu_in)
            inside = true
            for k in 1:ndims(Nu_in)
                inside &= (1 <= J[k] <= size(Nu_in, k))
            end
            if inside && Nu_in[J] <= one(FT)
                Nu_out[I] = one(FT)
            end
        end
    end
end

@inline _is_melt(ϕ, Phases, I, k, ϕc) = (ϕ[I, k] >= ϕc) & (Phases[I, k] != 0)

"""
    compute_convective_nusselt!(Nu, Kc, T, ϕ, Rho, Cp, Phases, Z, Δz, visc = DEFAULT_CONVECTION_VISCOSITY; z_split = (), g = 9.81, ϕ_crit = 0.5, α = 3.0e-5,
                                coeffs = (0.0087, 0.1350, -0.2943), Ra_crit = 1708.0, Nu_max = 100.0, strainrate = 1.0e-12)

Fill `Nu` with the Nusselt number of the convecting region every cell belongs to (1 outside): a region is a vertical run of cells with `ϕ ≥ ϕ_crit` (air cells, phase 0, excluded),
see the header of `Convection.jl`; every region (e.g. two chambers at different depths, separated by cells with less melt) has its own thickness, temperature contrast and Rayleigh number.
`Kc` is the conductivity, `T` the temperature in K and `Z` the vertical coordinate of the cells [m]; `Δz` is the vertical spacing, or for a refined grid the vector of the distances between neighbouring points (`diff(z)`). `visc` is a GeoParams creep law (any law with `compute_τII(law, ε; T, ϕ)`, e.g.
`GiordanoMeltViscosity` or `ViscosityPartialMelt_Costa_etal_2009`), evaluated at the strain rate `strainrate` [1/s] (relevant for strain-rate dependent laws), or a tuple of laws for chambers of
different composition: `visc[i]` applies to the regions whose mean `z` is above `z_split[i]` and the last law to the regions below all splits (`length(z_split) == length(visc) - 1`, `z_split` descending, e.g. `(-8.0e3,)`).
With `boundary_layer = true` (default) the outermost cells of every region keep `Nu = 1`: the solver averages the conductivity of two cells arithmetically on their face, so a jump of `Nu` across a region boundary would let the cold wall conduct at about `Nu K/2`; the boundary layer of one cell is the conductive thermal boundary layer of the region.
The conductivity to use in the energy equation is `Nu .* Kc`. The vertical direction is the last one; works for 2D, axisymmetric and 3D arrays on any backend.
"""
function compute_convective_nusselt!(Nu, Kc, T, ϕ, Rho, Cp, Phases, Z, Δz, visc = DEFAULT_CONVECTION_VISCOSITY; z_split = (), g = 9.81, ϕ_crit = 0.5, α = 3.0e-5, coeffs = (0.0087, 0.135, -0.2943), Ra_crit = 1708.0, Nu_max = 100.0, strainrate = 1.0e-12, boundary_layer = true)
    FT = eltype(Nu)
    nz = size(Nu, ndims(Nu))
    laws = visc isa Tuple ? visc : (visc,)
    splits = Tuple(FT.(z_split))
    length(splits) == length(laws) - 1 || throw(ArgumentError("z_split needs one depth less than the viscosity laws: got $(length(splits)) split(s) for $(length(laws)) law(s)"))
    issorted(splits; rev = true) || throw(ArgumentError("z_split must be in descending order (z is negative downward)"))
    _launch!(_convection_columns!, Nu, size(Nu)[1:(end - 1)], Nu, Kc, T, ϕ, Rho, Cp, Phases, Z, laws, splits, nz, FT(ϕ_crit), FT(g), FT(α), FT.(coeffs), FT(Ra_crit), FT(Nu_max), FT(strainrate), Δz isa Number ? FT(Δz) : _on_backend(Nu, Δz))
    if boundary_layer                                  # the outermost cells of every region keep Nu = 1 (conductive boundary layer)
        Nu0 = copy(Nu)
        _launch!(_convection_boundary_layer!, Nu, size(Nu), Nu, Nu0)
    end
    return nothing
end

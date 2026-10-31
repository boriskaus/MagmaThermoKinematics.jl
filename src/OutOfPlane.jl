# Out-of-plane heat loss for 2D (planar) models
#
# A planar section ignores conduction along the third (out-of-plane) direction, which cools a compact body faster than the 2D model allows.
# To first order that loss is the missing term `K ∂²θ/∂y²` of the temperature anomaly `θ = T - Tbg` relative to a background temperature `Tbg`.
# For an anomaly that is Gaussian in the out-of-plane direction with variance `L²`, `∂²θ/∂y² = -θ/L²` at the section, so the loss is the volumetric sink `-K θ / L²`.
# `L²` is measured from the model itself: at every depth the horizontal variance of the positive anomaly, assuming that the body is as wide along y as along x.

"""
    out_of_plane_width2(θ, X; smooth = 10, Lmin = 1.0e3, dx = nothing, dz = nothing) -> Vector

Squared out-of-plane length scale `L²(z)` [m²] per depth cell: the horizontal variance
`∫θ (x - x̄)² dx / ∫θ dx` of the (positive) temperature anomaly `θ` [K], with the moments summed over `±smooth`
depth cells and `L²` bounded below by `Lmin²`. `X` holds the horizontal coordinate of every cell, `θ` and `X` are
`Nx × Nz` arrays (any backend); the first dimension is horizontal. Depths without an anomaly get a very large `L²`
(no loss). On a refined grid, the vectors `dx` and `dz` (host) with the widths of the cells along `x` and `z`
weight the moments.
"""
function out_of_plane_width2(θ::AbstractMatrix, X::AbstractMatrix; smooth::Integer = 10, Lmin::Real = 1.0e3, dx = nothing, dz = nothing)
    dx === nothing || (θ = θ .* _on_backend(θ, dx ./ (sum(dx) / length(dx))))     # weights relative to the mean width, so that the anomaly threshold below keeps its meaning
    m0 = Array(sum(θ; dims = 1))[1, :]
    m1 = Array(sum(θ .* X; dims = 1))[1, :]
    m2 = Array(sum(θ .* X .^ 2; dims = 1))[1, :]
    if dz !== nothing
        wz = dz ./ (sum(dz) / length(dz))
        m0, m1, m2 = m0 .* wz, m1 .* wz, m2 .* wz
    end
    nz = length(m0)
    L2 = fill(1.0e14, nz)
    for k in 1:nz
        lo, hi = max(1, k - smooth), min(nz, k + smooth)
        a0 = sum(@view m0[lo:hi])
        a0 > 1.0e-3 || continue
        x̄ = sum(@view m1[lo:hi]) / a0
        L2[k] = max(sum(@view m2[lo:hi]) / a0 - x̄^2, Lmin^2)
    end
    return L2
end

"""
    add_out_of_plane_loss!(Hr, T, Tbg, Kc, X, Phases; factor = 1.5, smooth = 10, Lmin = 1.0e3, air_phase = 0, dx = nothing, dz = nothing) -> L2

Add the out-of-plane heat loss `-factor · Kc (T - Tbg) / L²(z)` [W/m³] to the volumetric heat source `Hr`, with `L²(z)`
from [`out_of_plane_width2`](@ref) (cells of `air_phase` are excluded). `factor = 1` is exact for a Gaussian anomaly; the default
`factor = 1.5` accounts for the heavier tails of the thermal halo of a long-lived intrusion, whose variance exceeds that of its core,
and reproduces 3D models of such a body best. `dx`, `dz`: cell widths of a refined grid (see [`out_of_plane_width2`](@ref)).
Returns the length-scale profile.
"""
function add_out_of_plane_loss!(Hr, T, Tbg, Kc, X, Phases; factor::Real = 1.5, smooth::Integer = 10, Lmin::Real = 1.0e3, air_phase::Integer = 0, dx = nothing, dz = nothing)
    FT = eltype(Hr)
    θ = @. ifelse(Phases == air_phase, zero(FT), max(T - Tbg, zero(FT)))
    L2 = out_of_plane_width2(θ, X; smooth, Lmin, dx, dz)
    L2d = reshape(_on_backend(Hr, L2), 1, :)
    @. Hr -= ifelse(Phases == air_phase, zero(FT), FT(factor) * Kc * (T - Tbg) / L2d)
    return L2
end

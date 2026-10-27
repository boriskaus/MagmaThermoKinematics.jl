# Grid Refinement

By default MTK uses a grid with constant spacing in every direction. A refined grid concentrates the grid points in a
zone of interest, for example around a magma chamber or the region where sills are emplaced, while the number of points
stays the same. Every direction is refined independently (in a 1D manner), so the grid remains a tensor product of 1D
coordinate vectors `x`, (`y`), `z` and all routines keep their structure. The spacing in the zone can thus be several
times smaller than that of a uniform grid with the same number of points, which allows, for example, thinner sills and
smaller eruptions to be resolved (an eruption is realised once the drained thickness exceeds `resolve` cells, see
[Eruptions](eruptions.md)) without increasing the cost per time step.

## Usage

With [`MTK_GeoParams`](@ref), refinement is set with the `refine` field of `NumParam`: a named tuple with an entry per
refined direction (`x`, `y`, `z`; `x` and `z` in 2D), each holding the keyword arguments of [`refined_coordinates`](@ref):

```julia
Num = NumParam(Nx = 201, Nz = 201, W = 20e3, H = 20e3, axisymmetric = true,
               refine = (x = (zone = (0.0, 5e3), ratio = 4),            # 4 times finer within 5 km of the axis
                         z = (zone = (-10e3, -4e3), ratio = 3)))         # 3 times finer between 4 and 10 km depth
```

This grid has a spacing of 50 m within 5 km of the axis (198 m far from it) and of 63 m between 4 and 10 km depth (189 m
outside), instead of 100 m everywhere with the same 201 × 201 points. Directions without an entry keep a constant
spacing. Once `MTK_GeoParams` has created the grid, `Num.dx`, `Num.dy`, `Num.dz`, `Num.Δ` and `Num.Δmin` hold the smallest
spacing per direction, and the time step `Num.dt = fac_dt Δmin²/(4 κ_time)` and the number of time steps `Num.nt` are
recomputed from it ([`set_grid_spacing!`](@ref)), unless they were set explicitly.

A grid can also be created directly,

```julia
Grid = CreateGrid(size = (201, 201), extent = (20e3, 20e3), refine = (x = (zone = (0.0, 5e3), ratio = 4),))
Grid = CreateGrid(coords = (x, z))        # from given coordinate vectors
```

and a `CartData` set of GeophysicalModelGenerator that was created on refined coordinates (e.g. with `xyz_grid` and
vectors from [`refined_coordinates`](@ref)) gives a refined grid with `CreateGrid(d::CartData)`. A grid with constant
spacing has ranges as coordinates (`Grid.ConstantΔ == true`); a refined grid stores the coordinates of all directions as
`Vector`s, and `Grid.Δ` holds the smallest spacing per direction.

## Coordinates

[`refined_coordinates`](@ref)`(lo, hi, n; zone = (a, b), ratio)` distributes `n` points over `[lo, hi]` such that the
spacing is proportional to

```math
s(x) = \begin{cases} 1 & a \le x \le b \\ 1 + (r - 1)\,\dfrac{1 - \cos(\pi d/L_t)}{2} & d < L_t \\ r & d \ge L_t \end{cases}
```

where $d$ is the distance from the zone, $r$ = `ratio` and $L_t$ the length of the transition: the points are equally spaced in
$u(x) = \int_{lo}^x \mathrm{d}x'/s(x')$. The spacing is therefore constant in the zone and grows smoothly to $r$ times that value.
By default $L_t$ is chosen such that the spacing of neighbouring points differs by at most a factor `max_growth` (1.05), as
the error of the finite differences grows with the change of the spacing between neighbours. A longer transition costs
points, which limits the refinement that can be reached with a given number of points; a larger `max_growth` or an explicit
`transition` allows a stronger refinement. Instead of `ratio`, the spacing in the zone can be prescribed with `Δ`.

With `transition = 0` the spacing changes abruptly at the edges of the zone, which are then grid points: the spacing is
constant inside the zone and (about `ratio` times larger) constant outside it. No points are spent on a transition, so for a
given number of points the zone is resolved best. The discretisation remains conservative across the jump, but the
truncation error at the jump is first order, so an abrupt change works best when the zone encloses the region of strong
temperature gradients with some margin. For the decay of a Gaussian anomaly (initial width 1 km, after 20 kyr, explicit
solver, 2D, 61 × 61 points on 20 × 20 km, refined zone centred on the anomaly, `ratio = 3`):

| grid | spacing in the zone | max. error, zone ± 3 km | max. error, zone ± 5 km |
|:-----|:--------------------|:------------------------|:------------------------|
| uniform | 333 m | 0.061 K | 0.061 K |
| gradual (`max_growth = 1.1`) | 239 m / 291 m | 0.029 K | 0.044 K |
| abrupt (`transition = 0`) | 176 m / 222 m | 0.102 K (at the jump) | 0.026 K |

When the zone edge lies at 2.5 widths of the anomaly (± 3 km), the error of the abrupt grid is largest at the jump; with
the edge further out (± 5 km) the abrupt grid is the most accurate.

## Discretisation

The temperature lives on the grid points and the heat flux on the faces midway between them, so on a refined grid the
diffusion operator becomes, along every direction,

```math
\left(\frac{\partial}{\partial x} k \frac{\partial T}{\partial x}\right)_i \approx
\frac{1}{(h_{i-1} + h_i)/2}\left( k_{i+\frac{1}{2}}\frac{T_{i+1} - T_i}{h_i} - k_{i-\frac{1}{2}}\frac{T_i - T_{i-1}}{h_{i-1}} \right),
\qquad h_i = x_{i+1} - x_i ,
```

the balance of the fluxes through the faces of the control volume of width $(h_{i-1} + h_i)/2$ (with the face radii as
weights in the axisymmetric case). The scheme is conservative, exact for quadratic temperature fields (planar geometry) and second-order
accurate on smoothly varying grids; with constant spacing it is the usual stencil, and grids with constant spacing give the
same results as before. The explicit and the implicit solver use the same operator. The explicit time step is limited by the
smallest spacing, so a refinement by a factor $r$ reduces the time step by $r^2$; the implicit solver
(`solver = :implicit_enthalpy`) is not subject to this limit, although its pseudo-transient iterations increase with the
ratio of the time step to the explicit limit.

All other routines take the variable spacing into account:

- sill injection (`inject_sills`) and the semi-Lagrangian advection of temperature (`AdvectTemperature!`) locate the departure
  points on the coordinate vectors (by bisection, on any backend), and the number of pseudo-time steps of the host-rock
  displacement follows from the smallest spacing;
- tracers interpolate the temperature and melt fraction bilinearly (trilinearly) on the coordinate vectors, and phases are
  assigned to the nearest grid point;
- the chamber of the eruption model is measured with the cell volumes (volume, footprint, thickness, mean temperature, melt
  fraction and pressure are volume-weighted), the eruption sources carry the volumes of their cells, and the free surface
  is tracked between the vertical grid points;
- the lithostatic pressure, the thickness of convecting layers and the moments of the out-of-plane heat loss use the
  spacing of the grid points.

The quadratic and cubic interpolation options of `AdvectTemperature` (CPU) need a grid with constant spacing.
`Num.refine` cannot be combined with `CartData` input in `MTK_GeoParams`; create the `CartData` set on refined coordinates
instead. The mean melt fraction and magma temperature stored by the default `MTK_update_TimeDepProps!` are volume-weighted.

## Tests

`test/test_RefinedGrid.jl` verifies that:

- the coordinates have the requested spacing in the zone, ratio and maximum growth, and the prescribed spacing `Δ` is reached;
- the refined stencils reproduce the results of the constant-spacing stencils when the spacing is constant (explicit and
  implicit, 2D, axisymmetric and 3D), are exact for quadratic temperature fields, and that the implicit step agrees with a
  direct solve of the same discrete system;
- the decay of a Gaussian temperature anomaly agrees better with the analytical solution on a refined grid than on a uniform
  grid with the same number of points (planar and axisymmetric), and that the heat content is conserved;
- the lithostatic pressure, the advection of a linear field, the interpolation to tracers of a bilinear field and the
  assignment of phases are exact on refined grids;
- the chamber geometry, pressure, eruption volumes and surface displacement on a grid that is refined around the chamber
  equal those of a uniform grid with the same spacing in the chamber;
- `MTK_GeoParams` runs on refined grids in 2D and 3D, with identical results on the CPU and `JLArrays` backends, and
  with the implicit solver.

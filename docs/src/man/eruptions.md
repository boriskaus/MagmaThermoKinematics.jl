# Magma-chamber overpressure and eruptions

MagmaThermoKinematics.jl can track the pressure of a melt-rich magma chamber on the temperature grid, let the chamber erupt when its overpressure is large enough, and turn the volume that is added to or removed from the host rock into a displacement field. That field advects the temperature and the tracers and moves the free surface. The model is available in 2D (planar and axisymmetric sections are both described by a chamber of equivalent radius) and 3D, on the CPU and on GPUs. It is a set of building blocks that you call from the simulation loop, see [Using the model in a simulation](#Using-the-model-in-a-simulation).

## The chamber

The chamber is the largest face-connected region of the grid in which the melt fraction is at least `ϕ_erupt` (`chamber_cells`; on a GPU the regions are labelled on the device with `chamber_labels!` and only the labels are copied to the host). It only counts as a chamber if its melt thickness

```math
h_{melt} = \frac{\sum_i \phi_i\, \Delta V_i}{W}
```

is at least `h_melt_min`, where ``W`` is its footprint (the width in 2D, the projected area in 3D). If no region qualifies, or the chamber moves to a different depth range, the state is reset. The chamber is described by an equivalent column (`chamber_geometry`): volume ``V`` (m² in 2D, m³ in 3D), footprint ``W``, mean thickness ``h = V/W``, lateral radius ``R`` (``W/2`` in 2D, ``\sqrt{W/\pi}`` in 3D) and the factor `scale` ``=\pi R^2 h / V`` that converts the native volume into the equivalent 3D volume.

## Overpressure

The state of the chamber (`EruptionState`) is its pressure ``P``, the overpressure ``\Delta P = P - P_{lith}`` with respect to the lithostatic pressure at the centre of the chamber (integrated through the geotherm with the crustal density law), its volume and the inventories of magma and H₂O. The magma is a mixture of melt, crystals and exsolved H₂O; the amount of dissolved H₂O follows the solubility law (`solubility`, a GeoParams law), the density of the gas the equation of state `ρ_gas`, and the melt viscosity `melt_viscosity`.

During a thermal step ``\Delta t`` the grid only contributes the *change* of the chamber volume (melting and freezing at the margins). The magma mass changes by the recharge ``\dot a`` (volume per second and footprint area, supplied by the caller) and the pressure relaxes towards the equilibrium of a visco-elastic shell,

```math
\frac{d\,\Delta P}{dt} = \frac{\Delta P_{eq} - \Delta P}{\tau}, \qquad \Delta P_{eq} = \eta_r\, S, \qquad \tau = \eta_r \left(\frac{1}{\beta_r} + \frac{1}{\beta_m}\right),
```

with the source ``S = \dot M_{in}/(\rho V) - \dot\rho/\rho`` (recharge minus the density change of the magma by cooling and crystallisation), the compressibility of the magma ``1/\beta_m = \rho^{-1}\,\partial\rho/\partial P``, and the compliance of the host rock ``1/\beta_r = 3/\bigl(4\mu\,\min(h/2R, 1)\bigr)`` of a sill-like chamber (the spherical limit for an aspect ratio of one; ``\mu`` is `μ_shear`). The relaxation viscosity ``\eta_r`` is that of the shell around the chamber: a steady spherical conduction profile between the temperature of the chamber and the far-field geotherm, an Arrhenius creep law ``\eta = A \exp(G/RT)`` (`A_visc`, `G_act`) and a shell radius of 11 chamber radii, clamped to ``10^{17}``–``10^{24}`` Pa s (`crustal_relaxation_viscosity`). The step is divided into as many sub-steps as needed to keep the pressure change per sub-step below a tenth of `ΔP_crit`, and the pressure and volume are projected onto the mass constraint ``\rho(P)\,V(P) = M`` after every sub-step.

## Eruption

A chamber erupts when, at the end of a sub-step,

* the overpressure has reached the critical value, ``\Delta P \ge \Delta P_{crit}``,
* the exsolved gas fraction is below `ϕ_g_crit` (a gas-rich chamber is compressible and cannot erupt), and
* a dike can reach the surface (`dike_ascends`): the pressure that survives the ascent exceeds the stress barrier, ``\Delta P + P_{lith} - \rho g\, d \ge \sigma_{barrier}`` for a chamber at depth ``d``, and the depth is smaller than the distance that the dike travels before it freezes, the positive root ``L`` of ``L^2 - \Delta\rho g\, c\, L - \Delta P\, c = 0`` with ``c = w^4/(3\eta\kappa)`` (`w_dike` the dike half-width, `κ_magma` the thermal diffusivity of the magma and ``\eta`` its viscosity).

The chamber then drains to ``\Delta P_{relax}`` (`ΔP_relax`, zero by default); the magma mass that leaves follows from the mass balance and is returned as a bulk thickness ``h_{erupt}``, that is, as the fraction ``f = h_{erupt}/h`` of the chamber volume.

### Resolvable withdrawal

On the grid the drained volume can only be removed in whole displacement fields, and a drainage of a small fraction of a cell would otherwise be lost. `chamber_step!` therefore queues the drained thickness (`pending_withdrawal!`) and only realises an eruption when the queue exceeds `resolve` ``\cdot\,\Delta z`` (`resolve = 2` by default; smaller values give more frequent, smaller events and the displacement field is continuous, so values below one are admissible). The queued thickness is debited with `commit_pending_withdrawal!`. Eruptions of chambers closer than `margin` to the top or bottom of the grid are not realised. The size of the smallest event is therefore set by the grid, ``\approx f_{min}\,\pi R^2 h`` with ``f_{min} h = resolve\cdot\Delta z``, and the total erupted volume is controlled by the mass balance of the chamber, not by the resolution.

`chamber_step!` returns `nothing` as long as no eruption is realised, otherwise a named tuple with the drained thickness `h_erupt`, the fraction `f`, the chamber `cells`, its geometry `geo` and the chamber generation `chamber`.

## Host-rock displacement

`erupt!` removes the fraction `f` of the volume of every chamber cell. Every cell becomes a point source of volume change in the elastic half-space (`mogi_displacement!`, `halfspace` keyword):

| `halfspace` | solution |
|:------------|:---------|
| `:mogi` (default) | Mogi (1958): the source's full-space displacement ``(1-\nu)\,\Delta V/\pi\; \mathbf{d}/\lvert\mathbf{d}\rvert^3``, which equals the half-space surface displacement. Used with line sources in 2D (below). |
| `:image` | volume-consistent dilatation centre ``\Delta V/(4\pi)\,\mathbf d/\lvert\mathbf d\rvert^3`` with an opposite-sign image above the surface; the surface uplift is twice the full-space value. |
| `:mindlin` | the exact half-space solution for a centre of dilatation (Mindlin and Cheng, 1950): volume change ``\Delta V`` in the bulk and the Mogi displacement at the free surface. 3D point sources only. |

The sum over all sources is done on the backend of the arrays (`KernelAbstractions`), with a softening length `softening` (default the smallest grid spacing) for the nodes at the sources themselves. `volume_factor` scales the displaced volume.

In a planar 2D section a cell has no extension along the out-of-plane direction. By default each cell is a line source over the half-extent ``L(x) = \sqrt{R^2 - (x-x_c)^2}`` of a circular footprint of the chamber (`erupt_sources`), so that the sources integrate to the 3D volume ``f\,\pi R^2 h``. A line source decays differently from a point source, and the half-space surface response of the planar image kernel is only ``1/(2(1-\nu))`` of the Mogi value, so that planar sections tend to under-estimate the subsidence of a 3D chamber. With `revolve = true` the chamber is instead treated as a **body of revolution about its own vertical axis** (`revolved_sources`): a cell of area ``dA`` at the horizontal distance ``a = \lvert x - x_c\rvert`` from the axis (the mean ``x`` of the chamber cells) is a ring, shared by the cell and its mirror cell, that loses the volume

```math
\Delta V_{ring} = -f\,\pi\,\lvert a\rvert\, dA ,
```

split over `nθ` Mindlin point sources at the azimuths ``\vartheta_j = 2\pi (j - 1/2)/n_\vartheta``, ``(a\cos\vartheta_j, a\sin\vartheta_j, z)``. The total drained volume is ``\sum_i f\pi\lvert a_i\rvert dA = f\,\pi R^2 h`` for a chamber of uniform thickness and half-width ``R``. The displacement is evaluated on the section through the axis, ``(x - x_c, 0, z)``; for a symmetric chamber it is the axisymmetric response, and the chamber may sit anywhere in the section. Because the receivers lie in the plane ``y=0``, the sources at ``\pm\vartheta`` give the same ``(u_x, u_z)``, so by default (`symmetric = true`) only half of the azimuths are summed with twice the volume, which is exact. With `coarsen > 1` the cells are binned in blocks of `coarsen` × `coarsen` cells (one ring per block) and with `float_type = Float32` the sum is done in single precision. `halfspace` defaults to `:mindlin` with `revolve = true`.

!!! warning "Flat half-space and the position of the free surface"
    The `:image` and `:mindlin` kernels are solutions for an elastic half-space with a *flat* free surface at `z_surface` (default 0). Topography is not part of the elastic problem. The solutions are not valid above that plane: on the axis of a source at depth ``c`` below it the vertical displacement changes sign at about ``c/3`` above the plane and diverges at the mirror image, ``c`` above it. `z_surface` must therefore be at or above the highest rock surface of the model and has to be updated when the free surface rises, for example `z_surface = maximum(Zsurf)`. `erupt!` takes the surface-elevation array as `Zsurf` and warns if the free surface is above `z_surface`.

The displacement field is applied with `displace_temperature!` (semi-Lagrangian advection of the temperature in as many sub-steps as needed to keep the displacement per sub-step below half a cell), and the tracers inside the chamber are removed with probability `f` (returned as `cargo`), and the remaining tracers are advected with the displacement. Both are done by `erupt!`, which returns the displacement `U` and the removed tracers.

## Moving free surface

With `move_free_surface!` the boundary between the air and the rock moves with the vertical displacement at the surface. `Zsurf` holds the surface elevation of every column (one value per horizontal grid node, initialised from the phases when it is `NaN`) and is shifted by the displacement at the surface, so that displacements smaller than a cell accumulate. Cells above the surface become `air`, cells below it that were air become `fill` (the rock phase), and `Phases_init` is updated in the same way so that the reset of the phases after an injection keeps the moved surface. The temperature of the cells that turned into air is that of the air above them.

The simulation does this after every sill injection if `NumParam(move_free_surface = true, air_phase = 0)` is set (the displacement of `inject_sills` is used). The arrays `Arrays.Zsurf`, `Arrays.Phases` and `Arrays.Phases_init` are created by the simulation. To keep the air at the surface temperature, combine it with the sticky-air boundary condition, `NumParam(sticky_air_BC = true, AirPhase = 0, Tsurface_Celsius = ...)`; the air phase then needs `Phases` and material parameters like any other phase.

## Inflation by sills

The displacement of the injected magma can be applied in the same way. `InjectSills.PennyShapedSill` gives the displacement of a single penny-shaped sill; `revolved_inflation_rate` gives the mean displacement rate of an ensemble of ``N`` sills on a planar section. The sills are treated as parts of bodies of revolution as in the eruption sink: sill ``k`` has its centre at the distance `offset[k]` from a vertical axis at `axis[k]`, at the depth `zc[k]`, radius `radius[k]` and thickness `thickness`; the displacement is averaged over `nθ` azimuths and evaluated on the section through the axis. The caller chooses the positions, radii and orientations (`angles`, optional dip and strike). With the time `interval` between two sills of the ensemble (the mean sill volume divided by the injection rate, ``\Delta t_s = \bar V_s / Q`` with ``\bar V_s = \langle \tfrac{2}{3}\pi R_s^2 H_s\rangle``) the result is

```math
\dot{\mathbf u}_{infl}(\mathbf x) = \frac{1}{N\,\Delta t_s} \sum_{k=1}^{N} \frac{1}{n_\vartheta} \sum_{j=1}^{n_\vartheta} \mathbf u_{penny}\bigl((x - x_k, 0, z);\ \mathbf c_{k,j}, R_k, H\bigr).
```

Multiply it by a time step to get the displacement for `displace_temperature!` and `move_free_surface!`.

!!! note "Volume consistency at the surface"
    The displacement of a `PennyShapedSill` is a full-space solution: the volume of the surface uplift it produces is about half of the sill volume ``V`` (a horizontal crack in a half-space lifts the surface by about ``V``), whereas the Mindlin sink displaces ``2(1-\nu)\,\Delta V \approx 1.5\,\Delta V`` at the surface. Inflation and eruption sinks are therefore not volume-consistent at the surface. Use `volume_factor` of `erupt!` and a scaling of the inflation rate if a consistent balance is wanted, and compare the net uplift with an axisymmetric or 3D reference when the surface elevation matters.

## Parameters

`EruptionParams` collects the parameters of the model (pressures in Pa, lengths in m, times in s):

| field | default | meaning |
|:------|:--------|:--------|
| `ϕ_erupt` | 0.5 | minimum melt fraction of a chamber cell |
| `h_melt_min` | 500 | minimum melt thickness of a chamber [m] |
| `ΔP_crit` | 20e6 | overpressure at which the chamber drains [Pa] |
| `ΔP_relax` | 0 | overpressure that the chamber relaxes to after an eruption [Pa] |
| `ϕ_g_crit` | 0.5 | gas volume fraction above which the chamber cannot erupt |
| `σ_barrier` | 10e6 | stress barrier that the dike has to overcome [Pa] |
| `w_dike`, `κ_magma` | 2, 1e-6 | dike half-width [m] and thermal diffusivity of the magma [m²/s] |
| `μ_shear`, `ν` | 1e10, 0.25 | shear modulus [Pa] and Poisson ratio of the host rock |
| `A_visc`, `G_act` | 4.25e7, 141e3 | Arrhenius constants of the crustal creep law, ``\eta = A\exp(G/RT)`` [Pa s, J/mol] |
| `m_w` | 0 | total H₂O mass fraction of the magma |
| `ρ_melt`, `ρ_x`, `crust` | (2400, 3e-5, 1e-10), (2700, 3e-5, 1.5e-11), (2700, 3e-5, 1.5e-11) | density laws ``\rho_0\,(1 - \alpha\,(T - 273.15) + \beta P)`` of the melt, crystals and crust as `(ρ0, α, β)` |
| `melt_viscosity`, `solubility`, `ρ_gas` | `LinearMeltViscosity()`, `Liu2005_Solubility()`, `RedlichKwong_Density()` | GeoParams laws for the melt viscosity, H₂O solubility and gas density |
| `g` | 9.81 | gravity [m/s²] |

The arguments of the functions of the model:

| function | purpose |
|:---------|:--------|
| `chamber_step!(state, ep, ϕ, T, coords; recharge, Δt, time, T_background, margin, resolve)` | find the chamber, advance its pressure by `Δt`, return the eruption event or `nothing`. `recharge` is the magma volume added inside the chamber per second in native units (m²/s in 2D, m³/s in 3D), `T_background(depth)` the undisturbed geotherm [°C] and `coords` the 1D coordinate vectors (`z` last, negative downward). |
| `erupt!(T, Tbuf, Tracers, coords, event; ...)` | remove the drained volume: displacement field, advection of `T` and the tracers, returns `(U, cargo)`. |
| `move_free_surface!(Phases, Phases_init, T, Zsurf, coords, U; air, fill)` | move the free surface with the displacement `U`. |
| `revolved_inflation_rate(X, axis, offset, zc, radius, interval; thickness, ...)` | mean displacement rate of an ensemble of sills on a planar section. |
| `displace_temperature!(T, Tbuf, coords, U)` | advect the temperature with a displacement field. |
| `reset_chamber!(state)` | forget the chamber. |

## Using the model in a simulation

The solver calls a number of user functions at every time step (see `MTK_GeoParams`); any of them can be replaced for one simulation through the `callbacks` keyword. How the magma that feeds the chamber is supplied is up to the user: by injected sills, or by a volumetric source set in `MTK_source!` (see its docstring). The eruption model only needs the resulting `recharge` of the chamber, the magma volume added to it per second (m²/s in 2D, m³/s in 3D). With topography, the lithostatic pressure of the chamber is best taken from the model itself: one top-down integration of the density with the air at zero density, and the surface elevation `Arrays.Zsurf` for the depth below the surface. The eruption model is called from a function that runs after the diffusion step, for example the `updateTracers` callback, which also receives the tracers:

```julia
using MagmaThermoKinematics, GeoParams
using MagmaThermoKinematics.MTK_GMG

ep = EruptionParams(ΔP_crit = 15.0e6, σ_barrier = 5.0e6, w_dike = 5.0, m_w = 0.04, h_melt_min = 400.0,
    melt_viscosity = LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.9606K, η0 = 1Pas))
state = EruptionState()

function erupt_step(Grid, Arrays, Tracers, Dikes, time_props, Num)
    Tracers = MTK_GMG.MTK_updateTracers(Grid, Arrays, Tracers, Dikes, time_props, Num)    # default behaviour
    coords = Tuple(Grid.coord1D)
    ρ_rock .= ifelse.(Arrays.Phases .== Num.air_phase, 0.0, Arrays.Rho)          # air has no weight
    MagmaThermoKinematics.lithostatic_pressure!(P_lith, ρ_rock, 9.81, step(coords[end]))
    recharge = state.init ? magma_supply(state) : 0.0                             # [m²/s], from your source of magma
    ev = chamber_step!(state, ep, Arrays.ϕ, Arrays.T, coords; recharge, Δt = Num.dt, time = Num.time,
                       T_background = depth -> 10.0 + 25.0e-3 * depth, margin = 500.0, P = P_lith, Zsurf = Arrays.Zsurf)
    ev === nothing && return Tracers
    zs = maximum(filter(!isnan, Array(Arrays.Zsurf)))                          # flat half-space up to the free surface
    U, cargo = erupt!(Arrays.T, similar(Arrays.T), Tracers, coords, ev; revolve = true, nθ = 8, coarsen = 2,
                      z_surface = max(zs, 0.0), Zsurf = Arrays.Zsurf)
    move_free_surface!(Arrays.Phases, Arrays.Phases_init, Arrays.T, Arrays.Zsurf, coords, U; air = Num.air_phase, fill = 1)
    Arrays.Tnew .= Arrays.T
    return Tracers
end

Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(MatParam, Num, Sill_params;
    CartData_input = Data_2D, callbacks = (; updateTracers = erupt_step))
```

The complete example `examples/MTK_GMG_2D_eruptions.jl` runs this on a 2D section with a placeholder source (a heated box and a recharge proportional to the width of the chamber), the lithostatic pressure of the model, inflation of the source region with `revolved_inflation_rate`, a free surface that moves with the displacement and a sticky-air boundary condition. It runs in about a minute on a CPU and produces a few eruptions, which are printed together with the pressure of the chamber.

## Choices and limitations

* The displacement is the elastic response of a flat homogeneous half-space; the viscous relaxation of the host rock enters only through the pressure model (the shell viscosity ``\eta_r``).
* The chamber is a single connected region; a chamber that splits is followed through its largest part. The overpressure model works with an equivalent column and the mean melt fraction and temperature of the chamber cells.
* An eruption removes the same fraction `f` of the volume of every chamber cell; the removed volume of the tracers is sampled with the probability `f`.
* The size of the smallest eruption is controlled by `resolve` and the grid spacing, see above; the total erupted volume is not.
* Planar sections represent the 3D chamber as a body of revolution (`revolve = true`) or by line sources over a circular footprint; combine them with the out-of-plane heat loss if the temperature field of a compact body matters.

## Tests

`test/test_Eruptions.jl` checks the Mogi, image and Mindlin kernels (against the analytic expressions, the flux of the displacement through a box around a source and `InjectSills.MindlinSphere`), the chamber detection and labelling in 2D and 3D, the overpressure and drainage cycle, the free surface, and the revolved sink (total volume, equality of the half and full azimuth sums, the Mogi far field at the free surface and with `z_surface` at a different elevation, and the single-precision sum) and the azimuth-averaged inflation (against the 3D sill displacement); `test/test_JLArrays.jl` runs the revolved sink and the inflation on `JLArrays`.

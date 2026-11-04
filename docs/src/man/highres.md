# High-resolution runs

These two simulations show most of the features of MagmaThermoKinematics together, at a high resolution: the last 150 kyr of a magmatic system below [Unzen volcano](https://en.wikipedia.org/wiki/Mount_Unzen) (Japan), in a 2D section and in 3D, below the present topography.

## 2D section

![High-resolution 2D run](../assets/movies/HighRes2D.gif)

A grid of 16001 × 10001 points (2.5 m spacing, 160 million cells), 19.8 h on one NVIDIA GH200. Sills and dikes intruded since the previous frame are shown in cyan, older ones in faint blue; the bottom panel gives the erupted volume per kyr (842 eruptions, 73 km³ in total).

## 3D

![High-resolution 3D run](../assets/movies/HighRes3D.gif)

A grid of 801 × 661 × 501 points (50 m spacing, 265 million cells), 21.8 h on one NVIDIA GH200 (44 eruptions, 26 km³ in total). Left: the temperature below the topography with a corner cut away, and the outlines of the sills and dikes on the cut faces. Right: the topography, the intrusive complex (faint blue) and the magma.

The shoreline moves during the run because the magmatic system uplifts the surface: every sill pushes the rock above it upwards, and only the eruptions let it subside again. Over the 150 kyr the surface above the intrusive complex rises by up to 400 m, about 1000 km² of it by more than 10 m. The coast moves outwards and the land area grows from 555 to 789 km².

## What the runs include

- **Sill intrusion** in two zones, a shallow one (3–7 km depth, 70 % of the magma) and a deep one (8–16 km), with a total magma flux of 2.36 km³/kyr (`Q_KM3`). Every sill opens as a penny-shaped crack and displaces the surrounding rock ([InjectSills.jl](https://github.com/JuliaGeodynamics/InjectSills.jl)). The sills are two grid cells thick, and at least 10 m. The intruded magma is a phase of its own (phase 4), which moves with the host rock; the 2D script also writes every sill to `sills.jld2`, which the movie uses to show the new ones.
- **Overpressure and eruptions** of the magma chamber that forms ([Eruptions](eruptions.md)): the erupted magma is removed, the host rock and the free surface move accordingly.
- **Layered host rock**: volcanic cover, upper and lower crust, built with the `add_layer!` function of [GeophysicalModelGenerator.jl](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl). Each layer has its own creep law for the viscous relaxation of the overpressure (`shell_rheology` and `shell_z_split` of `EruptionParams`), and the viscosity of the shell around the chamber is taken from the temperature field.
- **Deformation** of the crust: passive markers of the initial depth (the grey lines in the 2D movie, one per km) move with every sill and eruption.
- **Convective cooling** of melt-rich regions, **out-of-plane heat loss** (2D, towards the undisturbed geotherm, given as the field `Tbg` of the input), a **sticky-air** free surface, and the **implicit enthalpy solver** with a time step of 25 years.
- **Real topography** (`examples/Topo_cart`): the run starts from the present surface, lowered by an estimate of the uplift of the following 150 kyr (a Gaussian of up to 154 m). The computed surface moves more than that: in 3D it rises by up to 400 m; in the 2D section the eruptions dominate, and it subsides by up to 200 m.
- **Zircon ages** from the temperature–time paths of the tracers in the sills, computed with [ZirconGrowth.jl](https://github.com/JuliaGeodynamics/ZirconGrowth.jl) ([Zircon ages](zircon_growth.md)).

The runs start from the temperature field and the surface of a lower-resolution simulation of the preceding evolution of the system, given in closed form in the scripts: a background geotherm plus three warm bodies (Gaussians in depth and in distance from their own vertical axis), and the uplift of the surface during the run as a Gaussian, by which the initial surface lies below the present one.

## Running them

```bash
cd examples
RES=2.5 Q_KM3=2.36 GPU=1 julia --project HighRes_Unzen_2D.jl   # 2D, 2.5 m grid spacing
RES=50 julia --project HighRes_Unzen_3D.jl                      # 3D, 50 m grid spacing (GPU=0 runs on the CPU)
julia --project HighRes_Unzen_plots.jl HighRes2D_2m             # 2D movie and zircon ages
PVBATCH=/path/to/pvbatch julia --project HighRes_Unzen_3D_vtk.jl HighRes3D_50m   # 3D: VTK files and the movie with ParaView
```

`RES` sets the grid spacing [m]; coarser values (e.g. `RES=50` in 2D, `RES=200` in 3D) run in minutes on a single GPU. The memory needed is about 450 bytes per cell in 2D and 430 in 3D, so the 144 GB of a GH200 fit 2.5 m in 2D and 50 m in 3D, and an 80 GB GPU about 3 m in 2D and 60 m in 3D. `DT_YR` sets the time step, `SNAP_KYR` the interval of the snapshots and `SNAP_STRIDE` thins them out for large grids.

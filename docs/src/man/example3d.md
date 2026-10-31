# 3D Example

This page mirrors the 3D snippet from the repository README.

To go from 2D to 3D, only a few changes are required. The 3D example uses real topography, which [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) (≥ 0.7.23, add it to your environment) downloads without needing GMT; the air above the surface is held at 0°C. A movie of the example is shown below.

![](../assets/movies/Example3D.gif)

Here the full 3D code snippet from the README.

Note:
- The model arrays live on the KernelAbstractions `backend`; the example runs on an NVIDIA GPU (`CUDABackend()`, needs `CUDA` in your environment); use `backend = CPU()` to run it without a GPU.

```julia
using MagmaThermoKinematics
using CUDA
backend = CUDABackend()                 # runs on an NVIDIA GPU; without one, use backend = CPU() (and drop `using CUDA`)
using InjectSills
using WriteVTK
using GeophysicalModelGenerator                                   # downloads the topography (no GMT needed)

#------------------------------------------------------------------------------------------
@views function MainCode_3D()
    Nx, Ny, Nz = 250, 250, 250
    Grid = CreateGrid(size = (Nx, Ny, Nz), x = (-15.0e3, 15.0e3), y = (-15.0e3, 15.0e3), z = (-30.0e3, 4.0e3)) # grid points & domain [m]
    Num = Numeric_params(verbose = false, sticky_air_BC = true, AirPhase = 0)   # Nonlinear solver options; air (phase 0) is held at 0°C

    # Real topography, interpolated on the x,y grid
    Topo = import_topo(lon = [-71.9, -71.1], lat = [-39.95, -39.35], res = "03s")
    X, Y, Z = xyz_grid(Grid.coord1D[1] / 1.0e3, Grid.coord1D[2] / 1.0e3, 0)
    Surf = project_CartData(CartData(X, Y, Z, (Z = Z,)), Topo, ProjectionPoint(Lat = -39.6375, Lon = -71.5025))

    # Set material parameters
    MatParam = (
        SetMaterialParams(
            Name = "Air", Phase = 0,
            Density = ConstantDensity(ρ = 2800kg / m^3),
            HeatCapacity = ConstantHeatCapacity(Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(k = 1.5Watt / K / m)
        ),
        SetMaterialParams(
            Name = "Rock", Phase = 1,
            Density = ConstantDensity(ρ = 2800kg / m^3),
            HeatCapacity = ConstantHeatCapacity(Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(k = 1.5Watt / K / m),
            LatentHeat = ConstantLatentHeat(Q_L = 350.0e3J / kg),
            Melting = MeltingParam_Caricchi()
        ),
    )

    GeoT = 20.0 / 1.0e3                    # Geothermal gradient [K/km]
    W_in, H_in = 5.0e3, 0.5e3               # Width and thickness of dike
    T_in = 900.0                       # Intrusion temperature
    InjectionInterval = 0.1kyr                      # Inject a new dike every X kyrs
    maxTime = 15kyr                       # Maximum simulation time in kyrs
    H_ran, W_ran = Grid.L[1] * 0.3, Grid.L[3] * 0.4 # Size of domain in which we randomly place dikes and range of angles
    κ = 1.2 / (2800 * 1050)             # thermal diffusivity
    dt = minimum(Grid.Δ .^ 2) / κ / 10     # stable timestep (required for explicit FD)
    nt = floor(Int64, maxTime / dt)     # number of required timesteps
    nTr_dike = 300                         # number of tracers inserted per dike

    # Array initializations
    Arrays = CreateArrays(Dict((Nx, Ny, Nz) => (T = 0, T_K = 0, T_it_old = 0, Tupdate = 0, Rho = 2800, Cp = 1050, Tnew = 0, Hr = 0, Hl = 0, Kc = 1, P = 0, X = 0, Y = 0, Z = 0, ϕₒ = 0, ϕ = 0, dϕdT = 0)); backend)
    # CPU buffers
    Tnew_cpu = zeros(Float64, Grid.N...)
    Phi_melt_cpu = similar(Tnew_cpu)

    GridArray!(Arrays.X, Arrays.Y, Arrays.Z, Grid)
    Tracers = StructArray{Tracer{Float32}}(undef, 1)                   # Initialize tracers
    Zsurf = copyto!(similar(Arrays.T, Nx, Ny, 1), Surf.z.val .* 1.0e3)   # surface elevation [m]
    Phases = similar(Arrays.T, Int64); Phases .= ifelse.(Arrays.Z .> Zsurf, 0, 1) # air above the surface, rock below
    Arrays.T .= max.(Zsurf .- Arrays.Z, 0) .* GeoT                        # Initial (linear) temperature profile below the surface

    # Preparation of VTK/Paraview output
    if isdir("viz3D_out") == false
        mkdir("viz3D_out")
    end
    loadpath = "./viz3D_out/"
    pvd = paraview_collection("Example3D")

    time, dike_inj, InjectVol, Time_vec, Melt_Time = 0.0, 0.0, 0.0, zeros(nt, 1), zeros(nt, 1)
    for it in 1:nt   # Time loop

        if floor(time / InjectionInterval) > dike_inj       # Add new dike every X years
            dike_inj = floor(time / InjectionInterval)                                               # Keeps track on what was injected already
            cen = (Grid.max .+ Grid.min) ./ 2 .+ rand(-0.5:1.0e-3:0.5, 3) .* [W_ran;W_ran;H_ran]    # Randomly vary center of dike
            if cen[end] < -12.0e3
                Angle_rand = [rand(80.0:0.1:100.0); rand(0:360)]                        # Dikes at depth
            else
                Angle_rand = [rand(-10.0:0.1:10.0); rand(0:360)]
            end                    # Sills at shallower depth
            sill = InjectSills.EllipticalIntrusion(Center = InjectSills.Point3(cen[1], cen[2], cen[3]) * m, Angle = InjectSills.Vec2(Angle_rand[1], Angle_rand[end]) * NoUnits, W = W_in * m, H = H_in * m)
            Tnew_cpu .= Array(Arrays.T)
            Tracers, Tnew_cpu, Vol, _, _ = inject_sills(Tracers, Tnew_cpu, Grid.coord1D, sill, T_in, 2, nTr_dike)    # Add dike, move hostrocks
            copyto!(Arrays.T, Tnew_cpu)
            InjectVol += Vol                                                                 # Keep track of injected volume
            println("Added new dike; total injected magma volume = $(round(InjectVol / km³, digits = 2)) km³; rate Q=$(round(InjectVol / (time), digits = 2)) m³/s")
        end

        Nonlinear_Diffusion_step!(Arrays, MatParam, Phases, Grid, dt, Num)   # Perform a nonlinear diffusion step

        copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, Arrays.Tnew, Arrays.ϕ)     # Copy arrays to CPU to update properties
        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu)       # Update info on tracers

        Arrays.T .= Arrays.Tnew                                # Update temperature
        time = time + dt                                       # Keep track of evolved time
        Melt_Time[it] = sum(Arrays.ϕ) / prod(Grid.N)                      # Melt fraction in crust
        Time_vec[it] = time                                            # Vector with time
        println(" Timestep $it = $(round(time / kyr * 100) / 100) kyrs")

        if mod(it, 4) == 0  # Visualisation
            x, y, z = Grid.coord1D[1], Grid.coord1D[2], Grid.coord1D[3]
            vtkfile = vtk_grid("./viz3D_out/ex3D_$(Int32(it + 1.0e4))", Vector(x / 1.0e3), Vector(y / 1.0e3), Vector(z / 1.0e3)) # 3-D VTK file
            vtkfile["Temperature"] = Array(Arrays.T); vtkfile["MeltFraction"] = Array(Arrays.ϕ); vtkfile["Phases"] = Array(Phases) # Store fields in file
            outfiles = vtk_save(vtkfile); pvd[time / kyr] = vtkfile                                   # Save file & update pvd file
        end
    end
    vtk_save(pvd)
    return Time_vec, Melt_Time, Tracers, Grid, Arrays
end # end of main function

Time_vec, Melt_Time, Tracers, Grid, Arrays = MainCode_3D(); # start the main code
```

The result of the script is a range of VTK files that can be visualized with ParaView. The full code example and the ParaView Python script that renders the movie (with `pvbatch`) are available at:

- https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/Example3D.jl
- https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/movies/Example3D_Paraview.py

# 2D Example

This page mirrors the 2D snippet from the repository README.

A simple example that simulates emplacement of dikes within the crust over a period of 10'000 years is shown below.

![](../assets/movies/Example2D.gif)

The code to simulate this, including visualization, is <100 lines (if we remove empty ones) and the key parts are shown below.

Note:
- The model arrays live on the KernelAbstractions `backend`; use `CUDABackend()` (after `using CUDA`) to run on an NVIDIA GPU.

```julia
using MagmaThermoKinematics
# using CUDA                        # for an NVIDIA GPU, then: backend = CUDABackend()
backend = CPU()
using CairoMakie

#------------------------------------------------------------------------------------------
@views function MainCode_2D();
Nx,Nz                   =   500,500
Grid                    =   CreateGrid(size=(Nx,Nz), extent=(30e3, 30e3)) # grid points & domain size
Num                     =   Numeric_params(verbose=false)                   # Nonlinear solver options

# Set material parameters
MatParam                =   (
        SetMaterialParams(Name="Rock", Phase=1,
             Density    = ConstantDensity(ρ=2800kg/m^3),
           HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
           Conductivity = ConstantConductivity(k=1.5Watt/K/m),
             LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
                Melting = MeltingParam_Caricchi()),
                            )

GeoT                    =   20.0/1e3;                   # Geothermal gradient [K/km]
W_in, H_in              =   5e3,    0.2e3;              # Width and thickness of dike
T_in                    =   900;                        # Intrusion temperature
InjectionInterval       =   0.1kyr;                     # Inject a new dike every X kyrs
maxTime                 =   25kyr;                      # Maximum simulation time in kyrs
H_ran, W_ran            =   Grid.L.*[0.3; 0.4];         # Size of domain in which we randomly place dikes and range of angles
κ                       =   1.2/(2800*1050);            # thermal diffusivity
dt                      =   minimum(Grid.Δ.^2)/κ/10;    # stable timestep (required for explicit FD)
nt                      =   floor(Int64,maxTime/dt);    # number of required timesteps
nTr_dike                =   300;                        # number of tracers inserted per dike

# Array initializations
Arrays = CreateArrays(Dict( (Nx,  Nz)=>(T=0,T_K=0, T_it_old=0, Rho=2800, Cp=1050, Tnew=0, Tupdate=0, Hr=0, Hl=0, Kc=1, P=0, X=0, Z=0, ϕₒ=0, ϕ=0, dϕdT=0)); backend)
# CPU buffers
Tnew_cpu                =   Matrix{Float64}(undef, Grid.N...)
Phi_melt_cpu            =   similar(Tnew_cpu)
Phases                  =   similar(Arrays.T, Int64); fill!(Phases, 1)

GridArray!(Arrays.X, Arrays.Z, Grid)
Tracers                 =   StructArray{Tracer{Float32}}(undef, 1)                   # Initialize tracers
Arrays.T               .=   -Arrays.Z.*GeoT;                                        # Initial (linear) temperature profile

# Preparation of visualization
x, z        =   Grid.coord1D[1]/1e3, Grid.coord1D[2]/1e3
T_plot, ϕ_plot, title = Observable(Array(Arrays.T)), Observable(Array(Arrays.ϕ)), Observable("0.0 kyrs")
fig         =   Figure(size=(1000,450))
ax1         =   Axis(fig[1,1], aspect=DataAspect(), xlabel="Width [km]", ylabel="Depth [km]", title=title)
Colorbar(fig[1,2], heatmap!(ax1, x, z, T_plot, colormap=:lajolla, colorrange=(0.,900.)), label="Temperature")
ax2         =   Axis(fig[1,3], aspect=DataAspect(), xlabel="Width [km]")
Colorbar(fig[1,4], heatmap!(ax2, x, z, ϕ_plot, colormap=:nuuk, colorrange=(0.,1.)), label="Melt Fraction")
anim        =   VideoStream(fig, framerate=15)

time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
for it = 1:nt   # Time loop

    if floor(time/InjectionInterval)> dike_inj       # Add new dike every X years
        dike_inj  =     floor(time/InjectionInterval)                                               # Keeps track on what was injected already
        cen       =     (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];         # Randomly vary center of dike
        if cen[end]<-12e3;
            Angle_rand = rand( 80.0:0.1:100.0)                                      # Orientation: near-vertical @ depth
        else
            Angle_rand = rand(-10.0:0.1:10.0);
        end                                  # Orientation: near-vertical @ shallower depth
        sill      =     EllipticalIntrusion(Center=Point2(cen[1],cen[2])*m, Angle=Vec1(Angle_rand)*NoUnits, W=W_in*m, H=H_in*m)
        Tnew_cpu .=     Array(Arrays.T)
        Tracers, Tnew_cpu, Vol, _, _   =   inject_sills(Tracers, Tnew_cpu, Grid.coord1D, sill, T_in, 2, nTr_dike);   # Add dike, move hostrocks
        copyto!(Arrays.T, Tnew_cpu)
        InjectVol +=    Vol                                                                 # Keep track of injected volume
        println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(time),digits=2)) m³/s")
    end

    Nonlinear_Diffusion_step!(Arrays, MatParam, Phases, Grid, dt, Num)   # Perform a nonlinear diffusion step

    copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, Arrays.Tnew, Arrays.ϕ)     # Copy arrays to CPU to update properties
    UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers

    Arrays.T .= Arrays.Tnew                                # Update temperature
    time                =   time + dt;                                      # Keep track of evolved time
    Melt_Time[it]       =   sum(Arrays.ϕ)/prod(Grid.N)                      # Melt fraction in crust
    Time_vec[it]        =   time;                                           # Vector with time
    println(" Timestep $it = $(round(time/kyr*100)/100) kyrs")

    if mod(it,20)==0  # Visualization
        T_plot[]    =   Array(Arrays.T)
        ϕ_plot[]    =   Array(Arrays.ϕ)
        title[]     =   "$(round(time/kyr, digits=2)) kyrs"
        recordframe!(anim)
    end
end
save("Example2D.gif", anim)   # create gif animation
return Time_vec, Melt_Time, Tracers, Grid, Arrays;
end # end of main function

Time_vec, Melt_Time, Tracers, Grid, Arrays = MainCode_2D(); # start the main code
save("Time_vs_Melt_Example2D.png", lines(vec(Time_vec/kyr), vec(Melt_Time), axis=(xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten"))) # Create plot
```

The main routines are thus `inject_sills(..)`, which inserts a new dike or sill (of given dimensions and orientation) into the domain using [InjectSills.jl](https://github.com/JuliaGeodynamics/InjectSills.jl), and Nonlinear_Diffusion_step!(...), which computes thermal diffusion. Variable thermal conductivity and latent heat are taken into account.

The full code example is available at:

- https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/examples/Example2D.jl

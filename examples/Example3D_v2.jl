using MagmaThermoKinematics
const USE_GPU=false;
if USE_GPU
    environment!(:gpu, Float64, 3)      # initialize parallel stencil in 2D
else
    environment!(:cpu, Float64, 3)      # initialize parallel stencil in 2D
end
using MagmaThermoKinematics.Diffusion3D
using MagmaThermoKinematics.Fields3D
using Plots
using WriteVTK

#------------------------------------------------------------------------------------------
@views function MainCode_3D();
    Nx,Ny,Nz                =   250,250,250
    Grid                    =   CreateGrid(size=(Nx,Ny,Nz), extent=(30e3, 30e3, 30e3)) # grid points & domain size
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
    W_in, H_in              =   5e3,    0.5e3;              # Width and thickness of dike
    T_in                    =   900;                        # Intrusion temperature
    InjectionInterval       =   0.1kyr;                     # Inject a new dike every X kyrs
    maxTime                 =   15kyr;                      # Maximum simulation time in kyrs
    H_ran, W_ran            =   Grid.L[1]*0.3,Grid.L[3]*0.4;# Size of domain in which we randomly place dikes and range of angles
    DikeType                =   "ElasticDike"               # Type to be injected ("ElasticDike","SquareDike")
    κ                       =   1.2/(2800*1050);            # thermal diffusivity
    dt                      =   minimum(Grid.Δ.^2)/κ/10;    # stable timestep (required for explicit FD)
    nt                      =   floor(Int64,maxTime/dt);    # number of required timesteps
    nTr_dike                =   300;                        # number of tracers inserted per dike

    # Array initializations
    Arrays = CreateArrays(Dict( (Nx,  Ny, Nz)=>(T=0,T_K=0, T_it_old=0, K=1.5, Rho=2800, Cp=1050, Tnew=0,  Hr=0, Hl=0, Kc=1, P=0, X=0, Y=0, Z=0, ϕₒ=0, ϕ=0, dϕdT=0),
                                (Nx-1,Ny,Nz)=>(qx=0,Kx=0), (Nx, Ny-1, Nz)=>(qy=0,Ky=0 ) , (Nx, Ny, Nz-1)=>(qz=0,Kz=0 ) ))
    # CPU buffers
    Tnew_cpu                =   zeros(Float64, Grid.N...)
    Phi_melt_cpu            =   similar(Tnew_cpu)
    if USE_GPU;
        Phases      =   CUDA.ones(Int64,Grid.N...)
    else
        Phases      =   ones(Int64,Grid.N...)
    end

    @parallel (1:Nx,1:Ny,1:Nz) GridArray!(Arrays.X,Arrays.Y,Arrays.Z, Grid.coord1D[1], Grid.coord1D[2], Grid.coord1D[3])
    Tracers                 =   StructArray{Tracer}(undef, 1)                           # Initialize tracers
    dike                    =   Dike(W=W_in,H=H_in,Type=DikeType,T=T_in);               # "Reference" dike with given thickness,radius and T
    Arrays.T               .=   -Arrays.Z.*GeoT;                                        # Initial (linear) temperature profile

    # Preparation of VTK/Paraview output
    if isdir("viz3D_out")==false
        mkdir("viz3D_out")
    end;
    loadpath = "./viz3D_out/";
    pvd = paraview_collection("Example3D");

    time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
    for it = 1:nt   # Time loop

        if floor(time/InjectionInterval)> dike_inj       # Add new dike every X years
            dike_inj  =     floor(time/InjectionInterval)                                               # Keeps track on what was injected already
            cen       =     (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, 3).*[W_ran;W_ran;H_ran];   # Randomly vary center of dike
            if cen[end]<-12e3;
                Angle_rand = [rand(80.0:0.1:100.0); rand(0:360)]                        # Dikes at depth
            else
                Angle_rand = [rand(-10.0:0.1:10.0); rand(0:360)]
            end                    # Sills at shallower depth
            dike      =     Dike(dike, Center=cen[:],Angle=Angle_rand);                                 # Specify dike with random location/angle but fixed size/T
            Tnew_cpu .=     Array(Arrays.T)
            Tracers, Tnew_cpu, Vol   =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, nTr_dike);   # Add dike, move hostrocks
            Arrays.T .=     Data.Array(Tnew_cpu)
            InjectVol +=    Vol                                                                 # Keep track of injected volume
            println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(time),digits=2)) m³/s")
        end

        Nonlinear_Diffusion_step_3D!(Arrays, MatParam, Phases, Grid, dt, Num)   # Perform a nonlinear diffusion step

        copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, Arrays.Tnew, Arrays.ϕ)     # Copy arrays to CPU to update properties
        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers

        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)                                # Update temperature
        time                =   time + dt;                                      # Keep track of evolved time
        Melt_Time[it]       =   sum(Arrays.ϕ)/prod(Grid.N)                      # Melt fraction in crust
        Time_vec[it]        =   time;                                           # Vector with time
        println(" Timestep $it = $(round(time/kyr*100)/100) kyrs")

        if mod(it,20)==0  # Visualisation
            x,y,z         =   Grid.coord1D[1], Grid.coord1D[2], Grid.coord1D[3]
            vtkfile = vtk_grid("./viz3D_out/ex3D_$(Int32(it+1e4))", Vector(x/1e3), Vector(y/1e3), Vector(z/1e3)) # 3-D VTK file
            vtkfile["Temperature"] = Array(Arrays.T); vtkfile["MeltFraction"] = Array(Arrays.ϕ);                 # Store fields in file
            outfiles = vtk_save(vtkfile); pvd[time/kyr] = vtkfile                                   # Save file & update pvd file
        end
    end
    vtk_save(pvd)
    return Time_vec, Melt_Time, Tracers, Grid, Arrays;
end # end of main function

Time_vec, Melt_Time, Tracers, Grid, Arrays = MainCode_3D(); # start the main code
plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") # Create plot

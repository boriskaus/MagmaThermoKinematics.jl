using MagmaThermoKinematics
const USE_GPU=false;
if USE_GPU  environment!(:gpu, Float64, 2)      # initialize parallel stencil in 2D
else        environment!(:cpu, Float64, 2)      # initialize parallel stencil in 2D
end
using MagmaThermoKinematics.Diffusion2D # to load AFTER calling environment!()
using Plots                                     

#------------------------------------------------------------------------------------------
@views function MainCode_2D();

    Grid                    =   CreateGrid(size=(500,500), extent=(30e3, 30e3)) # grid points & domain size
    Num                     =   Numeric_params(verbose=false)                   # Nonlinear solver options

    # Set material parameters                                       
    MatParam                =   (
            SetMaterialParams(Name="Rock", Phase=1, 
                 Density    = ConstantDensity(ρ=2800kg/m^3),               
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
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
    DikeType                =   "ElasticDike"               # Type to be injected ("ElasticDike","SquareDike")
    κ                       =   1.2/(2800*1050);            # thermal diffusivity   
    dt                      =   minimum(Grid.Δ.^2)/κ/10;    # stable timestep (required for explicit FD)
    nt                      =   floor(Int64,maxTime/dt);    # number of required timesteps
    nTr_dike                =   300;                        # number of tracers inserted per dike

    # Array initializations
    Arrays = CreateArrays(Dict( (Grid.N[1],  Grid.N[2])=>(T=0,T_K=0, T_it_old=0, K=1.5, Rho=2800, Cp=1050, Tnew=0,  Hr=0, Hl=0, Kc=1, P=0, X=0, Z=0, ϕₒ=0, ϕ=0, dϕdT=0),
                                (Grid.N[1]-1,Grid.N[2])=>(qx=0,Kx=0), (Grid.N[1], Grid.N[2]-1)=>(qz=0,Kz=0 ) ))
    # CPU buffers 
    Tnew_cpu                =   Matrix{Float64}(undef, Grid.N...)
    Phi_melt_cpu            =   similar(Tnew_cpu)
    if USE_GPU; Phases      =   CUDA.ones(Int64,Grid.N...)
    else        Phases      =   ones(Int64,Grid.N...)   end

    GridArray!(Arrays.X,  Arrays.Z, Grid.coord1D[1], Grid.coord1D[2])   
    Tracers                 =   StructArray{Tracer}(undef, 1)                           # Initialize tracers   
    dike                    =   Dike(W=W_in,H=H_in,Type=DikeType,T=T_in);               # "Reference" dike with given thickness,radius and T
    Arrays.T               .=   -Arrays.Z.*GeoT;                                        # Initial (linear) temperature profile

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])

    time, dike_inj, InjectVol, Time_vec,Melt_Time = 0.0, 0.0, 0.0,zeros(nt,1),zeros(nt,1);
    for it = 1:nt   # Time loop

        if floor(time/InjectionInterval)> dike_inj       # Add new dike every X years
            dike_inj  =     floor(time/InjectionInterval)                                               # Keeps track on what was injected already
            cen       =     (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];         # Randomly vary center of dike 
            if cen[end]<-12e3;  Angle_rand = rand( 80.0:0.1:100.0)                                      # Orientation: near-vertical @ depth             
            else                Angle_rand = rand(-10.0:0.1:10.0); end                                  # Orientation: near-vertical @ shallower depth     
            dike      =     Dike(dike, Center=cen[:],Angle=[Angle_rand]);                               # Specify dike with random location/angle but fixed size/T 
            Tnew_cpu .=     Array(Arrays.T)
            Tracers, Tnew_cpu, Vol   =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, nTr_dike);   # Add dike, move hostrocks
            Arrays.T .=     Data.Array(Tnew_cpu)
            InjectVol +=    Vol                                                                 # Keep track of injected volume
            println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(time),digits=2)) m³/s")
        end

        Nonlinear_Diffusion_step_2D!(Arrays, MatParam, Phases, Grid, dt, Num)   # Perform a nonlinear diffusion step

        copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, Arrays.Tnew, Arrays.ϕ)     # Copy arrays to CPU to update properties
        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers 

        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)                                # Update temperature
        time                =   time + dt;                                      # Keep track of evolved time
        Melt_Time[it]       =   sum(Arrays.ϕ)/prod(Grid.N)                      # Melt fraction in crust    
        Time_vec[it]        =   time;                                           # Vector with time
        println(" Timestep $it = $(round(time/kyr*100)/100) kyrs")

        if mod(it,20)==0  # Visualisation
            x,z         =   Grid.coord1D[1], Grid.coord1D[2]
            p1          =   heatmap(x/1e3, z/1e3, Array(Arrays.T)',  aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:lajolla, clims=(0.,900.), xlabel="Width [km]",ylabel="Depth [km]", title="$(round(time/kyr, digits=2)) kyrs", dpi=200, fontsize=6, colorbar_title="Temperature")
            p2          =   heatmap(x/1e3,z/1e3, Array(Arrays.ϕ)',  aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:nuuk,    clims=(0., 1. ), xlabel="Width [km]",             dpi=200, fontsize=6, colorbar_title="Melt Fraction")
            plot(p1, p2, layout=(1,2)); frame(anim)
        end
    end
    gif(anim, "Example2D.gif", fps = 15)   # create gif animation
    return Time_vec, Melt_Time;
end # end of main function

Time_vec,Melt_Time = MainCode_2D(); # start the main code
plot(Time_vec/kyr, Melt_Time, xlabel="Time [kyrs]", ylabel="Fraction of crust that is molten", label=:none); png("Time_vs_Melt_Example2D") # Create plot
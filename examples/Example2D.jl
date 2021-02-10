using ZirconThermoKinematics
using ZirconThermoKinematics.Diffusion2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots                                     # plotting; install with ], add Plots


# Initialize 
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D



# Benchmarks tests to be included in the package : 
#   1)  Gaussian diffusion (2D, 3D)
#   2)  Steady state with variable k (2D,3D, different directions)
#   3)  Halfspace cooling
#   4)  Advection with rotational velocity field 


#------------------------------------------------------------------------------------------
@views function MainCode_Axisymmetric();

# Model parameters
W,H     =   50, 30;                             # Width, Height in km
ρ       =   2800;                               # Density 
cp      =   1050;                               # Heat capacity
k       =   3;                                  # Thermal conductivity
L       =   350e3;                              # 
GeoT    =   20;                                 # Geothermal gradient [K/km]

z_in    =   -15e3;                              # z-center intrusion  [m]
x_in    =   20e3;                               # x-center intrusion  [m]

H_in    =   1e3;                                # intrusion height [m]
W_in    =   20e3;                               # Width intrusion  [m]
T_in    =   900;                                # intrusion temperature

# Numerics
Nx, Nz  =   200, 200;                            # resolution
nt      =   200;                                 # number of timesteps
dr      =   W/(Nx-1)*1e3; dz = H*1e3/(Nz-1);     # grid size [m]
κ       =   k./(ρ*cp);                           # diffusivity   
dt      =   min(dr^2, dz^2)./κ/10;               # stable timestep (required for explicit FD)
nTr_dike=   1000;                                # number of tracers/dike

# Array initializations
T       =   @zeros(Nx,   Nz);                    # 
Tnew    =   @zeros(Nx,   Nz);
qr      =   @zeros(Nx-1, Nz);
qz      =   @zeros(Nx,   Nz-1);
K       =   @ones(Nx,    Nz)*k;
Kr      =   @zeros(Nx-1, Nz);
Kz      =   @zeros(Nx,   Nz-1);
Rho     =   @ones(Nx,    Nz)*ρ;       
Cp      =   @ones(Nx,    Nz)*cp;
Vr      =   @zeros(Nx,   Nz);
Vz      =   @zeros(Nx,   Nz);
R       =   @zeros(Nx,   Nz);
Rc      =   @zeros(Nx-1, Nz-1);
Z       =   @zeros(Nx,   Nz);
Phi_o   =   @zeros(Nx,   Nz);
Phi     =   @zeros(Nx,   Nz);
dPhi_dt =   @zeros(Nx,   Nz);

# Set up model geometry
r,z     =   0:dr:W*1e3, -H*1e3:dz:(-H*1e3+(Nz-1)*dz);
R,Z     =   ones(Nz)' .* r, z' .* ones(Nx);                         # 2D coordinate grids
Rc      =   (R[2:Nx,:]+R[1:Nx-1,:])/2.0;
Grid    =   (R,Z);

# initial temperature structure
T      .=   -Z./1e3.*GeoT;                                          # initial T profile
Tnew   .=   T;                                                      # To get correct boundary conditions.

# Create dike polygon 
dike        =   Dike(W_in,H_in,T_in,x_in,z_in,20, "SquareDike");    # Specify dike  (can vary throughout simulation)
Tracers     =   StructArray([Tracer(1,dike.x0,dike.z0,dike.T)]);    # Initialize tracer array
Velocity    =   HostRockVelocityFromDike( Grid,dt,dike);            # compute velocity field
SolidFraction(T, Phi, Phi_o, dPhi_dt, dt);                          # Compute solid fraction

# Preparation of visualisation
ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
println("Animation directory: $(anim.dir)")

# Time loop
time = 0;
for it = 1:nt
    global dike_poly;


    
    if (it>50) & (it<60)
        # inject a new Dike
        # Create new dike
        dike        =   Dike(W_in,H_in,T_in,x_in,z_in,0, "SquareDike");      # Specify dike  (can vary throughout simulation)    

        # Compute velocity required to create space for dike
        Velocity    =   HostRockVelocityFromDike( Grid,dt*10,dike);           # compute velocity field
        
        # Move hostrock & already existing tracers
        Tnew        =   AdvectTemperature(T,        Grid,  Velocity,   (dr,dz),    dt);    
        Tracers     =   AdvectTracers(Tracers, Tnew,Grid,  Velocity,   (dr,dz),    dt);
        T           =   Tnew;

        # Insert dike in T profile and add new tracers
        T,Tracers, dike_poly  =   AddDike(T, Tracers, Grid,dike, nTr_dike);                 # Add dike to T-field & insert tracers within dike

    else
        Velocity    =    map(x->x.*0, Velocity) ;    # set to 0
     
        # Advect tracers (and update T on them)
        Tracers = AdvectTracers(Tracers, T,Grid,  Velocity,   (dr,dz),    dt);

    end

    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")

    # Perform a diffusion step
   # @parallel diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, K, Kr, Kz, Rho, Cp, dt, dr, dz,  L, dPhi_dt);   # axi-symmetric
    @parallel diffusion2D_step!(Tnew, T, qr, qz, K, Kr, Kz, Rho, Cp, dt, dr, dz,  L, dPhi_dt);                 # 2D    
    @parallel (1:size(T,2)) bc2D_x!(Tnew); # set BC's

   
    T,Tnew = Tnew,T;
    
    # Compute solid fraction    
    SolidFraction(T, Phi, Phi_o, dPhi_dt, dt);                        # Compute solid fraction

    # Inject new tracers within the dike area
    # Tr          =   InjectNewTracers(Tr,Poly_in, t_kyrs);
    # Poly_in     =   CreateIntrusionZonePolygon(W_in*1e3,H_in*0.01 , z_in*1e3);    % recreate inner poly

    # Update arrays
    # T, Tnew = Tnew, T;
    time    = time + dt;

    if mod(it,10)==0
        # Visualisation
        Phi_melt    =   1.0 .- Phi; 
        time_kyrs   =   round(time/SecYear/1e3);
        r_km, z_km  =   r./1e3, z./1e3;
        p1          =   heatmap(r_km, z_km, T',  aspect_ratio=1, xlims=(r_km[1],r_km[end]), ylims=(z_km[1],z_km[end]),  c=:inferno, title="Temperature, $time_kyrs kyrs", dpi=300)
    #    plot!(dike_poly.x/1e3, dike_poly.z/1e3,linecolor=:black,legend=false,linewidth=0.5)    
        scatter!(Tracers.x/1e3, Tracers.z/1e3, zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), markersize=1.0)


        # make the quiver 'look nice', takes a few lines
        Vr = Velocity[1];
        Vz = Velocity[2];
        

        step_x = Int8(Nx/20); step_z = Int8(Nz/20);
        R_km = R[1:step_x:end,1:step_z:end]./1e3;
        Z_km = Z[1:step_x:end,1:step_z:end]./1e3;
        Vel = sqrt.(Vr.^2 .+ Vz.^2);
        maxVel = maximum(Vel[:])*0.4;

        Vr_n = Vr[1:step_x:end,1:step_z:end]./maxVel; 
        Vz_n = Vz[1:step_x:end,1:step_z:end]./maxVel; 
        
        #quiver!(R_km[:], Z_km[:], gradient=(Vr_n[:],Vz_n[:]), arrow = :arrow)
        
        p2          =   heatmap(r_km, z_km, Phi_melt', aspect_ratio=1, xlims=(r_km[1],r_km[end]), ylims=(z_km[1],z_km[end]), c=:jet, title="Melt fraction", xlabel="Width [km]", dpi=300)
#        p2          =   heatmap(r_km, z_km, Vz', aspect_ratio=1, xlims=(r_km[1],r_km[end]), ylims=(z_km[1],z_km[end]), c=:jet, title="Melt fraction", xlabel="Width [km]", dpi=300)

        quiver!(R_km[:], Z_km[:], gradient=(Vr_n[:],Vz_n[:]), arrow = :arrow)
        
        plot(p1, p2, layout=(2,1)); frame(anim)

#        plot(p3); frame(anim)
        
    end



end
gif(anim, "Temp2D.gif", fps = 15)

end # end of main function



MainCode_Axisymmetric()

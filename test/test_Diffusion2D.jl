using MagmaThermoKinematics

using ParallelStencil
ParallelStencil.@reset_parallel_stencil()
environment!(:cpu, Float64, 2) 
using MagmaThermoKinematics.Diffusion2D

using Plots  
using LinearAlgebra
using SpecialFunctions
using Test

# Initialize for multiple threads (GPU is not tested here)
@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D

const CreatePlots = false      # easy way to deactivate plotting throughout

function Diffusion_SteadyState2D(Setup="Constant_Zdirection")
# steady state diffusion in x and z-direction for constant and variable K

# Model parameters
if      Setup=="Constant_Zdirection" || Setup=="VariableK_Zdirection"
    W,H                 =   3, 30;                              # Width, Height in km
    k_rock1             =   3;
    k_rock2             =   1.5;
    Nx, Nz              =   10, 100;                            # resolution
    Tbot                =   600;
    H1                  =   20;
    GeoT                =   Tbot/H;
elseif  Setup=="Constant_Xdirection" || Setup=="VariableK_Xdirection"
    W,H                 =   40, 3;                              # Width, Height in km
    k_rock1             =   3;
    k_rock2             =   1.5;
    Nx, Nz              =   100, 10;                            # resolution
    Tbot                =   600;
    W1                  =   20;
    GeoT                =   Tbot/W;
else
    error("unknown setup")
end

SecYear                 =   365.25*24*3600
ρ                       =   2800;                               # Density 
cp                      =   1050;                               # Heat capacity
L                       =   350e3;                              # Latent heat J/kg/K
dx                      =   W/(Nx-1)*1e3; dz = H*1e3/(Nz-1);    # grid size [m]
κ                       =   k_rock1./(ρ*cp);                     # thermal diffusivity   
dt                      =   min(dx^2, dz^2)./κ/10;             # stable timestep (required for explicit FD)

# Array initializations (1 - main arrays on which we can initialize properties)
T                       =   @zeros(Nx,   Nz);                    
K                       =   @ones(Nx,    Nz)*k_rock1;
Rho                     =   @ones(Nx,    Nz)*ρ;       
Hs                      =   @zeros(Nx,   Nz);   
Hl                      =   @ones(Nx,   Nz)*L;   
Cp                      =   @ones(Nx,    Nz)*cp;
dPhi_dt                 =   @zeros(Nx,   Nz);    

# Work array initialization
Tnew, qx,qz, Kx, Kz     =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
X,Xc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints

# Set up model geometry & initial T structure
x,z                     =   0:dx:W*1e3, -H*1e3:dz:(-H*1e3+(Nz-1)*dz);
X,Z                     =   ones(Nz)' .* x, z' .* ones(Nx);                             # 2D coordinate grids
Xc                      =   (X[2:Nx,:] + X[1:Nx-1,:])/2.0;
Grid, Spacing           =   (X,Z), (dx,dz);
T                       .=   Tbot;                                             # initial (linear) temperature profile

if      Setup=="Constant_Zdirection" || Setup=="VariableK_Zdirection"
    # bottom BC
    T[1,:]              .=   0;
else
    T[:,1]              .=   0;
end


if Setup=="VariableK_Zdirection"
    # variable k
    K[Z.<-H1*1e3] .= k_rock2;


elseif Setup=="VariableK_Xdirection"
    # variable k
    K[X.< W1*1e3] .= k_rock2;
end


time,time_kyrs, dike_inj = 0.0, 0.0, 0.0;
err = 100;
it = 0;
while (err>1e-10) & (it<1e6)

    it += 1
    # Perform a diffusion step
    @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, Hs, Hl, dt, dx, dz, dPhi_dt);  
    if Setup=="Constant_Zdirection" || Setup=="VariableK_Zdirection"
         # diffusion in z-direction
        @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                      # set lateral boundary conditions (flux-free)
        Tnew[:,1] .= Tbot; Tnew[:,end] .= 0.0;                                                    # bottom & top temperature (constant)

    else
        # diffusion in x-direction
        @parallel (1:size(T,1)) bc2D_z!(Tnew);                                                      # set lateral boundary conditions (flux-free)
        Tnew[1,:] .= 0; Tnew[end,:] .= Tbot;                                                    # bottom & top temperature (constant)
    end
    
    err        =   maximum(abs.(Tnew-T));
    
    # Tracers         =   UpdateTracers(Tracers, Grid, Spacing, Tnew, Phi);                       # Update info on tracers 
    T, Tnew         =   Tnew, T;                                                                # Update temperature
    time, time_kyrs =   time + dt, time/SecYear/1e3;                                            # Keep track of evolved time

    if mod(it,10000)==0  # Visualisation           
    #    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")
    end

end

x_km, z_km  =   x./1e3, z./1e3;


# compute analytical solution
if  Setup=="Constant_Zdirection"        
    Tanal = -z_km.*GeoT;
    Tnum  = T[1,:];
    fname = "Diffusion_2D_SS_constantK_Z";

elseif  Setup=="VariableK_Zdirection"
    # 1D steady steate analytical solution for variable K is given by the folliwing balance equations
    #   k1*dT/dz|_1                     =   k2*dT/dz|_2     (heat flux)
    #   dT/dz|_1 * H1 + dT/dz|_2 * H2 =   Tbot            (assuming Ttop=0)
    #   H1 + H2                         =   H               (total thickness)
    #
    # substitute eq. 1 into eq 2 to eliminate dT/dz|_2
    #  dT/dz|_1 * H_1 + dT/dz|_1 * k1/k2*H2 =   Tbot           
    #  dT/dz|_1 = Tbot/(H1 + k1/k2*H2)

    H2      =   H - H1;
    dTdz1   =   Tbot/(H1 + k_rock1/k_rock2*H2)
    dTdz2   =   k_rock1/k_rock2.*dTdz1
    
    Tanal   =   zeros(size(z_km));              Tanal2  =   copy(Tanal);
    Tanal   .=   -z_km[:].*dTdz1;
    Tanal2  .=   -z_km.*dTdz2 .- dTdz1*H1;
    Tanal[z_km .< -H1 ] .= Tanal2[z_km .< -H1 ]

    Tnum    =    T[1,:];
    fname = "Diffusion_2D_SS_variableK_Z";

elseif  Setup=="Constant_Xdirection"  
    Tanal = x_km.*GeoT;
    Tnum  = T[:,1];
    fname = "Diffusion_2D_SS_constantK_X";

elseif  Setup=="VariableK_Xdirection"
    W2      =   W - W1;
    dTdx1   =   Tbot/(W1 + k_rock1/k_rock2*W2)
    dTdx2   =   k_rock1/k_rock2.*dTdx1
    
    Tanal   =   zeros(size(x_km));              Tanal2  =   copy(Tanal);
    Tanal  .=   x_km.*dTdx2 ;
    Tanal2  .=   x_km[:].*dTdx1 .+ dTdx1*W2;
    Tanal[x_km .> W1 ] .= Tanal2[x_km .> W1 ]

    Tnum    =    T[:,1];
    fname = "Diffusion_2D_SS_variableK_X";
end

if      Setup=="Constant_Zdirection" || Setup=="VariableK_Zdirection"
    if CreatePlots
        # create plot 
        plot(Tanal,z_km, label = "Analytics");
        plot!(Tnum,z_km,ylabel="Depth [km]",xlabel="Temperature [C]", label = "Numerics",  marker = 2,   linewidth = 0);
    end
    error = norm(T[1,:] .- Tanal,2); 
else
    if CreatePlots
        # create plot 
        plot(x_km, Tanal,label = "Analytics");
        plot!(x_km,Tnum, xlabel="Width [km]",ylabel="Temperature [C]", label = "Numerics",  marker = 2,   linewidth = 0);
    end
    error = norm(T[:,1] .- Tanal,2); 
end
if CreatePlots
    png(fname)
end


return error;        # return error

end # end of steady state diffusion test

function Diffusion_Halfspace2D()
    # Halfspace cooling example 
    
    # Model parameters
    W,H                     =   3, 300;                             # Width, Height in km
    k_rock1                 =   3;
    Nx, Nz                  =   10, 100;                            # resolution
    Tbot                    =   1200;
    SecYear                 =   365.25*24*3600
    CoolingAge              =   30e6*SecYear;                       # thermal cooling age
    ρ                       =   2800;                               # Density 
    cp                      =   1050;                               # Heat capacity
    L                       =   350e3;                              # Latent heat J/kg/K
    dx                      =   W/(Nx-1)*1e3; dz = H*1e3/(Nz-1);    # grid size [m]
    κ                       =   k_rock1./(ρ*cp);                     # thermal diffusivity   
    dt                      =   min(dx^2, dz^2)./κ/10;             # stable timestep (required for explicit FD)
    
    numTime                 =   ceil(CoolingAge/dt);
    dt                      =   CoolingAge/numTime;
    nt                      =   Int(numTime);

    # Array initializations (1 - main arrays on which we can initialize properties)
    T                       =   @ones(Nx,    Nz)*Tbot;                    
    K                       =   @ones(Nx,    Nz)*k_rock1;
    Rho                     =   @ones(Nx,    Nz)*ρ;       
    Cp                      =   @ones(Nx,    Nz)*cp;
    Hs                      =   @zeros(Nx,   Nz);  
    Hl                      =   @zeros(Nx,   Nz)*L;  
    dPhi_dt                 =   @zeros(Nx,   Nz);    
    
    # Work array initialization
    Tnew, qx,qz, Kx, Kz     =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
    X,Xc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints
    
    # Set up model geometry & initial T structure
    x,z                     =   0:dx:W*1e3, -H*1e3:dz:(-H*1e3+(Nz-1)*dz);
    X,Z                     =   ones(Nz)' .* x, z' .* ones(Nx);                             # 2D coordinate grids
    Xc                      =   (X[2:Nx,:] + X[1:Nx-1,:])/2.0;
    Grid, Spacing           =   (X,Z), (dx,dz);
    T                       .=   Tbot;                                             # initial (linear) temperature profile
    
    T[:,end]               .=   0;     # top BC
    Tnew                   .= T;
    
    time,time_kyrs, dike_inj = 0.0, 0.0, 0.0;
    err = 100;
    it = 0;
    for it=1:nt
    
        # Perform a diffusion step
        @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, Hs, Hl, dt, dx, dz, dPhi_dt);  
        # diffusion in z-direction
        @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                      # set lateral boundary conditions (flux-free)
        Tnew[:,1] .= Tbot; Tnew[:,end] .= 0.0;                                                    # bottom & top temperature (constant)
    
        
        
        # Tracers         =   UpdateTracers(Tracers, Grid, Spacing, Tnew, Phi);                       # Update info on tracers 
        T, Tnew         =   Tnew, T;                                                                # Update temperature
        time, time_kyrs =   time + dt, time/SecYear/1e3;                                            # Keep track of evolved time
    
        if mod(it,1000)==0  # print progress      
        #    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")
        end
    
    end
    
    x_km, z_km  =   x./1e3, z./1e3;
    
    
    # compute analytical solution
    Tanal = -z_km;

    Tanal      =   (0-Tbot).*erfc.((abs.(z_km).*1e3)./(2*sqrt(κ*CoolingAge))) .+ Tbot;


    Tnum  = T[1,:];
    fname = "Diffusion_2D_Halfspace";
    
    if CreatePlots
        # create plot 
        plot(Tanal,z_km, label = "Analytics");
        plot!(Tnum,z_km,ylabel="Depth [km]",xlabel="Temperature [C]", label = "Numerics",  marker = 2,   linewidth = 0);
    
        png(fname)
    end
    
    error = norm(T[1,:] .- Tanal,2); 
    return error;        # return error
    
end # end of halfspace cooling test
    

function Diffusion_Gaussian2D(Setup="2D")
    # Gaussian diffusion test in 2D
    
    # Model parameters
    W,H                     =   300, 300;                           # Width, Height in km
    k_rock1                 =   3;
    Nx, Nz                  =   100, 100;                           # resolution
    Tbot                    =   0;
    SecYear                 =   365.25*24*3600
    σ                       =   15e3;                               # halfwidth of gaussian
    Tmax                    =   1000;                                # max. of gaussian
    TotalTime               =   3e6*SecYear;                        # thermal cooling age
    ρ                       =   2800;                               # Density 
    cp                      =   1050;                               # Heat capacity
    L                       =   350e3;                              # Latent heat J/kg/K
    dx                      =   W/(Nx-1)*1e3; dz = H*1e3/(Nz-1);    # grid size [m]
    κ                       =   k_rock1./(ρ*cp);                    # thermal diffusivity   
    dt                      =   min(dx^2, dz^2)./κ/10;              # stable timestep (required for explicit FD)
    
    numTime                 =   ceil(TotalTime/dt);
    dt                      =   TotalTime/numTime/100;
    nt                      =   Int(numTime);

    # Array initializations (1 - main arrays on which we can initialize properties)
    T                       =   @ones(Nx,    Nz)*Tbot;                    
    K                       =   @ones(Nx,    Nz)*k_rock1;
    Rho                     =   @ones(Nx,    Nz)*ρ;       
    Cp                      =   @ones(Nx,    Nz)*cp;
    dPhi_dt                 =   @zeros(Nx,   Nz);    
    Hs                      =   @zeros(Nx,   Nz);  
    Hl                      =   @zeros(Nx,   Nz);  
    
    # Work array initialization
    Tnew, qx,qz, Kx, Kz     =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz),     @zeros(Nx,   Nz-1), @zeros(Nx-1, Nz), @zeros(Nx,   Nz-1)    # thermal solver
    X,Xc,Z                  =   @zeros(Nx,   Nz), @zeros(Nx-1, Nz-1),   @zeros(Nx,   Nz)    # 2D gridpoints
    
    # Set up model geometry & initial T structure
    x,z                     =   -W/2*1e3:dx:W/2*1e3, -H/2*1e3:dz:(-H/2*1e3+(Nz-1)*dz);
    X,Z                     =   ones(Nz)' .* x, z' .* ones(Nx);                             # 2D coordinate grids
    Xc                      =   (X[2:Nx,:] + X[1:Nx-1,:])/2.0;
    Grid, Spacing           =   (X,Z), (dx,dz);
    
    if Setup=="2D"
        T                  .=  Tmax.*exp.(  -(X.^2 .+ Z.^2)./(σ^2));                     # initial gaussian profile
    elseif Setup=="Axisymmetric"
        T                  .=  Tmax.*exp.(  -(X.^2 .+ Z.^2)./(σ^2));                     # initial gaussian profile
    else
        error("Unknown setup")
    end

    Tnew                   .=   T;
    
    #ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    #println("Animation directory: $(anim.dir)")

    time,time_kyrs          = 0.0, 0.0;
    err = 100;
    it = 0;
    nt = 500
    for it=1:nt
    
        # Perform a diffusion step
        if Setup=="2D"
            @parallel diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, Hs, Hl, dt, dx, dz,  dPhi_dt);  
        elseif Setup=="Axisymmetric"
            @parallel diffusion2D_AxiSymm_step!(Tnew, T, X, Xc, qx, qz, K, Kx, Kz, Rho, Cp, Hs, Hl, dt, dx, dz,  dPhi_dt);  
        end

        # diffusion in z-direction
        @parallel (1:size(T,2)) bc2D_x!(Tnew);                                                      # set lateral boundary conditions (flux-free)
        Tnew[:,1] .= Tbot; Tnew[:,end] .= 0.0;                                                    # bottom & top temperature (constant)
    
        
        T, Tnew         =   Tnew, T;                                                                # Update temperature
        time, time_kyrs =   time + dt, time/SecYear/1e3;                                            # Keep track of evolved time
    
        if mod(it,50)==0  # print progress      
        #    println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")

        #    x_km, z_km  =   x./1e3, z./1e3;
        #    p1          =   heatmap(x_km, z_km, T',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="Temperature, $(round(time_kyrs, digits=2)) kyrs",  dpi=150)
        #    plot(p1); frame(anim)
        end
    
    end
    
    x_km, z_km  =   x./1e3, z./1e3;
    
    
    # compute analytical solution
    
    if Setup=="2D"
        Tanal  =  Tmax./(1 + 4*time*κ/σ^2)^(2/2).*exp.(  -(X.^2 .+ Z.^2)./(σ^2 + 4*time*κ));                     # initial gaussian profile
        fname = "Diffusion_2D_Gaussian";
    
    elseif Setup=="Axisymmetric"
        # Axisymmetric is like 3D, where Y=X. Hence we can use the 3D solution, which is
        Tanal  =  Tmax./( (1 + 4*time*κ/σ^2)^(3/2) ).*exp.(  -(X.^2 .+ Z.^2)./(σ^2 + 4*time*κ));                     # initial gaussian profile
        fname = "Diffusion_Axisymmetric_Gaussian";
    end
    Terror =  T - Tanal

    if CreatePlots
        # create plot 
        p1          =   heatmap(x_km, z_km, Terror',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="T error 2D $(round(time_kyrs/1e3, digits=2)) Myrs",  dpi=150)
        plot(p1);
        png(fname)
    end
    
    error = norm(Terror[:],2); 
    
    return error;        # return error
    
end # end of gaussian diffusion test

# Create a range of 2D diffusion tests which calls the routines above
@testset "2D steady state diffusion" begin
    @test Diffusion_SteadyState2D("VariableK_Zdirection") ≈ 8.749  atol=1e-3;
    @test Diffusion_SteadyState2D("Constant_Zdirection")  ≈ 1e-5   atol=1e-4;
    @test Diffusion_SteadyState2D("VariableK_Xdirection") ≈ 2.041  atol=1e-3;
    @test Diffusion_SteadyState2D("Constant_Xdirection")  ≈ 1e-4   atol=1e-4;
end;

@testset "2D halfspace cooling" begin
    @test  Diffusion_Halfspace2D() ≈ 0.6247443434361517 atol=1e-5; 
end;
@testset "2D Gaussian diffusion" begin
    @test Diffusion_Gaussian2D("2D")           ≈  5.688521713402446 atol=1e-3;
    @test Diffusion_Gaussian2D("Axisymmetric") ≈ 11.361627247736031 atol=1e-3;
end;


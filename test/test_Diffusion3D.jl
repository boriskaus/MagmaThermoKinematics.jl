using MagmaThermoKinematics
using ParallelStencil
ParallelStencil.@reset_parallel_stencil()
environment!(:cpu, Float64, 3) 
using MagmaThermoKinematics.Diffusion3D

using Plots  
using LinearAlgebra
using SpecialFunctions
using Test

const CreatePlots = false      # easy way to deactivate plotting throughout

# Initialize for multiple threads (GPU is not tested here)
@init_parallel_stencil(Threads, Float64, 3);    # initialize parallel stencil in 3D

function Diffusion_Gaussian3D(Setup="3D")
    # Test the 
    
    # Model parameters
    W,L,H                   =   300., 300., 300.;                               # Width, Length,    Height in km
    k_rock1                 =   3;
    Nx, Ny, Nz              =   100, 100, 100;                                  # resolution
    Tbot                    =   0;
    SecYear                 =   365.25*24*3600
    σ                       =   15e3;                                           # halfwidth of gaussian
    Tmax                    =   1000;                                           # max. of gaussian
    TotalTime               =   3e6*SecYear;                                    # thermal cooling age
    ρ                       =   2800;                                           # Density 
    cp                      =   1050;                                           # Heat capacity
    La                      =   350e3;                                          # Latent heat J/kg/K
    dx,dy,dz                =   W/(Nx-1)*1e3, L/(Ny-1)*1e3, H*1e3/(Nz-1);       # grid size [m]
    κ                       =   k_rock1./(ρ*cp);                                # thermal diffusivity   
    dt                      =   min(dx^2,dy^2,dz^2)./κ/2;                       # stable timestep (required for explicit FD)
    
    numTime                 =   ceil(TotalTime/dt);
    dt                      =   TotalTime/numTime/20;
    nt                      =   Int(numTime);

    # Array initializations (1 - main arrays on which we can initialize properties)
    T                       =   @ones(Nx,Ny,Nz)*Tbot;                    
    K                       =   @ones(Nx,Ny,Nz)*k_rock1;
    Rho                     =   @ones(Nx,Ny,Nz)*ρ;       
    Cp                      =   @ones(Nx,Ny,Nz)*cp;
    dPhi_dt                 =   @zeros(Nx,Ny,Nz);    
    Hs                      =   @zeros(Nx,Ny,Nz);    
    Hl                      =   @ones(Nx,Ny,Nz)*La;    
    
    # Work array initialization
    Tnew, qx,qy,qz          =   @zeros(Nx,Ny,Nz),       @zeros(Nx-1,Ny,Nz),     @zeros(Nx, Ny-1, Nz),   @zeros(Nx,Ny,Nz-1)  # thermal solver
    Kx, Ky, Kz              =                           @zeros(Nx-1, Ny, Nz),   @zeros(Nx,Ny-1,Nz),     @zeros(Nx,Ny,Nz-1)  # thermal conductivities
    X,Y,Z                   =   @zeros(Nx,Ny,Nz),       @zeros(Nx,Ny,Nz),       @zeros(Nx,Ny,Nz)                            # 3D gridpoints
    
    # Set up model geometry & initial T structure
    x,y,z                   =   -W/2*1e3:dx:(-W/2*1e3+(Nx-1)*dx), -L/2*1e3:dy:(-L/2*1e3+(Ny-1)*dy), -H/2*1e3:dz:(-H/2*1e3+(Nz-1)*dz);
    coords                  =   collect(Iterators.product(x,y,z))                               # generate coordinates from 1D coordinate vectors   
    X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);     # transfer coords to 3D arrays
    Grid, Spacing           =   (X,Y,Z), (dx,dy,dz);
    T                      .=   Tmax.*exp.(  -((X.^2 .+ Y.^2 .+ Z.^2)./(σ^2)) );                 # initial gaussian profile
    Tnew                   .=   T;
  
    #ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    #println("Animation directory: $(anim.dir)")

    time,time_kyrs          = 0.0, 0.0;
    err = 100;
    it = 0;
    nt = 500
    for it=1:nt
    
        # Perform a diffusion step
        if Setup=="3D"
            @parallel diffusion3D_step_varK!(Tnew, T, qx, qy, qz, K, Kx, Ky, Kz, Rho, Cp, Hs, Hl, dt, dx, dy, dz, dPhi_dt);  
            #@parallel diffusion3D_step!(Tnew, T, K, 1.0/(ρ*cp), dt, dx, dy, dz)
        end

        # diffusion in z-direction
        @parallel (1:size(T,2), 1:size(T,3)) bc3D_x!(Tnew);                                         # set lateral boundary conditions (flux-free)
        @parallel (1:size(T,1), 1:size(T,3)) bc3D_y!(Tnew);                                         # set lateral boundary conditions (flux-free)
   
        Tnew[:,:,1] .= Tbot; Tnew[:,:,end] .= 0.0;                                                  # bottom & top temperature (constant)
    

        T, Tnew         =   Tnew, T;                                                                # Update temperature
        time, time_kyrs =   time + dt, time/SecYear/1e3;                                            # Keep track of evolved time
    
        if mod(it,100)==0  # print progress      
           # println(" Timestep $it = $(round(time/SecYear)/1e3) kyrs")
        #    x_km, z_km  =   x./1e3, z./1e3;
        #    #p1          =   heatmap(x_km, z_km, T[:,Int(Ny/2),:]',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="Temperature, $(round(time_kyrs, digits=2)) kyrs",  dpi=150)
        #    p1          =   heatmap(x_km, z_km, T[Int(Nx/2),:,:]',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="Temperature, $(round(time_kyrs, digits=2)) kyrs",  dpi=150)
        #    
        #    plot(p1); frame(anim)
        end
    
    end
    
    x_km, z_km  =   x./1e3, z./1e3;
    
    
    # compute analytical solution
    
    if Setup=="3D"
        Tanal  =  Tmax./(1 + 4*time*κ/σ^2)^(3/2).*exp.(  -(X.^2 .+ Y.^2 .+ Z.^2)./(σ^2 + 4*time*κ));                     # initial gaussian profile
        fname = "Diffusion_3D_Gaussian";
    end
    Terror =  T - Tanal

    Tslice  = T[:,Int(Ny/2),:];
    Tanal1  = Tanal[:,Int(Ny/2),:];
    Terror1 = Terror[:,Int(Ny/2),:];
    
    if CreatePlots
        # create plot 
        p1          =   heatmap(x_km, z_km, Tslice',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="T  3D $(round(time_kyrs/1e3, digits=2)) Myrs",  dpi=150)
        p2          =   heatmap(x_km, z_km, Tanal1',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="T anal 3D $(round(time_kyrs/1e3, digits=2)) Myrs",  dpi=150)
        p3          =   heatmap(x_km, z_km, Terror1',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="T error 3D $(round(time_kyrs/1e3, digits=2)) Myrs",  dpi=150)
        plot(p1,p2,p3);
        png(fname)
    end
    
    error = norm(Terror[:],2)/length(Terror[:]); 

    return error;        # return error
end # end of gaussian diffusion test

# Create a range of diffusion tests which calls the routines above
@testset "3D Gaussian diffusion" begin
    @test Diffusion_Gaussian3D("3D")           ≈  1.74e-5 atol=1e-4;
end;



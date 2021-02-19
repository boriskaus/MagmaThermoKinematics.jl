# this file tests various aspects of the 
using ZirconThermoKinematics
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Plots  
using LinearAlgebra
using SpecialFunctions
using Test

# Initialize for multiple threads (GPU is not tested here)
@init_parallel_stencil(Threads, Float64, 3);    # initialize parallel stencil in 3D


function test_Interpolation(Dimension="2D", InterpolationMethod="Linear")
  # test interpolation methods in 2D and 3D

  if Dimension=="2D"
    # Model parameters
    W,H                     =   1.,  1.;                                    # Width, Length, Height
    
    # Define coarse grid
    Nx, Nz                  =   33, 33;                                     # resolution of coarse grid
    dx,dz                   =   W/(Nx-1), H/(Nz-1);                         # grid size [m]
    x,z                     =   0:dx:((Nx-1)*dx), 0:dz:((Nz-1)*dz);         # 1D coordinate arrays
    coords                  =   collect(Iterators.product(x,z))             # generate coordinates from 1D coordinate vectors   
    X,Z                     =   (x->x[1]).(coords), (x->x[2]).(coords);     # transfer coords to 3D arrays
    Grid, Spacing           =   (x,z), (dx,dz);
    
    # Define fine grid
    Nx_f, Nz_f              =   65, 65;                                     # resolution of fine grid
    dx_f,dz_f               =   W/(Nx_f-1), H/(Nz_f-1);                     # grid size [m]
    x_f,z_f                 =   0:dx_f:H, 0:dz_f:H;                         # 1D coordinate arrays
    coords_f                =   collect(Iterators.product(x_f,z_f))         # generate coordinates from 1D coordinate vectors   
    X_f,Z_f                 =   (x->x[1]).(coords_f), (x->x[2]).(coords_f); # transfer coords to 3D arrays
    Grid_f                  =   (x_f, z_f);     # note
    
    # Define function on coarse grid
    T                       =   cos.(pi.*X).*sin.(2*pi.*Z)

    Data_coarse1            =   tuple(T);         # 1 dataset on grid
    Data_coarse2            =   (T, T);           # 2 datasets on grid 
    
    Grid_fine               =   (X_f, Z_f)

  elseif Dimension=="3D"
      # Model parameters
      W,L,H                 =   1., 1., 1.;                                    # Width, Length, Height
      
      # Define coarse grid
      Nx, Ny, Nz              =   33,33, 33;                                                    # resolution of coarse grid
      dx,dy,dz                =   W/(Nx-1), L/(Ny-1), H/(Nz-1);                                 # grid size [m]
      x,y,z                   =   0:dx:((Nx-1)*dx),  0:dy:((Ny-1)*dy), 0:dz:((Nz-1)*dz);        # 1D coordinate arrays
      coords                  =   collect(Iterators.product(x,y,z))                             # generate coordinates from 1D coordinate vectors   
      X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);   # transfer coords to 3D arrays
      Grid, Spacing           =   (x,y,z), (dx,dy,dz);
  
      # Define fine grid
      Nx_f, Ny_f, Nz_f        =   65, 65, 65;                                                       # resolution of fine grid
      dx_f,dy_f,dz_f          =   W/(Nx_f-1), L/(Ny_f-1), H/(Nz_f-1);                               # grid size [m]
      x_f,y_f,z_f             =   0:dx_f:H, 0:dy_f:L,0:dz_f:H;                                      # 1D coordinate arrays
      coords_f                =   collect(Iterators.product(x_f,y_f, z_f))                          # generate coordinates from 1D coordinate vectors   
      
      X_f,Y_f, Z_f            =   (x->x[1]).(coords_f), (x->x[2]).(coords_f), (x->x[3]).(coords_f); # transfer coords to 3D arrays
      Grid_f                  =   (x_f, y_f, z_f);    
      
      # Define function on coarse grid
      T                       =   cos.(pi.*X).*sin.(2*pi.*Z).*sin.(2*pi.*Y)
  
      Data_coarse1            =   tuple(T);         # 1 dataset on grid
      Data_coarse2            =   (T, T, T);        # 3 datasets on grid 

      Grid_fine               =   (X_f, Y_f, Z_f)
  end

 
  # Perform interpolation from coarse-fine grid
  Data_fine1 =  Interpolate( Grid, Spacing, Data_coarse1, Grid_fine, InterpolationMethod);       # one field
  Data_fine2 =  Interpolate( Grid, Spacing, Data_coarse2, Grid_fine, InterpolationMethod);       # several fields

  # Compute error 
  if Dimension=="2D"
    Tanal       =   cos.(pi.*X_f).*sin.(2*pi.*Z_f);
    Terror      =   Tanal - Data_fine2[2];

    p1          =   contourf(x, z,      Data_coarse1[1]',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Coarse ",  dpi=150, levels=10)
    p2          =   contourf(x_f, z_f,  Data_fine2[2]',         aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Fine ",    dpi=150, levels=10)
    p3          =   contourf(x_f, z_f,  Terror',                aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="error ",    dpi=150, levels=10)
    
    plot(p1,p2,p3);

    png("Interpolation_2D_$InterpolationMethod")
          
  
  elseif Dimension=="3D"
    Tanal       =   cos.(pi.*X_f).*sin.(2*pi.*Z_f).*sin.(2*pi.*Y_f);
    Terror      =   Tanal - Data_fine2[2];

    p1          =   contourf(x, z,      Data_coarse1[1][:,10,:]', aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Coarse ",  dpi=150, levels=10)
    p2          =   contourf(x_f, z_f,  Data_fine2[2][:,10,:]',   aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Fine ",    dpi=150, levels=10)
    p3          =   contourf(x_f, z_f,  Terror[:,10,:]',          aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="error ",    dpi=150, levels=10)
    
    plot(p1,p2,p3);
    png("Interpolation_3D_$InterpolationMethod")

  end

  error = norm(Terror[:],2)/length(Terror[:]); 
  return error;        # return error
end 


function test_SemiLagrangian2D(Method="ConstantZ",  InterpolationMethod="Linear", AdvectionMethod="RK2")
  # SemiLagrangian advection test in 2D
  
  # Model parameters
  W,H                     =   1.,  1.;                      # Width, Length, Height
  Nx, Nz                  =   129, 129;                      # resolution
  σ                       =   0.1;                          # halfwidth of gaussian
  Tmax                    =   2;                            # max. of gaussian
  dx,dz                   =   W/(Nx-1), H/(Nz-1);           # grid size [m]

  # Array initializations (1 - main arrays on which we can initialize properties)
  T                       =   @zeros(Nx,Nz);  
  Tnew                    =   @zeros(Nx,Nz);  
  
  # Set up model geometry & initial T structure
  x,z                     =   0:dx:((Nx-1)*dx), 0:dz:((Nz-1)*dz);
  coords                  =   collect(Iterators.product(x,z))             # generate coordinates from 1D coordinate vectors   
  X,Z                     =   (x->x[1]).(coords), (x->x[2]).(coords);     # transfer coords to 2D arrays
  Grid, Spacing           =   (x,z), (dx,dz);
 
  Xc                      =   0.5;
  Zc                      =   0.75;
  if      Method=="ConstantZ"
    # Simple advection in z-direction
    Vx        =   X.*0;
    Vz        =   Z.*0 .- 1;
    TotalTime =   0.5;                         # thermal cooling age
    
  elseif  Method=="Rotation"
    # Rigid body rotation:
    Vx        =    (Z .- 0.5);
    Vz        =   -(X .- 0.5);
    TotalTime =   2*pi;    

  elseif  Method=="Shear"
    # Shear velocity field:
    Vx        =  -sin.(pi.*X).*cos.(pi.*Z);
    Vz        =   cos.(pi.*X).*sin.(pi.*Z);
    
  end

  T                      .=   Tmax.*exp.(  -( (X .- Xc).^2 .+  (Z .- Zc).^2 )./σ^2);                 # initial gaussian profile
  Tnew                   .=   T;
  
  maxVel      =   max( maximum(abs.(Vx)), maximum(abs.(Vz)) );
  dt          =   min(dx,dz)/2.0/maxVel;   # stable timestep (to ensure we move no more than 0.5*(dx|dz) per timesstep )
  numTime     =   ceil(TotalTime/dt);
  nt          =   Int(numTime);
  dt          =   TotalTime/nt;

  
#  ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
#  println("Animation directory: $(anim.dir)")

  time,time_kyrs          = 0.0, 0.0;
  err = 100;
  it = 0;
  #nt = 100
  for it=1:nt
  
      # Perform an advection step for temperature 
      Tnew        =   AdvectTemperature(T,    Grid,  (Vx,Vz),   Spacing,    dt, AdvectionMethod, InterpolationMethod);    
  

      T, Tnew     =   Tnew, T;                                                                # Update temperature
      time        =   time + dt;                                            # Keep track of evolved time
  
      if mod(it,1000)==0  # print progress      
          #println(" Timestep $it = $((time)) ")
          #p1          =   heatmap(x_km, z_km, T[:,Int(Ny/2),:]',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="Temperature, $(round(time_kyrs, digits=2)) kyrs",  dpi=150)
          #p1          =   contourf(x, z, T[:,:]',         aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Temperature, $(round(time, digits=2)) ",  dpi=150, levels=10)
          
          #plot(p1); frame(anim)
      end
  
  end
  
  x_km, z_km  =   x, z;
  
  
  # compute analytical solution
  if      Method=="ConstantZ"
    # Simple advection in z-direction
    Xc    =   0.5;
    Zc    =   0.25;

  elseif      Method=="Rotation"
      # Should arrive @ same point
  end
  Tanal =   Tmax.*exp.(  -( (X .- Xc).^2 .+  (Z .- Zc).^2 )./σ^2);                 # initial gaussian profile
  fname       = "Advection_2D_$(Method)_$(AdvectionMethod)_$(InterpolationMethod)";


  Terror      =  T - Tanal;   # error
  
  # create plot 
  p1          =   contourf(x, z, T',      aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Tnumeric, $(round(time, digits=2)) ",  dpi=300, levels=10)
  p2          =   contourf(x, z, Tanal',  aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Tanal, $(round(time, digits=2)) ",     dpi=300, levels=10)
  p3          =   contourf(x, z, Terror', aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Terror, $(round(time, digits=2)) ",    dpi=300, levels=10)
  plot(p1,p2,p3);

 
  png(fname)
  
  error = norm(Terror[:],2)/length(Terror[:]); 
  return error;        # return error
end 

function test_SemiLagrangian3D(Method="ConstantZ",  InterpolationMethod="Linear", AdvectionMethod="RK2")
    # SemiLagrangian advection test in 2D
    
    # Model parameters
    W,L,H                   =   1., 1., 1.;                         # Width, Length, Height
    Nx, Ny, Nz              =   50, 50, 50;                         # resolution
    σ                       =   0.1;                                # halfwidth of gaussian
    Tmax                    =   2;                                  # max. of gaussian
    TotalTime               =   1;                                  # thermal cooling age
    dx,dy,dz                =   W/(Nx-1), L/(Ny-1), H/(Nz-1);       # grid size [m]
    dt                      =   min(dx,dy,dz)./2;                   # stable timestep (required for explicit FD)
    
    numTime                 =   ceil(TotalTime/dt);
    nt                      =   Int(numTime);
  
    # Array initializations (1 - main arrays on which we can initialize properties)
    T                       =   @zeros(Nx,Ny,Nz);  
    Tnew                    =   @zeros(Nx,Ny,Nz);  
    
    # Set up model geometry & initial T structure
    x,y,z                   =   0:dx:((Nx-1)*dx), 0:dy:((Ny-1)*dy), 0:dz:((Nz-1)*dz);
    coords                  =   collect(Iterators.product(x,y,z))                               # generate coordinates from 1D coordinate vectors   
    X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);     # transfer coords to 3D arrays
    Grid, Spacing           =   (x,y,z), (dx,dy,dz);
    Vx                      =  -sin.(pi.*X).*cos.(pi.*Z);
    Vy                      =   zeros(size(X));
    Vz                      =   cos.(pi.*X).*sin.(pi.*Z);
    

    Xc                      =   0.5;
    Yc                      =   0.5;
    Zc                      =   0.75;
    if      Method=="ConstantZ"
      # Simple advection in z-direction
      Vx        =   X.*0;
      Vy        =   Y.*0;
      Vz        =   Z.*0 .- 1;
      TotalTime =   0.5;                         # thermal cooling age
      
    elseif  Method=="Rotation_alongY"
      # Rigid body rotation:
      Vx        =    (Z .- 0.5);
      Vy        =   X.*0;
      Vz        =   -(X .- 0.5);
      TotalTime =   2*pi;    

    elseif  Method=="Rotation_alongX"
      # Rigid body rotation:
      Vy        =    (Z .- 0.5);
      Vx        =   Y.*0;
      Vz        =   -(Y .- 0.5);
      TotalTime =   2*pi;    
  
    elseif  Method=="Rotation_alongZ"
      # Rigid body rotation:
      Vy        =   -(X .- 0.5);
      Vx        =    (Y .- 0.5);
      Vz        =   Z.*0.0;
      TotalTime =   2*pi;    
      Xc                      =   0.5;
      Yc                      =   0.75;
      Zc                      =   0.5;
   end

  
    T           .=   Tmax.*exp.(  -(((X .- Xc).^2 .+ (Y .- Yc).^2 .+ (Z .- Zc).^2)./(σ^2)) );                 # initial gaussian profile
    Tnew        .=   T;
   
    
    maxVel      =   max( maximum(abs.(Vx)), maximum(abs.(Vy)), maximum(abs.(Vz)) );
    dt          =   min(dx,dy,dz)/2.0/maxVel;   # stable timestep (to ensure we move no more than 0.5*(dx|dz) per timesstep )
    numTime     =   ceil(TotalTime/dt);
    nt          =   Int(numTime);
    dt          =   TotalTime/nt;
  

    #ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    #println("Animation directory: $(anim.dir)")

    time          = 0.0;
    for it=1:nt
    
        # Perform an advection step for temperature 
        Tnew    =   AdvectTemperature(T,    Grid,  (Vx,Vy,Vz),   Spacing,    dt, AdvectionMethod, InterpolationMethod);    
    

        T, Tnew         =   Tnew, T;                                                                # Update temperature
        time            =   time + dt;                                            # Keep track of evolved time
    
        if mod(it,1000)==0  # print progress      
            println(" Timestep $it = $((time)) ")
            #p1          =   heatmap(x_km, z_km, T[:,Int(Ny/2),:]',         aspect_ratio=1, xlims=(x_km[1],x_km[end]), ylims=(z_km[1],z_km[end]),   c=:inferno, title="Temperature, $(round(time_kyrs, digits=2)) kyrs",  dpi=150)
            #p1          =   heatmap(x, z, T[:,Int(Ny/2),:]',         aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Temperature, $time",  dpi=150)
            p1          =   heatmap(x, y, T[:,:,Int(Nz/2)]',         aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Temperature, $time",  dpi=150)
            
           # p1          =   heatmap(x, z, T[Int(Nx/2),:,:]',         aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Temperature, $time",  dpi=150)
            
            plot(p1); frame(anim)
        end
    
    end
    
    x_km, z_km  =   x, z;
    
    
    # compute analytical solution
    if      Method=="ConstantZ"
      # Simple advection in z-direction
      Xc    =   0.5;
      Yc    =   0.5;
      Zc    =   0.25;

    elseif  (Method=="Rotation_alongY") || (Method=="Rotation_alongX")
      # Should arrive @ same point

    end
    Tanal =   Tmax.*exp.(  -( (X .- Xc).^2 .+  (Y .- Yc).^2 .+  (Z .- Zc).^2 )./σ^2);                 # gaussian profile
    fname = "Advection_3D_$(Method)_$(AdvectionMethod)_$(InterpolationMethod)";

    Terror      =  T - Tanal;   # error

    Tslice  = T[:,Int(Ny/2),:];
    Tanal1  = Tanal[:,Int(Ny/2),:];
    Terror1 = Terror[:,Int(Ny/2),:];

    Vx1 = Vx[:,Int(Ny/2),:];
    Vy1 = Vy[:,Int(Ny/2),:];
    Vz1 = Vz[:,Int(Ny/2),:];
    X1  = X[:,Int(Ny/2),:];
    Y1  = Y[:,Int(Ny/2),:];
    Z1  = Z[:,Int(Ny/2),:];
    
    
    # create plot 
    p1          =   contourf(x, z, Tslice',      aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Tnumeric, $(round(time, digits=2)) ",  dpi=300, levels=10)
    p2          =   contourf(x, z, Tanal1',  aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Tanal 3D, $(round(time, digits=2)) ",     dpi=300, levels=10)
    p3          =   contourf(x, z, Terror1', aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Terror, $(round(time, digits=2)) ",    dpi=300, levels=10)
    plot(p1,p2,p3);


   # st=20;
   # quiver!(X1[1:st:end], Z1[1:st:end], gradient=(Vx1[1:st:end]*0.1,Vz1[1:st:end]*0.1), arrow = :arrow, color = :white)
        
    png(fname)
    
    error = norm(Terror[:],2)/length(Terror[:]); 
    return error;        # return error
end 


if 1==0
# test the interpolation methods
@testset "Interpolation" begin
  @test test_Interpolation("2D", "Linear") ≈  2.9880072526933544e-5  atol=1e-8;
  @test test_Interpolation("2D", "Cubic")  ≈  8.708144266671426e-7   atol=1e-8;
  @test test_Interpolation("3D", "Linear") ≈  4.283068644002478e-6   atol=1e-8;
  @test test_Interpolation("3D", "Cubic")  ≈  7.607530621010341e-8   atol=1e-8;
end;

@testset "2D semi-lagrangian advection" begin
  @test test_SemiLagrangian2D("ConstantZ","Cubic", "RK4"    )   ≈ 7.257090264260951e-6    atol=1e-8;
  @test test_SemiLagrangian2D("ConstantZ","Linear","Euler"  )   ≈ 0.00026385729240145655  atol=1e-6;
  
  @test test_SemiLagrangian2D("Rotation", "Linear", "Euler" )   ≈ 0.0009061021221256843   atol=1e-8;
  @test test_SemiLagrangian2D("Rotation", "Linear", "RK2"   )   ≈ 0.000892833826724465    atol=1e-8;
  @test test_SemiLagrangian2D("Rotation", "Linear", "RK4"   )   ≈ 0.0008928336158448796   atol=1e-8;
  
  @test test_SemiLagrangian2D("Rotation", "Cubic", "Euler"  )   ≈ 0.00013400741577631413  atol=1e-8;
  @test test_SemiLagrangian2D("Rotation", "Cubic", "RK2"    )   ≈ 1.1006030724096006e-6   atol=1e-8;
  @test test_SemiLagrangian2D("Rotation", "Cubic", "RK4"    )   ≈ 1.0944206043526207e-6   atol=1e-8;


end;

@testset "3D semi-lagrangian advection" begin
  @test test_SemiLagrangian3D("ConstantZ","Linear","RK2"   )          ≈ 6.822115778040405e-5    atol=1e-8;
  @test test_SemiLagrangian3D("ConstantZ","Cubic", "RK2"   )          ≈ 1.1038374380141209e-6   atol=1e-8;
  @test test_SemiLagrangian3D("Rotation_alongY","Linear", "Euler")    ≈ 0.00017064553949646487  atol=1e-8;
  @test test_SemiLagrangian3D("Rotation_alongY","Cubic", "RK2"   )    ≈ 1.899516467222456e-6    atol=1e-8;
  @test test_SemiLagrangian3D("Rotation_alongY","Cubic", "RK2"   )    ≈ 1.899516467222456e-6    atol=1e-8;

  @test test_SemiLagrangian3D("Rotation_alongX","Cubic", "RK2"   )    ≈ 1.8995164672224572e-6    atol=1e-8;
  @test test_SemiLagrangian3D("Rotation_alongY","Cubic", "RK2"   )    ≈ 1.899516467222456e-6    atol=1e-8;
  @test test_SemiLagrangian3D("Rotation_alongZ","Cubic", "RK2"   )    ≈ 1.8995164672224786e-6   atol=1e-8;
  

end;


end

err = test_SemiLagrangian3D("Rotation_alongY","Cubic","RK2")  


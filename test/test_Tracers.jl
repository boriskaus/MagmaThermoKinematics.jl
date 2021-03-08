# this file tests various aspects of the tracers routines
using MagmaThermoKinematics
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Plots  
using LinearAlgebra
using SpecialFunctions
using Test


const CreatePlots = true      # easy way to deactivate plotting throughout

function test_TracerUpdate(Dimension="2D", InterpolationMethod="Linear")
  # test interpolation methods from grid to tracers in 2D and 3D

  if Dimension=="2D"
    # Model parameters
    W,H                     =   1.,  1.;                                    # Width, Length, Height
    
    # Define grid
    Nx, Nz                  =   33, 33;                                     # resolution of coarse grid
    dx,dz                   =   W/(Nx-1), H/(Nz-1);                         # grid size [m]
    x,z                     =   0:dx:((Nx-1)*dx), 0:dz:((Nz-1)*dz);         # 1D coordinate arrays
    coords                  =   collect(Iterators.product(x,z))             # generate coordinates from 1D coordinate vectors   
    X,Z                     =   (x->x[1]).(coords), (x->x[2]).(coords);     # transfer coords to 3D arrays
    Grid,FullGrid,Spacing   =   (x,z), (X,Z), (dx,dz);
   
    # Define function on grid
    T                       =   cos.(pi.*X).*sin.(2*pi.*Z)
  
  elseif Dimension=="3D"
      # Model parameters
      W,L,H                 =   1., 1., 1.;                                    # Width, Length, Height
      
      # Define grid
      Nx, Ny, Nz              =   33,33, 33;                                                    # resolution of coarse grid
      dx,dy,dz                =   W/(Nx-1), L/(Ny-1), H/(Nz-1);                                 # grid size [m]
      x,y,z                   =   0:dx:((Nx-1)*dx),  0:dy:((Ny-1)*dy), 0:dz:((Nz-1)*dz);        # 1D coordinate arrays
      coords                  =   collect(Iterators.product(x,y,z))                             # generate coordinates from 1D coordinate vectors   
      X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);   # transfer coords to 3D arrays
      Grid, FullGrid, Spacing =   (x,y,z), (X,Y,Z), (dx,dy,dz);
  
      # Define function on coarse grid
      T                       =   cos.(pi.*X).*sin.(2*pi.*Z).*sin.(2*pi.*Y)

  end

  # Create tracer structure that fill te full grid
  Tracers                     =   InitializeTracers(FullGrid,3, false);

  Phi     =   Z./H;
 
  # Perform interpolation from grid -> tracers
  Tracers =  UpdateTracers(Tracers, Grid, T, Phi, InterpolationMethod);
          
  # Compute error 
  Tr_coord    =   Tracers.coord; Tr_coord = hcat(Tr_coord...)';       # extract array with coordinates of tracers
    
  if Dimension=="2D"
    Tanal       =   cos.(pi.*Tr_coord[:,1]).*sin.(2*pi.*Tr_coord[:,2]);
    Phi_anal    =    1 .- Tr_coord[:,end]/H;
    Error       =   (Tanal - Tracers.T)  .+ (Phi_anal-Tracers.Phi_melt); 

    if CreatePlots
      p1          =   contourf(x, z,      T',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=300, levels=10)
      p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=2.0,dpi=300)

    #   p1          =   contourf(x, z,      (1.0-Phi)',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
    #   p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.Phi_melt, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

      plot(p1,p2); 

     png("TracerUpdate_2D_$InterpolationMethod")
    end
  
  elseif Dimension=="3D"
    Tanal       =   cos.(pi.*Tr_coord[:,1]).*sin.(2*pi.*Tr_coord[:,3]).*sin.(2*pi.*Tr_coord[:,2]);
    Phi_anal    =   1 .- Tr_coord[:,end]/H;
    Error       =   (Tanal - Tracers.T)  .+ (Phi_anal-Tracers.Phi_melt); 

    if CreatePlots
    #  p1          =   contourf(x, z,      Data_coarse1[1][:,10,:]', aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
    #  p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

   #   plot(p1,p2);
   #   png("TracerUpdate_2D_$InterpolationMethod")
    end

  end

  error = norm(Error[:],2)/length(Error[:]); 
  return error;        # return error
end 


function test_Tracer2Grid(Dimension="2D", InterpolationMethod="Linear")
  # test routines to go from tracers -> Grid

  if Dimension=="2D"
    dim                     = 2;
    # Model parameters
    W,H                     =   1.,  1.;                                    # Width, Length, Height
    
    # Define grid
    Nx, Nz                  =   65, 65;                                   # resolution of coarse grid
    dx,dz                   =   W/(Nx-1), H/(Nz-1);                         # grid size [m]
    x,z                     =   0:dx:((Nx-1)*dx), 0:dz:((Nz-1)*dz);         # 1D coordinate arrays
    coords                  =   collect(Iterators.product(x,z))             # generate coordinates from 1D coordinate vectors   
    X,Z                     =   (x->x[1]).(coords), (x->x[2]).(coords);     # transfer coords to 3D arrays
    Grid,FullGrid,Spacing   =   (x,z), (X,Z), (dx,dz);
   
    # Define function on grid
    T                       =   cos.(pi.*X).*sin.(2*pi.*Z)
  
  elseif Dimension=="3D"
      dim                   = 3;
      # Model parameters
      W,L,H                 =   1., 1., 1.;                                    # Width, Length, Height
      
      # Define grid
      Nx, Ny, Nz              =   33,33, 33;                                                    # resolution of coarse grid
      dx,dy,dz                =   W/(Nx-1), L/(Ny-1), H/(Nz-1);                                 # grid size [m]
      x,y,z                   =   0:dx:((Nx-1)*dx),  0:dy:((Ny-1)*dy), 0:dz:((Nz-1)*dz);        # 1D coordinate arrays
      coords                  =   collect(Iterators.product(x,y,z))                             # generate coordinates from 1D coordinate vectors   
      X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);   # transfer coords to 3D arrays
      Grid, FullGrid, Spacing =   (x,y,z), (X,Y,Z), (dx,dy,dz);
  
      # Define function on coarse grid
      T                       =   cos.(pi.*X).*sin.(2*pi.*Z).*sin.(2*pi.*Y)

  end

  # Create tracer structure that fill the full grid
  #println("InitializeTracers:")
  Tracers                     =   InitializeTracers(FullGrid,3, false);

  # Set different phases/temperatures on tracers (default Phase = 0)
  # Sphere 1 (Phase=2)
  if      dim==2; cen = [0.5; 0.5]; 
  elseif  dim==3; cen = [0.5; 0.5; 0.5]; end
  R = 0.1;
  [Tracers.Phase[i]=2  for i=1:length(Tracers) if sum( (Tracers.coord[i]-cen).^2)<R^dim ]   # julia iterator to set props on Tracers

  if      dim==2; cen = [0.15; 0.7]; 
  elseif  dim==3; cen = [0.15; 0.1; 0.7]; end
  R = 0.1;
  [Tracers.Phase[i]=3  for i=1:length(Tracers) if sum( (Tracers.coord[i]-cen).^2)<R^dim ]

  # Call main routine to compute phase fractions from particles
  #println("PhaseRatioFromTracers:")
  PhaseRatio     = PhaseRatioFromTracers(FullGrid, Grid, Tracers, InterpolationMethod);

  
  Tr_coord    =   Tracers.coord; Tr_coord = hcat(Tr_coord...)';       # extract array with coordinates of tracers
  if Dimension=="2D"
    Data =   PhaseRatio[:,:,1];
    if CreatePlots
     # @show size(PhaseRatio)
      #p1          =   contourf(x, z,      PhaseRatio[:,:,3]',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=300, levels=10)
      p1          =   heatmap(x, z,      PhaseRatio[:,:,3]',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=300)
    #  p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=2.0,dpi=300)

    #   p1          =   contourf(x, z,      (1.0-Phi)',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
    #   p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.Phi_melt, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

      #plot(p1,p2); 
      plot(p1); 
      

     png("Tracer2Grid_2D_$InterpolationMethod")
    end
  
  elseif Dimension=="3D"
    Data =   PhaseRatio[:,:,:,1];

    if CreatePlots
    #  p1          =   contourf(x, z,      Data_coarse1[1][:,10,:]', aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
    #  p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

   #   plot(p1,p2);
   #   png("Tracer2Grid_2D_$InterpolationMethod")
    end

  end

  return norm(Data[:]);        # return norm of data
end 


  
# ===================================================================================================
if 1==1
  @testset "Update Tracer" begin
    @test test_TracerUpdate("2D", "Linear") ≈  2.2699573946787955e-5  atol=1e-8;
    @test test_TracerUpdate("2D", "Cubic")  ≈  4.217098535399923e-7   atol=1e-8;
    @test test_TracerUpdate("3D", "Linear") ≈  2.9095783218223196e-6  atol=1e-8;
    @test test_TracerUpdate("3D", "Cubic")  ≈  3.252909873990004e-8   atol=1e-10;
  end;

  @testset "Tracer2Grid" begin
    @test test_Tracer2Grid("2D", "DistanceWeighted")  ≈  62.931013816718384   atol=1e-3;
    @test test_Tracer2Grid("2D", "Constant")          ≈  62.889772274910676   atol=1e-3;
    @test test_Tracer2Grid("3D", "DistanceWeighted")  ≈  189.5417994796926    atol=1e-3;
    @test test_Tracer2Grid("3D", "Constant")          ≈  189.53531363996018   atol=1e-3;
  end;

end




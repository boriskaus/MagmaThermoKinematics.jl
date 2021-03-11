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


function test_PhaseRatioFromTracers(Dimension="2D", InterpolationMethod="Linear", Method="TracersEverywhere")
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
  if      Method=="TracersEverywhere"
    Tracers                     =   InitializeTracers(FullGrid,3, false);   # tracers defined everywhere
    
  elseif  Method=="TracersLimitedRegion"
    # Tracers defined in a limited region only, with a default phase outside this area

    # we use the same routine to initialize tracers but in a smaller square region
    W_l, H_l      =   W/3.0, H/3.0;
    dx_l,dz_l     =   W_l/(Nx-1), H_l/(Nz-1);                         # grid size [m]
    xl,zl         =   0.3:dx_l:((Nx-1)*dx_l)+0.3, 0.3:dz_l:((Nz-1)*dz_l)+0.3;         # 1D coordinate arrays of limited region
    if dim==3
      L_l         =   L/3.0;
      dy_l        =   L_l/(Ny-1);
      yl          =   0.3:dy_l:((Ny-1)*dy_l)+0.3;
      coords      =   collect(Iterators.product(xl,yl,zl)) 
      X,Y,Z       =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);   # transfer coords to 3D arrays
      FullGrid_l  =   (X,Y,Z)

    else
      coords      =   collect(Iterators.product(xl,zl)) 
      X,Z         =   (x->x[1]).(coords), (x->x[2]).(coords);   # transfer coords to 2D arrays
      FullGrid_l  =   (X,Z)
    end
    Tracers       =   InitializeTracers(FullGrid_l,2, false);   # tracers defined everywhere

  else
    error("unknown Method = $Method")
  end

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
  if      Method=="TracersEverywhere"
    PhaseRatio, NumTracers  =   PhaseRatioFromTracers(FullGrid, Grid, Tracers, InterpolationMethod, true);
  else
    PhaseRatio,  NumTracers  =   PhaseRatioFromTracers(FullGrid, Grid, Tracers, InterpolationMethod, true, BackgroundPhase=4);
  end

  # Also test the Rocktype routine
  RockType    =   RockAssemblage(PhaseRatio);


  Tr_coord    =   Tracers.coord; Tr_coord = hcat(Tr_coord...)';       # extract array with coordinates of tracers
  if Dimension=="2D"
    Data =   PhaseRatio[:,:,1];
    if CreatePlots
     # @show size(PhaseRatio)
      #p1          =   contourf(x, z,      PhaseRatio[:,:,3]',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=300, levels=10)
      #p1          =   heatmap(x, z,      PhaseRatio[:,:,3]',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=300)
      
       p1          =   heatmap(x, z,      RockType',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="RockType",  dpi=300)
      #p1          =   heatmap(x, z,      NumTracers',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="NumTracers",  dpi=300)
      
      #p2 = plot(X[:],Z[:],markershape = :plus, markersize=0.2)
      p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=1.0,dpi=300)
      

    #   p1          =   contourf(x, z,      (1.0-Phi)',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
    #   p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.Phi_melt, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

      #plot(p1,p2); 
      plot(p1); 
      

      png("PhaseRatioFromTracers_2D_$(InterpolationMethod)_$(Method)")
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

  return norm(RockType);        # return norm of data
end 

function test_TracerToGrid(Dimension="2D")
  # tests interpolating TracersToGrid! routine, which interpolates, e.g. temperature from tracers to the grid 

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
  @time Tracers               =   InitializeTracers(FullGrid,3, false);

  Phi     =   Z./H;
 
  # Perform interpolation from grid -> tracers
  Tracers =  UpdateTracers(Tracers, Grid, T, Phi, "Linear");
          

  # Go back from tracers to a new grid 
  Tnew        =   copy(T)*.0;
  NumTracers  =   TracersToGrid!(Tnew, FullGrid, Grid, Tracers, "T", "Constant", true);
  NumTracers  =   TracersToGrid!(Tnew, FullGrid, Grid, Tracers, "T", "Constant", true);   # do it a second time, to make sire it doesn't double



  # Compute error 
  Tr_coord    =   Tracers.coord; Tr_coord = hcat(Tr_coord...)';       # extract array with coordinates of tracers
    
  if Dimension=="2D"
    Tanal       =   cos.(pi.*X).*sin.(2*pi.*Z);
    #Phi_anal    =    1 .- Tr_coord[:,end]/H;
    Error       =   (Tanal - Tnew); 

    if CreatePlots
      p2          =   contourf(x, z,      Tnew',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Tnew",  dpi=300, levels=10)
      p1          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="T on Tracers", markersize=2.0,dpi=300)

    #   p1          =   contourf(x, z,      (1.0-Phi)',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
    #   p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.Phi_melt, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

      plot(p1,p2); 

     png("TracerToGrid_2D")
    end
  
  elseif Dimension=="3D"
    Tanal       =   cos.(pi.*X).*sin.(2*pi.*Z).*sin.(2*pi.*Y);
    #Phi_anal    =   1 .- Tr_coord[:,end]/H;
    Error       =   (Tanal - Tnew); 

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

  
# ===================================================================================================
if 1==0
  @testset "Update Tracer" begin
    @test test_TracerUpdate("2D", "Linear") ≈  2.2699573946787955e-5  atol=1e-8;
    @test test_TracerUpdate("2D", "Cubic")  ≈  4.217098535399923e-7   atol=1e-8;
    @test test_TracerUpdate("3D", "Linear") ≈  2.9095783218223196e-6  atol=1e-8;
    @test test_TracerUpdate("3D", "Cubic")  ≈  3.252909873990004e-8   atol=1e-10;
  end;

  @testset "PhaseRatioFromTracers" begin
    @test test_PhaseRatioFromTracers("2D", "DistanceWeighted")  ≈  74.96665925596525    atol=1e-3;
    @test test_PhaseRatioFromTracers("2D", "Constant")          ≈  75.17978451685       atol=1e-3;
    @test test_PhaseRatioFromTracers("3D", "DistanceWeighted")  ≈  189.66285877841239   atol=1e-3;
    @test test_PhaseRatioFromTracers("3D", "Constant")          ≈  189.68658360569415   atol=1e-3;
    @test test_PhaseRatioFromTracers("2D", "DistanceWeighted","TracersLimitedRegion") ≈  246.4710936398019 atol=1e-3;
  end;


  @testset "TracerToGrid" begin
    @test test_TracerToGrid("2D")  ≈  0.0006078889344375056    atol=1e-8;
    @test test_TracerToGrid("3D")  ≈  0.00010053219987571875   atol=1e-8;
  end;

end

 test_TracerToGrid("2D")  
 test_TracerToGrid("3D")  




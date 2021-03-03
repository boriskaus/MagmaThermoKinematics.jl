# this file tests various aspects of the tracers routines
using MagmaThermoKinematics
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Plots  
using LinearAlgebra
using SpecialFunctions
using Test


const CreatePlots = false      # easy way to deactivate plotting throughout

function test_TracerUpdate(Dimension="2D", InterpolationMethod="Linear")
  # test interpolation methods from grid to tracers in 2D and 3D

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
    
    # generate tracer coordinates
    Nxt, Nzt                =   50, 50          # num of tracers in x,z
    Wt,Ht                   =   W, H;
    dxt,dzt                 =   Wt/(Nxt-1), Ht/(Nzt-1);
    xt,zt                   =   0:dxt:Wt, 0:dz:Ht;
    coordst                 =   collect(Iterators.product(xt,zt))             # generate coordinates from 1D coordinate vectors   
    Xt,Zt                   =   (x->x[1]).(coordst), (x->x[2]).(coordst);     # transfer coords to 2D arrays
  
    # Define function on coarse grid
    T                       =   cos.(pi.*X).*sin.(2*pi.*Z)

    # Create tracer structure
    new_tracer  =   Tracer(num=1, coord=[Xt[1]; Zt[1]], T=0.,);               # Create new tracer
    Tracers     =   StructArray([new_tracer]);                                # Create tracer array
    for i=firstindex(Xt)+1:lastindex(Xt)
      new_tracer  =   Tracer(num=i, coord=[Xt[i]; Zt[i]], T=0.);              # Create new tracer
      push!(Tracers, new_tracer);                             # Add new point to existing array
    end
    Tracers0 = copy(Tracers);         # create a copy of the original tracers

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
  
      # generate tracer coordinates
      Nxt, Nyt, Nzt           =   50, 50, 50          # num of tracers in x,z
      Wt,Lt,Ht                =   0.1, 0.1, 0.1;
      dxt,dyt,dzt             =   Wt/(Nxt-1), Lt/(Nyt-1), Ht/(Nzt-1);
      xt,yt,zt                =   0:dxt:Wt, 0:dyt:Lt, 0:dz:Ht;
      coordst                 =   collect(Iterators.product(xt,yt,zt))             # generate coordinates from 1D coordinate vectors   
      Xt,Yt,Zt                =   (x->x[1]).(coordst), (x->x[2]).(coordst),(x->x[3]).(coordst);     # transfer coords to 2D arrays
    
      # Define function on coarse grid
      T                       =   cos.(pi.*X).*sin.(2*pi.*Z).*sin.(2*pi.*Y)

      # Create tracer structure
      new_tracer  =   Tracer(num=1, coord=[Xt[1]; Yt[1]; Zt[1]], T=0.);           # Create new tracer
      Tracers     =   StructArray([new_tracer]);                                  # Create tracer array
      for i=firstindex(Xt)+1:lastindex(Xt)
        new_tracer  =   Tracer(num=i, coord=[Xt[i]; Yt[i]; Zt[i]], T=0.);         # Create new tracer
        push!(Tracers, new_tracer);                                               # Add new point to existing array
      end
  
  end

  Phi = T.*0.0;
 
  # Perform interpolation from grid -> tracers
  Tracers = UpdateTracers(Tracers, Grid, T, Phi, InterpolationMethod);
  
          
  # Compute error 
  Tr_coord    =   Tracers.coord; Tr_coord = hcat(Tr_coord...)';       # extract array with coordinates of tracers
    
  if Dimension=="2D"
    Tanal       =   cos.(pi.*Tr_coord[:,1]).*sin.(2*pi.*Tr_coord[:,2]);
    Terror      =   Tanal - Tracers.T;

    if CreatePlots
      p1          =   contourf(x, z,      T',       aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
      p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

      plot(p1,p2); 

      png("TracerUpdate_2D_$InterpolationMethod")
    end
  
  elseif Dimension=="3D"
    Tanal       =   cos.(pi.*Tr_coord[:,1]).*sin.(2*pi.*Tr_coord[:,3]).*sin.(2*pi.*Tr_coord[:,2]);
    Terror      =   Tanal - Tracers.T;

    if CreatePlots
    #  p1          =   contourf(x, z,      Data_coarse1[1][:,10,:]', aspect_ratio=1, xlims=(x[1],x[end]), ylims=(z[1],z[end]),   c=:inferno, title="Grid",  dpi=150, levels=10)
    #  p2          =   scatter(Tr_coord[:,1], Tr_coord[:,2], zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), title="Tracers", markersize=5.0)

   #   plot(p1,p2);
   #   png("TracerUpdate_2D_$InterpolationMethod")
    end

  end

  error = norm(Terror[:],2)/length(Terror[:]); 
  return error;        # return error
end 





# ===================================================================================================
@testset "Update Tracer" begin
  @test test_TracerUpdate("2D", "Linear") ≈  1.0555442659990556e-5  atol=1e-8;
  @test test_TracerUpdate("2D", "Cubic")  ≈  7.422707661072311e-7   atol=1e-8;
  @test test_TracerUpdate("3D", "Linear") ≈  4.839581367935359e-6   atol=1e-8;
  @test test_TracerUpdate("3D", "Cubic")  ≈  2.1008678624797836e-7  atol=1e-8;
end;

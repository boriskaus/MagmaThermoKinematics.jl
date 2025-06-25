# this file tests various aspects of the advection routines
using MagmaThermoKinematics
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Plots  
using LinearAlgebra
using SpecialFunctions
using Test

#using WriteVTK

const CreatePlots = true      # easy way to deactivate plotting throughout


function test_HostRockVelocityFromDike(Dimension="2D", DikeType="ElasticDike", DikeAngle=[45])
  # test generating host velocity from various dikes, with different size/orientation/type in both 2D and 3DD
  
  if Dimension=="2D"
    # Model parameters
    W,H                     =   30.0,  30.0;                                # Width, Length, Height
    
    # Define grid
    Nx, Nz                  =   129, 129;                                     # resolution of coarse grid
    dx,dz                   =   W*1e3/(Nx-1), H*1e3/(Nz-1);                   # grid size [m]
    x,z                     =   0:dx:W*1e3, -H*1e3:dz:0;                      # 1D coordinate arrays
    coords                  =   collect(Iterators.product(x,z))               # generate coordinates from 1D coordinate vectors   
    X,Z                     =   (x->x[1]).(coords), (x->x[2]).(coords);       # transfer coords to 3D arrays
    Grid, FullGrid, Spacing =   (x,z), (X,Z), (dx,dz);

    Hdike                   =   100.0;
    Wdike                   =   20000.0;
    T_in                    =   900.0;

    cen                     =   [W/2;-H/2].*1e3;
  elseif Dimension=="3D"
      # Model parameters
      W,L,H                 =   30., 40., 50.;                                    # Width, Length, Height
      
      # Define coarse grid
      Nx, Ny, Nz              =   65,65,65;                                                    # resolution of coarse grid
      dx,dy,dz                =   W*1e3/(Nx-1), L*1e3/(Ny-1), H*1e3/(Nz-1);                     # grid size [m]
      x,y,z                   =   0:dx:((Nx-1)*dx),  0:dy:((Ny-1)*dy), -((Nz-1)*dz):dz:0.;      # 1D coordinate arrays
      coords                  =   collect(Iterators.product(x,y,z))                             # generate coordinates from 1D coordinate vectors   
      X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);   # transfer coords to 3D arrays
      Grid, FullGrid, Spacing =   (x,y,z), (X,Y,Z), (dx,dy,dz);
      cen                     =   [W/2;L/2; -H/2].*1e3;
      
 
      Hdike                   =   100.0;
      Wdike                   =   20000.0;
      T_in                    =   900.0;
  end

  # Create dike
  dike              =   Dike(W=Wdike, H=Hdike,Center=cen,Angle=DikeAngle,Type=DikeType,T=T_in);  # Specify dike 

  # Compute velocity required to create space for dike
  Δ                 =   Hdike;          # max. dike opening (m)
  dt                =   1;              # time in which the dike opened fully

  Velocity          =   HostRockVelocityFromDike(Grid, FullGrid, Δ, dt,dike);          # compute velocity field

  
  if Dimension=="2D"
    Vel      =   Velocity[:];

    if CreatePlots
      Vx,Vz       =   Velocity[1],Velocity[2];
      p1          =   heatmap(x/1e3, z/1e3,      Vx',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="2D Vx",  dpi=300, levels=30)
      p2          =   heatmap(x/1e3, z/1e3,      Vz',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="Vz",  dpi=300, levels=30)
      
      #st=100; Xv=X[:]; Zv=Z[:];
      #quiver!(Xv[1:step:end]./1e3, Zv[1:step:end]./1e3, gradient=(Vx[1:step:end],Vz[1:step:end]), arrow = :arrow)

      plot(p1,p2); 

      png("HostRockVelocity_$(Dimension)_$(DikeType)")
    end
  

  elseif Dimension=="3D"
    Vel      =   Velocity[:];

    if CreatePlots
      Vx,Vy,Vz    =   Velocity[1],Velocity[2],Velocity[3];
      p1          =   heatmap(x/1e3, z/1e3,      Vx[:,Int((Ny-1)/2),:]',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="2D Vx",  dpi=300, levels=30)
      p2          =   heatmap(x/1e3, z/1e3,      Vz[:,Int((Ny-1)/2),:]',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="Vz",  dpi=300, levels=30)
      
      plot(p1,p2);
      png("HostRockVelocity_$(Dimension)_$(DikeType)")


      # write this to a paraview VTK file, using the package WriteVTK.jl
      #vtkfile = vtk_grid("HostVelocity_3D", Vector(x/1e3), Vector(y/1e3), Vector(z/1e3)) # 3-D
      #vtkfile["Velocity"] = (Vx,Vy,Vz);
      #outfiles = vtk_save(vtkfile)
    end

  end

  return norm(Vel,2);        # return measure of Vel
end 



function test_InjectDike(Dimension="2D", DikeType="ElasticDike", DikeAngle=[45], numDikeInjectionEvents=1; InterpolationMethod="Cubic", AdvectionMethod="RK2")
  # tests dike insertion in the domain including adding tracers

  
  if Dimension=="2D"
    # Model parameters
    W,H                     =   30.0,  30.0;                                # Width, Length, Height
    
    # Define grid
    Nx, Nz                  =   129, 129;                                     # resolution of coarse grid
    dx,dz                   =   W*1e3/(Nx-1), H*1e3/(Nz-1);                         # grid size [m]
    x,z                     =   0:dx:W*1e3, -H*1e3:dz:0;                            # 1D coordinate arrays
    coords                  =   collect(Iterators.product(x,z))               # generate coordinates from 1D coordinate vectors   
    X,Z                     =   (x->x[1]).(coords), (x->x[2]).(coords);       # transfer coords to 3D arrays
    Grid, GridFull,Spacing  =   (x,z), (X,Z), (dx,dz);

    Hdike                   =   1000.0;
    Wdike                   =   20000.0;
    T_in                    =   900.0;

    cen                     =   [W/2;-H/2].*1e3;
  elseif Dimension=="3D"
      # Model parameters
      W,L,H                   =   30., 30., 30.;                                    # Width, Length, Height
      
      # Define coarse grid
      Nx, Ny, Nz              =   129,129,129;                                                    # resolution of coarse grid
      dx,dy,dz                =   W*1e3/(Nx-1), L*1e3/(Ny-1), H*1e3/(Nz-1);                     # grid size [m]
      x,y,z                   =   0:dx:((Nx-1)*dx),  0:dy:((Ny-1)*dy), -((Nz-1)*dz):dz:0.;      # 1D coordinate arrays
      coords                  =   collect(Iterators.product(x,y,z))                             # generate coordinates from 1D coordinate vectors   
      X,Y,Z                   =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);   # transfer coords to 3D arrays
      Grid, GridFull,Spacing  =   (x,y,z), (X,Y,Z), (dx,dy,dz);
      cen                     =   [W/2; L/2; -H/2].*1e3;
      
 
      Hdike                   =   1000.0;
      Wdike                   =   20000.0;
      T_in                    =   900.0;
  end

  # Create BG temperature structure
  GeoT                    =   20;
  T                       =   -Z./1e3.*GeoT;                                             # initial (linear) temperature profile

  # Create dike
  dike                    =   Dike(W=Wdike, H=Hdike,Center=cen,Angle=DikeAngle,Type=DikeType,T=T_in);  # Specify dike 

  # Test the InsertDike routine, which modifies temperature and adds tracers
  nTr_dike                =   1000;
  Tracers                 =   StructArray{Tracer}(undef, 1)                                    # Initialize Tracers structure
  Tracers, Tnew, InjectVol, dike_poly, Velocity    =   InjectDike(Tracers, T, Grid, dike, nTr_dike,  InterpolationMethod=InterpolationMethod, AdvectionMethod=AdvectionMethod);           # Inject first dike
  

  for i=1:numDikeInjectionEvents-1
    T = Tnew;
    Tracers, Tnew, InjectVol, dike_poly, Velocity  =   InjectDike(Tracers, T, Grid, dike, nTr_dike, InterpolationMethod=InterpolationMethod, AdvectionMethod=AdvectionMethod);           # Inject more dikes
  end

  if Dimension=="2D"
    

    if CreatePlots
      Vx = Velocity[1];
      Vz = Velocity[2];
      
      Tr_coord    =   Tracers.coord; Tr_coord = hcat(Tr_coord...)';       # extract array with coordinates of tracers
      p1          =   heatmap(x/1e3, z/1e3,      T',     aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="T",  dpi=300, levels=30)
      p2          =   scatter(Tr_coord[:,1]/1e3, Tr_coord[:,2]/1e3, zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), markersize=5.0, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),title="Tracers")
      p3          =   heatmap(x/1e3, z/1e3,      Vx',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="Vx",  dpi=300, levels=30)
      p4          =   heatmap(x/1e3, z/1e3,      Vz',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="Vz",  dpi=300, levels=30)
     
      plot(p1,p2,p3,p4); 

      png("InsertDike_$(Dimension)_$(DikeType)")
    end
  

  elseif Dimension=="3D"

    if CreatePlots
      Vx = Velocity[1];
      Vz = Velocity[3];

      Tr_coord    =   Tracers.coord; Tr_coord = hcat(Tr_coord...)';       # extract array with coordinates of tracers
      p1          =   heatmap(x/1e3, z/1e3,     T[:,Int(ceil(Ny/2)),:]',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="T",  dpi=300, levels=30)
      p2          =   scatter(Tr_coord[:,1]/1e3, Tr_coord[:,3]/1e3, zcolor = Tracers.T, m = (:inferno , 0.8, Plots.stroke(0.01, :black)), markersize=5.0,title="Tracers",xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),)
      p3          =   heatmap(x/1e3, z/1e3,      Vx[:,Int(ceil(Ny/2)),:]',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="Vx",  dpi=300, levels=30)
      p4          =   heatmap(x/1e3, z/1e3,      Vz[:,Int(ceil(Ny/2)),:]',       aspect_ratio=1, xlims=(x[1]/1e3,x[end]/1e3), ylims=(z[1]/1e3,z[end]/1e3),   c=:inferno, title="Vz",  dpi=300, levels=30)
      
      plot(p1,p2,p3,p4); 
      png("InsertDike_$(Dimension)_$(DikeType)")


      # write this to a paraview VTK file, using the package WriteVTK.jl
      #vtkfile = vtk_grid("InsertDike_3D", Vector(x/1e3), Vector(y/1e3), Vector(z/1e3)) # 3-D
      #vtkfile["Temperature"] = (T);
      #vtkfile["Velocity"]    = (Velocity);
      #outfiles = vtk_save(vtkfile)
    end

  end

  return norm(T[:],2);        
end 


# ===================================================================================================

if 1==1

@testset "Dike_Velocity" begin
  #@test test_HostRockVelocityFromDike("2D", "ElasticDike",[80    ])  ≈   2663.677375120158  atol=1e-8;
  test_HostRockVelocityFromDike("2D", "ElasticDike",[0    ])
  #@test test_HostRockVelocityFromDike("2D", "ElasticDike",[0    ])  ≈   2663.677375120158  atol=1e-8;
  
  @test test_HostRockVelocityFromDike("2D","SquareDike",  [80    ])  ≈   5286.539510870982  atol=1e-8;
  @test test_HostRockVelocityFromDike("3D","SquareDike",  [90; 90])  ≈  13114.877048604001  atol=1e-4;
  @test test_HostRockVelocityFromDike("3D","ElasticDike",[90; 45])   ≈   4762.014274270334  atol=1e-4;
end

# test dike structure
@testset "Dike_Struct" begin
  @test typeof(Dike(Center=[0; 0],Angle=[45],     T=900, Type="SquareDike",  W=1000, H=100 ))==Dike
  @test typeof(Dike(Center=[0; 0],Angle=[45; 90], T=900, Type="ElasticDike", W=1000, H=100 ))==Dike
  @test typeof(Dike(Center=[0; 0],Angle=[45; 90], T=800, Type="ElasticDike", Q=1e6, ΔP=1e7, E=1e10) )==Dike  
end

# Volume of dike
@testset "Dike_Volume" begin
 # @test volume_dike(Dike(Center=[0; 0],Angle=[45; 90], T=800, Type="ElasticDike", Q=1e6, ΔP=1e7, E=1e10))[1] ≈   2539.6349734808196 atol=1e-8;  
  @test volume_dike(Dike(Center=[0; 0],Angle=[45; 90], T=800, Type="ElasticDike", Q=1e6, ΔP=1e7, E=1e10))[1] ≈   634.9087433702049 atol=1e-8;  
  @test volume_dike(Dike(Center=[0; 0],Angle=[45],     T=900, Type="SquareDike",  W=1000, H=100 ))[1] ≈ 100000.0  atol=1e-8;  
end

# Dike insertion algorithm
@testset "Dike_Inject" begin
  @test test_InjectDike("2D", "SquareDike", [80 ],1)    ≈   47525.46469221513 rtol=1e-5;
  
  @test test_InjectDike("2D", "ElasticDike",[45 ],2, InterpolationMethod="Linear")    ≈   48448.71557320294  rtol=1e-4;     # also tests what happens if we add 2 dikes
  @test test_InjectDike("2D", "ElasticDike",[45 ],2, InterpolationMethod="Quadratic") ≈   48770.72282124797  rtol=1e-4;     # also tests what happens if we add 2 dikes
  @test test_InjectDike("2D", "ElasticDike",[45 ],2, InterpolationMethod="Cubic")     ≈   48782.180721943965 rtol=1e-4;     # also tests what happens if we add 2 dikes

  @test test_InjectDike("3D", "ElasticDike",[80; 45])   ≈   519654.9176188356 rtol=1e-5;
  @test test_InjectDike("3D", "SquareDike",[15; -30])   ≈   527521.5507505678  rtol=1e-5;

end

end

#test_InjectDike("2D", "ElasticDike", [80 ],5, InterpolationMethod="Linear", AdvectionMethod="Euler")
#test_InjectDike("3D", "SquareDike",[80; 45], InterpolationMethod="Linear", AdvectionMethod="RK2")



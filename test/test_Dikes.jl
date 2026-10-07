# this file tests various aspects of the advection routines
using MagmaThermoKinematics
using InjectSills
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Plots
using LinearAlgebra
using SpecialFunctions
using Test

#using WriteVTK

const CreatePlots = false      # easy way to deactivate plotting throughout

# ---------------------------------------------------------------------------
# Helper: build the InjectSills AbstractSill that corresponds to a given MTK
# DikeType at the specified center / orientation / size.
#   SquareDike        → SquareDike       (W = full width, same as MTK)
#   ElasticDike / InjectSills / others → PennyShapedSill (W = radius = Wdike/2)
# ---------------------------------------------------------------------------
function _make_sill(DikeType, cen, DikeAngle, Wdike, Hdike, dim)
    if dim == 2
        angle  = Vec1(Float64(DikeAngle[1]))
        center = Point2(cen[1], cen[2]) * m
    else
        angle  = Vec2(Float64(DikeAngle[1]), Float64(DikeAngle[end]))
        center = Point3(cen[1], cen[2], cen[3]) * m
    end
    if DikeType in ("SquareDike", "SquareDike_TopAccretion")
        SquareDike(Center=center, Angle=angle, W=Wdike*m, H=Hdike*m)
    else  # ElasticDike, InjectSills, EllipticalIntrusion, …
        PennyShapedSill(Center=center, Angle=angle,
                        W=(Wdike/2)*m, H=Hdike*m,
                        E=1.5e10Pa, ν=0.3*NoUnits)
    end
end


function test_hostrock_velocity(Dimension="2D", DikeType="ElasticDike", DikeAngle=[45])
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

  # Compute velocity required to create space for dike
  sill = _make_sill(DikeType, cen, DikeAngle, Wdike, Hdike, length(Grid))
  if Dimension == "2D"
      Dx, Dz   = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Z))
      Velocity = (Dx, Dz)
  else
      Dx, Dy, Dz = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Y), Float64.(Z))
      Velocity   = (Dx, Dy, Dz)
  end


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



function test_inject_sills(Dimension="2D", DikeType="ElasticDike", DikeAngle=[45], numDikeInjectionEvents=1; InterpolationMethod="Cubic", AdvectionMethod="RK2")
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

  nTr_dike = 1000
  Tracers  = StructArray{Tracer{Float32}}(undef, 1)                           # Initialize Tracers structure

  sill = _make_sill(DikeType, cen, DikeAngle, Wdike, Hdike, length(Grid))
  Tracers, Tnew, _, _, Velocity = inject_sills(Tracers, T, Grid, sill, T_in, 2, nTr_dike;
                                                InterpolationMethod, AdvectionMethod)
  for _ = 1:numDikeInjectionEvents-1
      T = Tnew
      Tracers, Tnew, _, _, Velocity = inject_sills(Tracers, T, Grid, sill, T_in, 2, nTr_dike;
                                                    InterpolationMethod, AdvectionMethod)
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
  @test test_hostrock_velocity("2D","SquareDike",  [80    ])   ≈   5286.539510870982  rtol=1e-3;
  @test test_hostrock_velocity("3D","SquareDike",  [90; 90])   ≈  13114.877048604001  rtol=1e-3;
  @test test_hostrock_velocity("3D","ElasticDike", [90; 45])   ≈   4762.014274270334  rtol=1e-3;
end

# Dike insertion algorithm
@testset "Dike_Inject" begin
  @test test_inject_sills("2D", "SquareDike", [80 ],1) ≈   47525.465759514336 rtol=1e-4;
  @test test_inject_sills("2D", "ElasticDike",[45 ],2, InterpolationMethod="Linear") ≈   48448.85838494859  rtol=1e-4;
  @test test_inject_sills("2D", "ElasticDike",[45 ],2, InterpolationMethod="Quadratic") ≈   48770.817049970356 rtol=1e-4;
  @test test_inject_sills("2D", "ElasticDike",[45 ],2, InterpolationMethod="Cubic") ≈   48782.27237242118  rtol=1e-4;
  @test test_inject_sills("3D", "ElasticDike",[80; 45]) ≈   519654.91761887114 rtol=1e-4;
  @test test_inject_sills("3D", "SquareDike", [15; -30]) ≈   527521.5507477389  rtol=1e-4;
end

@testset "inject_sills" begin

  # ------------------------------------------------------------------
  # 2-D
  # ------------------------------------------------------------------
  let
    W_dom, H_dom = 30.0, 30.0
    Nx, Nz       = 129, 129
    dx, dz       = W_dom*1e3/(Nx-1), H_dom*1e3/(Nz-1)
    x, z         = 0:dx:W_dom*1e3, -H_dom*1e3:dz:0
    coords       = collect(Iterators.product(x, z))
    X, Z         = (c->c[1]).(coords), (c->c[2]).(coords)
    Grid         = (x, z)
    GeoT         = 20.0
    T            = -Z ./ 1e3 .* GeoT

    Hdike, Wdike = 1000.0, 20000.0
    cen          = [W_dom/2; -H_dom/2] .* 1e3
    T_in         = 900.0

    # inject_sills: basic sanity checks in 2D
    sill2d = PennyShapedSill(
                W      = (Wdike/2)*m,
                H      = Hdike*m,
                E      = 1.5e10*Pa,
                ν      = 0.3*NoUnits,
                Center = Point2(cen[1], cen[2])*m)
    Tr_new  = StructArray{Tracer{Float32}}(undef, 1)
    Tr_new, Tnew_new, InjVol, _, _ = inject_sills(Tr_new, copy(T), Grid, sill2d, T_in, 2, 300)

    @test all(isfinite, Tnew_new)
    @test maximum(Tnew_new) <= T_in + 1e-8
    @test minimum(Tnew_new) >= minimum(T) - 1e-8
    # Injected volume: sill.W.val is the radius, so volume = 4/3*π*r²*(H/2)
    @test InjVol ≈ 4/3*π*(Wdike/2)^2*(Hdike/2)  rtol=1e-6
    # Tracers were added
    @test length(Tr_new) == 300

    # The plotting polygon moves with the host rock by less than the sill opening
    poly0 = InjectSills.dike_polygon(sill2d)
    _, _, _, poly_adv, _ = inject_sills(StructArray{Tracer{Float32}}(undef, 1), copy(T), Grid, sill2d, T_in, 2, 0;
                                        dike_poly=deepcopy(poly0))
    @test maximum(abs.(poly_adv[1] .- poly0[1])) <= Hdike
    @test maximum(abs.(poly_adv[2] .- poly0[2])) <= Hdike

    # Injected volume of other sill types (W is the full width)
    for (sill, V_expected) in (
            (EllipticalIntrusion(Center=Point2(cen[1], cen[2])*m, W=Wdike*m, H=Hdike*m),          4/3*π*(Wdike/2)^2*(Hdike/2)),
            (CylindricalDikeTopAccretion(Center=Point2(cen[1], cen[2])*m, W=Wdike*m, H=Hdike*m),  π*(Wdike/2)^2*Hdike),
            (SquareDike(Center=Point2(cen[1], cen[2])*m, W=Wdike*m, H=Hdike*m),                   Wdike^2*Hdike))
        Tr_s = StructArray{Tracer{Float32}}(undef, 1)
        _, _, InjVol_s, _, _ = inject_sills(Tr_s, copy(T), Grid, sill, T_in, 2, 0)
        @test InjVol_s ≈ V_expected  rtol=1e-12
    end
  end

  # ------------------------------------------------------------------
  # 3-D
  # ------------------------------------------------------------------
  let
    W_dom, L_dom, H_dom = 30.0, 30.0, 30.0
    Nx, Ny, Nz          = 65, 65, 65
    dx, dy, dz          = W_dom*1e3/(Nx-1), L_dom*1e3/(Ny-1), H_dom*1e3/(Nz-1)
    x = 0:dx:(Nx-1)*dx;  y = 0:dy:(Ny-1)*dy;  z = -(Nz-1)*dz:dz:0.0
    coords = collect(Iterators.product(x, y, z))
    X      = (c->c[1]).(coords);  Y = (c->c[2]).(coords);  Z = (c->c[3]).(coords)
    Grid   = (x, y, z)
    GeoT   = 20.0
    T      = -Z ./ 1e3 .* GeoT

    Hdike, Wdike = 1000.0, 20000.0
    cen          = [W_dom/2; L_dom/2; -H_dom/2] .* 1e3
    T_in         = 900.0

    # inject_sills: basic sanity checks in 3D
    sill3d = PennyShapedSill(
                W      = (Wdike/2)*m,
                H      = Hdike*m,
                E      = 1.5e10*Pa,
                ν      = 0.3*NoUnits,
                Center = Point3(cen[1], cen[2], cen[3])*m,
                Angle  = Vec2(0.0, 0.0))
    Tr_new  = StructArray{Tracer{Float32}}(undef, 1)
    Tr_new, Tnew_new, InjVol, _, _ = inject_sills(Tr_new, copy(T), Grid, sill3d, T_in, 2, 300)

    @test all(isfinite, Tnew_new)
    @test maximum(Tnew_new) <= T_in + 1e-8
    @test minimum(Tnew_new) >= minimum(T) - 1e-8
    @test InjVol ≈ 4/3*π*(Wdike/2)^2*(Hdike/2)  rtol=1e-6   # Wdike/2 = sill radius
    @test length(Tr_new) == 300
  end

end

end

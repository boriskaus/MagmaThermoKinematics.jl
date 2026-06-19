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


function test_HostRockVelocityFromDike(Dimension="2D", DikeType="ElasticDike", DikeAngle=[45]; use_inject_sills=false)
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
  if use_inject_sills
      sill     = _make_sill(DikeType, cen, DikeAngle, Wdike, Hdike, length(Grid))
      if Dimension == "2D"
          Dx, Dz   = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Z))
          Velocity = (Dx, Dz)
      else
          Dx, Dy, Dz = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Y), Float64.(Z))
          Velocity   = (Dx, Dy, Dz)
      end
  else
      dike     = Dike(W=Wdike, H=Hdike, Center=cen, Angle=DikeAngle, Type=DikeType, T=T_in)
      Δ        = Hdike
      dt       = 1
      Velocity = HostRockVelocityFromDike(Grid, FullGrid, Δ, dt, dike)
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



function test_InjectDike(Dimension="2D", DikeType="ElasticDike", DikeAngle=[45], numDikeInjectionEvents=1; InterpolationMethod="Cubic", AdvectionMethod="RK2", use_inject_sills=false)
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

  if use_inject_sills
      sill_is = _make_sill(DikeType, cen, DikeAngle, Wdike, Hdike, length(Grid))
      Tracers, Tnew, _, _, Velocity = inject_sills(Tracers, T, Grid, sill_is, T_in, 2, nTr_dike;
                                                    InterpolationMethod=InterpolationMethod, AdvectionMethod=AdvectionMethod)
      for _ = 1:numDikeInjectionEvents-1
          T = Tnew
          Tracers, Tnew, _, _, Velocity = inject_sills(Tracers, T, Grid, sill_is, T_in, 2, nTr_dike;
                                                        InterpolationMethod=InterpolationMethod, AdvectionMethod=AdvectionMethod)
      end
  else
      dike = Dike(W=Wdike, H=Hdike, Center=cen, Angle=DikeAngle, Type=DikeType, T=T_in)
      Tracers, Tnew, InjectVol, dike_poly, Velocity = InjectDike(Tracers, T, Grid, dike, nTr_dike;
                                                                   InterpolationMethod=InterpolationMethod, AdvectionMethod=AdvectionMethod)
      for _ = 1:numDikeInjectionEvents-1
          T = Tnew
          Tracers, Tnew, InjectVol, dike_poly, Velocity = InjectDike(Tracers, T, Grid, dike, nTr_dike;
                                                                       InterpolationMethod=InterpolationMethod, AdvectionMethod=AdvectionMethod)
      end
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


"""
    test_InjectSills_vs_ElasticDike(Dimension, DikeAngle)

Creates the same dike geometry using both `Type="ElasticDike"` and `Type="InjectSills"`,
computes the host-rock velocity field for each, and returns the ratio of the two L2 norms.
A ratio close to 1 confirms that InjectSills reproduces the built-in elastic solution.

Note: PennyShapedSill uses W as the *radius* (half-width), whereas the MTK
`Dike` struct uses W as the full width, so `W_sill = Wdike / 2`.
The sill is created with default Center=(0,0) and Angle=0 because `HostRockVelocityFromDike`
already rotates/shifts the coordinate frame before calling the type-specific branch.
"""
function test_InjectSills_vs_ElasticDike(Dimension="2D", DikeAngle=[0])

    Hdike = 100.0
    Wdike = 20000.0
    E_val = 1.5e10
    ν_val = 0.3

    if Dimension == "2D"
        W, H = 30.0, 30.0
        Nx, Nz = 129, 129
        dx, dz = W*1e3/(Nx-1), H*1e3/(Nz-1)
        x, z   = 0:dx:W*1e3, -H*1e3:dz:0
        coords = collect(Iterators.product(x, z))
        X, Z   = (c->c[1]).(coords), (c->c[2]).(coords)
        Grid, FullGrid = (x, z), (X, Z)
        cen    = [W/2; -H/2] .* 1e3

        sill = PennyShapedSill(W = (Wdike/2)*m, H = Hdike*m,
                                           E = E_val*Pa, ν = ν_val*NoUnits,
                                           Center = Point2(0.0, 0.0)*m)
    else
        W, L, H = 30.0, 40.0, 50.0
        Nx, Ny, Nz = 65, 65, 65
        dx, dy, dz = W*1e3/(Nx-1), L*1e3/(Ny-1), H*1e3/(Nz-1)
        x, y, z    = 0:dx:((Nx-1)*dx), 0:dy:((Ny-1)*dy), -((Nz-1)*dz):dz:0.0
        coords     = collect(Iterators.product(x, y, z))
        X, Y, Z    = (c->c[1]).(coords), (c->c[2]).(coords), (c->c[3]).(coords)
        Grid, FullGrid = (x, y, z), (X, Y, Z)
        cen        = [W/2; L/2; -H/2] .* 1e3

        sill = PennyShapedSill(W = (Wdike/2)*m, H = Hdike*m,
                                           E = E_val*Pa, ν = ν_val*NoUnits,
                                           Center = Point3(0.0, 0.0, 0.0)*m,
                                           Angle  = Vec2(0.0, 0.0))
    end

    Δ = Hdike
    dt = 1.0

    dike_elastic  = Dike(W=Wdike, H=Hdike, Center=cen, Angle=DikeAngle, Type="ElasticDike",
                         T=900.0, E=E_val, ν=ν_val)
    dike_isills   = Dike(W=Wdike, H=Hdike, Center=cen, Angle=DikeAngle, Type="InjectSills",
                         T=900.0, E=E_val, ν=ν_val, sill=sill)

    Vel_elastic   = HostRockVelocityFromDike(Grid, FullGrid, Δ, dt, dike_elastic)
    Vel_isills    = HostRockVelocityFromDike(Grid, FullGrid, Δ, dt, dike_isills)

    norm_elastic  = norm(Vel_elastic[:], 2)
    norm_isills   = norm(Vel_isills[:],  2)

    return norm_isills / norm_elastic   # should be ≈ 1.0
end

# ===================================================================================================

if 1==1

@testset "Dike_Velocity" begin
  # Legacy Dikes.jl path disabled in this branch; keep InjectSills-only checks.
  @test test_HostRockVelocityFromDike("2D","SquareDike",  [80    ], use_inject_sills=true)   ≈   5286.539510870982  rtol=1e-3;
  @test test_HostRockVelocityFromDike("3D","SquareDike",  [90; 90], use_inject_sills=true)   ≈  13114.877048604001  rtol=1e-3;
  @test test_HostRockVelocityFromDike("3D","ElasticDike", [90; 45], use_inject_sills=true)   ≈   4762.014274270334  rtol=1e-3;
end

## Legacy Dikes.jl-specific tests intentionally disabled in this branch:
## - Dike_Struct
## - Dike_Volume
## - InjectSills_vs_ElasticDike (depends on HostRockVelocityFromDike via Dike)

# Dike insertion algorithm
@testset "Dike_Inject" begin
  # InjectSills-only path.
  @test test_InjectDike("2D", "SquareDike", [80 ],1,                          use_inject_sills=true) ≈   47525.465759514336 rtol=1e-4;
  @test test_InjectDike("2D", "ElasticDike",[45 ],2, InterpolationMethod="Linear",    use_inject_sills=true) ≈   48448.85838494859  rtol=1e-4;
  @test test_InjectDike("2D", "ElasticDike",[45 ],2, InterpolationMethod="Quadratic", use_inject_sills=true) ≈   48770.817049970356 rtol=1e-4;
  @test test_InjectDike("2D", "ElasticDike",[45 ],2, InterpolationMethod="Cubic",     use_inject_sills=true) ≈   48782.27237242118  rtol=1e-4;
  @test test_InjectDike("3D", "ElasticDike",[80; 45],                                 use_inject_sills=true) ≈   519654.91761887114 rtol=1e-4;
  @test test_InjectDike("3D", "SquareDike", [15; -30],                                use_inject_sills=true) ≈   527521.5507477389  rtol=1e-4;
end

@testset "inject_sills" begin

  # ------------------------------------------------------------------
  # 2-D: inject_sills should reproduce InjectDike/ElasticDike
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
    # Injected volume is always a true 3D volume [m³] (km³ convention), even in
    # 2D, taken straight from InjectSills.volume (W,H are diameters ⇒ semi-axes
    # W/2,H/2 — the same convention `InjectSills.inside` uses).
    @test InjVol ≈ ustrip(InjectSills.volume(sill2d))  rtol=1e-6
    # Tracers were added
    @test length(Tr_new) == 300
  end

  # ------------------------------------------------------------------
  # 3-D: inject_sills should reproduce InjectDike/ElasticDike
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
    # 3D ⇒ the injected volume is the source's equivalent 3D volume, taken from
    # InjectSills.volume (W,H are diameters ⇒ semi-axes W/2,H/2).
    @test InjVol ≈ ustrip(InjectSills.volume(sill3d))  rtol=1e-6
    @test length(Tr_new) == 300
  end

end

@testset "inject_sills sphere sources (Mogi/McTigue)" begin
  # Regression test for the "interior blows up" bug: a Mogi/McTigue sphere
  # has a near-singular displacement in its core. inject_sills must (a) not
  # produce NaN/Inf, (b) keep T bounded by [background, T_in] (no blow-up),
  # and (c) still seed tracers inside the (W/H-less) sphere source. The grid
  # is deliberately placed so a node coincides with the source centre (the
  # worst case, where the analytic field is NaN/∞).

  # ---- 2D : node exactly on the source centre ----
  let
    Nx, Nz = 101, 101
    x = range(-5000.0, 5000.0, length=Nx)      # x=0 is a grid node
    z = range(-10000.0, 0.0,   length=Nz)      # z=-5000 is a grid node
    Grid  = (x, z)
    T     = [20.0 - z[j]/1e3*20 for i in 1:Nx, j in 1:Nz]
    T_in  = 1000.0

    src = MogiSphere(Center = Point2(0.0, -5000.0)*m, r = 1500.0m,
                     ΔP = 50e6*Pa, G = 10e9*Pa, ν = 0.25*NoUnits)
    Tr  = StructArray{Tracer{Float32}}(undef, 1)
    Tr, Tnew, InjVol, _, _ = inject_sills(Tr, copy(T), Grid, src, T_in, 2, 200)

    @test all(isfinite, Tnew)                       # no NaN/Inf anywhere
    @test maximum(Tnew) <= T_in + 1e-8              # interior did not blow up
    @test minimum(Tnew) >= minimum(T) - 1e-8
    @test InjVol > 0                                # injected volume from InjectSills
    # M7: injected volume is the source's equivalent 3D volume (km³ convention),
    # even in 2D (see inject_sills).
    @test InjVol ≈ ustrip(InjectSills.volume(src))  rtol=1e-8
    @test length(Tr) == 200                         # tracers seeded inside the sphere
    # every seeded tracer is actually inside the sphere
    @test all(t -> InjectSills.inside(Point2(t.coord[1], t.coord[2]), src), Tr)
    # M4: the singular core does not leak outward. A Mogi source is long-range
    # (∝1/r²) so the far field DOES respond — but only by a tiny, bounded amount.
    # Domain corners (≈6 km, ≫ the 1.5 km source radius) must move ≪ 1 K, while
    # the interior is legitimately reset to T_in (checked above). Measured corner
    # response ≈ 0.0036 K; bound generously at 0.01 K.
    @test abs(Tnew[1,1]     - T[1,1])     < 1e-2
    @test abs(Tnew[end,end] - T[end,end]) < 1e-2
    @test abs(Tnew[1,end]   - T[1,end])   < 1e-2
  end

  # ---- 3D ----
  let
    Nx, Ny, Nz = 51, 51, 51
    x = range(-5000.0, 5000.0, length=Nx)
    y = range(-5000.0, 5000.0, length=Ny)
    z = range(-10000.0, 0.0,   length=Nz)
    Grid = (x, y, z)
    T    = [20.0 - z[k]/1e3*20 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    T_in = 1000.0

    src = MogiSphere(Center = Point3(0.0, 0.0, -5000.0)*m, r = 1500.0m,
                     ΔP = 50e6*Pa, G = 10e9*Pa, ν = 0.25*NoUnits)
    Tr  = StructArray{Tracer{Float32}}(undef, 1)
    Tr, Tnew, InjVol, _, _ = inject_sills(Tr, copy(T), Grid, src, T_in, 2, 200)

    @test all(isfinite, Tnew)
    @test maximum(Tnew) <= T_in + 1e-8
    @test minimum(Tnew) >= minimum(T) - 1e-8
    @test InjVol > 0
    @test length(Tr) == 200
  end
end

end

#test_InjectDike("2D", "ElasticDike", [80 ],5, InterpolationMethod="Linear", AdvectionMethod="Euler")
#test_InjectDike("3D", "SquareDike",[80; 45], InterpolationMethod="Linear", AdvectionMethod="RK2")

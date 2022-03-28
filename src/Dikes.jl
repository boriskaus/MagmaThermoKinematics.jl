"""

This contains a number of routines that are related to inserting new dikes to the simulation,
defining a velocity field that "opens" the host rock accordingly and to inserting the dike temperature to the
temperature field

"""

"""
    Structure that holds the geometrical parameters of the dike, which are slightly different
    depending on whether we consider a 2D or a 3D case

    General form:
        Dike(Width=.., Thickness=.., Center=[], Angle=[], Type="..", T=.., ΔP=.., E=.., ν=.., Q=..)

        with:

            [Width]:      width of dike  (optional, will be computed automatically if ΔP and Q are specified)

            [Thickness]:  (maximum) thickness of dike (optional, will be computed automatically if ΔP and Q are specified)
    
            Center:     center of the dike
                            2D - [x; z]
                            3D - [x; y; z]
            
            Angle:      Dip (and strike) angle of dike
                            2D - [Dip]
                            3D - [Strike; Dip]
            
            Type:           Type of dike
                            "SquareDike"    -   square dike area   
                            "SquareDike_TopAccretion"           -   square dike area, which grows by underaccreting   
                            "CylindricalDike_TopAccretion"      -   cylindrical dike area, which grows by underaccreting   
                            "CylindricalDike_TopAccretion_FullModelAdvection"      -   cylindrical dike area, which grows by underaccreting; also material to the side of the dike is moved downwards   
                            
                            "ElasticDike"   -   penny-shaped elastic dike in elastic halfspace
            
            T:          Temperature of the dike [Celcius]   
            
            ν:          Poison ratio of host rocks
            
            E:          Youngs modulus of host rocks [Pa]
            
            [ΔP]:       Overpressure of dike w.r.t. host rock [Pa], (optional in case we want to compute width/length directly)

            [Q]:        Volume of magma within dike [m^3], 
            
           
    All parameters can be specified through keywords as shown above. 
    If keywords are not given, default parameters are employed.
    
 The 
    
"""
@with_kw struct Dike    # stores info about dike
    # Note: since we utilize the "Parameters.jl" package, we can add more keywords here w/out breaking the rest of the code 
    #
    # We can also define only a few parameters here (like Q and ΔP) and compute Width/Thickness from that
    # Or we can define thickness 
    Angle       ::  Vector{Float64} =   [0.]                                  # Strike/Dip angle of dike
    Type        ::  String          =   "SquareDike"                          # Type of dike
    T           ::  Float64         =   950.0                                 # Temperature of dike
    E           ::  Float64         =   1.5e10                                # Youngs modulus (only required for elastic dikes)
    ν           ::  Float64         =   0.3                                   # Poison ratio of host rocks
    ΔP          ::  Float64         =   1e6;                                  # Overpressure of elastic dike
    Q           ::  Float64         =   1000;                                 # Volume of elastic dike
    W           ::  Float64         =   (3*E*Q/(16*(1-ν^2)*ΔP))^(1.0/3.0);    # Width of dike/sill   
    H           ::  Float64         =   8*(1-ν^2)*ΔP*W/(π*E);                 # (maximum) Thickness of dike/sill
    Center      ::  Vector{Float64} =   [20e3 ; -10e3]                        # Center
    Phase       ::  Int64           =   2;                                    # Phase of newly injected magma
end

struct DikePoly    # polygon that describes the geometry of the dike (only in 2D)
    x::Float64          # x-coordinates
    z::Float64          # z-coordinates
end

"""
    This injects a dike in the computational domain in an instantaneous manner,
    while "pushing" the host rocks to the sides. 

    The orientation and the type of the dike are described by the structure     

    General form:
        T, Velocity, VolumeInjected = InjectDike(Tracers, T, Grid, FullGrid, dike, nTr_dike; AdvectionMethod="RK2", InterpolationMethod="Quadratic")

    with:
        T:          Temperature grid (will be modified)

        Tracers:    StructArray that contains the passive tracers (will be modified)

        Grid:       regular grid on which the temperature is defined
                    2D - (X,Z)
                    3D - (X,Y,Z)

        FullGrid:   2D or 3D matrixes with the full grid coordinates
                    2D - (X,Z)
                    3D - (X,Y,Z)

        nTr_dike:   Number of new tracers to be injected into the new dike area

    optional input parameters with keywords (add them with: AdvectionMethod="RK4", etc.):

        AdvectionMethod:    Advection algorithm 
                    "Euler"     -    1th order accurate Euler timestepping
                    "RK2"       -    2nd order Runga Kutta advection method [default]
                    "RK4"       -    4th order Runga Kutta advection method
                
        InterpolationMethod: Interpolation Algorithm to interpolate data on advected points 
                    
                    Note:  higher order is more accurate for smooth fields, but if there are very sharp gradients, 
                        it may result in a 'Gibbs' effect that has over and undershoots.   

                    "Linear"    -    Linear interpolation
                    "Quadratic" -    Quadratic spline
                    "Cubic"     -    Cubic spline

"""
function InjectDike(Tracers, T::Array, Grid, dike::Dike, nTr_dike::Int64; AdvectionMethod="RK2", InterpolationMethod="Linear")

    # Some notes on the algorithm:
    #   For computational reasons, we do not open the dike at once, but in sufficiently small pseudo timesteps
    #   Sufficiently small implies that the motion per "pseudotimestep" cannot be more than 0.5*{dx|dy|dz}
    
    @unpack H   =   dike
    dim         =   length(Grid);
    Spacing     =   Vector{Float64}(undef, dim);
    
    if dim==2
        coords      =   collect(Iterators.product(Grid[1],Grid[2]))                             # generate coordinates from 1D coordinate vectors   
        X,Z         =   (x->x[1]).(coords), (x->x[2]).(coords);    
        GridFull    =   (X,Z); 
    elseif dim==3
        coords      =   collect(Iterators.product(Grid[1],Grid[2],Grid[3]))                     # generate coordinates from 1D coordinate vectors   
        X,Y,Z       =   (x->x[1]).(coords), (x->x[2]).(coords), (x->x[3]).(coords);     
        GridFull    =   (X,Y,Z); 
    end

    for i=1:dim
        Spacing[i] = Grid[i][2] - Grid[i][1];
    end
    d           =   minimum(Spacing)*0.5;                              # maximum distance the dike can open per pseudotimestep 
    nsteps      =   maximum([ceil(H/d), 2]);                           # the number of steps (>=10)

    # Compute velocity required to create space for dike
    Δ           =   H/(nsteps);
    dt          =   1.0
    Velocity    =  HostRockVelocityFromDike(Grid,GridFull, Δ, dt,dike);       # compute velocity field
 
    # Move hostrock & already existing tracers to the side to create space for new dike
    Tnew        =   zeros(size(T));
    for ipseudotime=1:nsteps 
        Tnew        =   AdvectTemperature(T, Grid,  GridFull, Velocity, dt, AdvectionMethod, InterpolationMethod);                      # use interpolation of velocity from grid to advect T
        #Tnew    =   AdvectTemperature(T, Grid,  GridFull, Velocity, dt, AdvectionMethod, InterpolationMethod, "FromDike", dike, Δ);    # optional method, in which we use the analytical velocity to advect T
        
        if isassigned(Tracers,1)
            AdvectTracers!(Tracers,  Grid,    Velocity, dt);
        end
        T      .=   Tnew;
    end

    # Insert dike in T profile and add new tracers
    Tnew,Tracers        =   AddDike(T, Tracers, Grid,dike, nTr_dike);                 # Add dike to T-field & insert tracers within dike

    # Compute volume of newly injected magma
    Area,InjectedVolume =   volume_dike(dike)

    return Tracers, Tnew, InjectedVolume, Velocity

end

#--------------------------------------------------------------------------
"""
    Host rock velocity obtained during opening of dike

    General form:
        Velocity = HostRockVelocityFromDike( Grid, Points, Δ, dt, dike);

        with:

            Grid: coordinates of regular grid @ which we compute velocity
                    2D - [X; Z]
                    3D - [x; y; z]

            dike: structure that holds info about dike
            dt:   time in which the full dike is opened
            
    Note: the velocity is computed in such a manner that a maximum opening 
        increment of Δ = Vmax*dt is obtained after this timestep


"""
function HostRockVelocityFromDike( Grid, Points, Δ, dt, dike::Dike)

    # Prescibe the velocity field to inject the dike with given orientation
    # 
    dim = length(Grid);
    if dim==2
        #X,Z          =  Points[1], Points[2];
         
        # Rotate and shift coordinate system into 'dike' reference frame
        @unpack Angle,Type = dike
        α           =   Angle[1];

        RotMat      =   SMatrix{2,2}([cosd(α) -sind(α); sind(α) cosd(α)]);    # 2D rotation matrix
       # Xrot,Zrot   =   zeros(size(Points[1])), zeros(size(Points[2]));

        Points[1]           .= Points[1] .- dike.Center[1];
        Points[2]           .= Points[2] .- dike.Center[2];
        RotatePoints_2D!(Points[1],Points[2],  Points[1], Points[2], RotMat)

        Vx_rot, Vz_rot  = zeros(size( Points[1])), zeros(size( Points[1]));
        Vx, Vz          = zeros(size( Points[1])), zeros(size( Points[1]));
            
        if Type=="SquareDike"
            @unpack H,W = dike
            Vint    =  Δ/dt/2.0;                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
                
            Vz_rot[(Points[2] .<= 0) .& (abs.(Points[1]).<= W/2.0)]  .= -Vint;
            Vz_rot[(Points[2] .>  0) .& (abs.(Points[1]).<  W/2.0)]  .=  Vint;

            Vx_rot[abs.(Points[1]).<W]          .=   0.0;      # set radial velocity to zero at left boundary
        elseif Type=="SquareDike_TopAccretion"
                @unpack H,W = dike
                Vint    =  Δ/dt/1.0;                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
                    
                Vz_rot[(Points[2] .<= 0) .& (abs.(Points[1]).<= W/2.0)]  .= -Vint;
                #Vz_rot[(Points[2] .>  0) .& (abs.(Points[1]).<  W/2.0)]  .=  Vint;
    
                Vx_rot[abs.(Points[1]).<W]          .=   0.0;      # set radial velocity to zero at left boundary
    
        elseif   Type=="CylindricalDike_TopAccretion" || Type=="CylindricalDike_TopAccretion_FullModelAdvection"
            @unpack H,W = dike
            Vint    =  Δ/dt/1.0;                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
                
            Vz_rot[(Points[2] .<= 0) .& ( (Points[1].^2 + Points[2].^2) .<= (W/2.0).^2)]  .= -Vint;
            #Vz_rot[(Points[2] .>  0) .& (abs.(Points[1]).<  W/2.0)]  .=  Vint;

            Vx_rot[abs.(Points[1]).<W]          .=   0.0;      # set radial velocity to zero at left boundary

        elseif  Type=="CylindricalDike_TopAccretion_FullModelAdvection"
            @unpack H,W = dike
            Vint    =  Δ/dt/1.0;                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
                
            Vz_rot[(Points[2] .<= 0) ]  .= -Vint;
            #Vz_rot[(Points[2] .>  0) .& (abs.(Points[1]).<  W/2.0)]  .=  Vint;

            Vx_rot[abs.(Points[1]).<=W] .=   0.0;      # set radial velocity to zero at left boundary

        elseif Type=="ElasticDike"
                @unpack H,W = dike
                Vint    =  Δ/dt;                            # open the dike by a maximum amount of Δ in one dt (no 1/2 as that is taken care off inside the routine below)
                
Threads.@threads for i in eachindex(Vz_rot)
                    # use elastic dike solution to compute displacement
                    Displacement, Bmax = DisplacementAroundPennyShapedDike(dike, SVector(Points[1][i], Points[2][i]), dim);

                    Displacement    .=  Displacement/Bmax;     # normalize such that 1 is the maximum
                        
                    Vz_rot[i]       =   Vint.*Displacement[2];
                    Vx_rot[i]       =   Vint.*Displacement[1];      
             
                end

        else
            error("Unknown Dike Type: $Type")
        end

        # "unrotate" vector fields and points using the transpose of RotMat
        RotatePoints_2D!(Vx,Vz, Vx_rot,Vz_rot, RotMat')
        RotatePoints_2D!(Points[1], Points[2],  Points[1],  Points[2],   RotMat')

        Points[1]   .= Points[1] .+ dike.Center[1];
        Points[2]   .= Points[2] .+ dike.Center[2];

        return (Vx, Vz);

    else
        @unpack Angle,Type   =   dike;
        α,β             =   Angle[1], Angle[end];
        RotMat_y        =   SMatrix{3,3}([cosd(α) 0.0 -sind(α); 0.0 1.0 0.0; sind(α) 0.0 cosd(α)  ]);                      # perpendicular to y axis
        RotMat_z        =   SMatrix{3,3}([cosd(β) -sind(β) 0.0; sind(β) cosd(β) 0.0; 0.0 0.0 1.0  ]);                      # perpendicular to z axis
        RotMat          =   RotMat_y*RotMat_z;

       # Xrot,Yrot,Zrot  =   zeros(size(Points[1])),  zeros(size(Points[2])), zeros(size(Points[3]));
        Points[1]       .=  Points[1] .- dike.Center[1];
        Points[2]       .=  Points[2] .- dike.Center[2];
        Points[3]       .=  Points[3] .- dike.Center[3];
        RotatePoints_3D!(Points[1],Points[2],Points[3], Points[1],Points[2],Points[3], RotMat)                 # rotate coordinates 
       
        Vx_rot, Vy_rot, Vz_rot  = zeros(size(Points[1])), zeros(size(Points[2])), zeros(size(Points[3]));
        Vx, Vy, Vz              = zeros(size(Points[1])), zeros(size(Points[2])), zeros(size(Points[3]));
        
        if Type=="SquareDike"
            @unpack H,W = dike                          # Dimensions of square dike
            Vint    =  Δ/dt/2.0;                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
                
            Vz_rot[(Points[3].<0) .& (abs.(Points[1]).<W/2.0) .& (abs.(Points[2]).<W/2.0)]  .= -Vint;
            Vz_rot[(Points[3].>0) .& (abs.(Points[1]).<W/2.0) .& (abs.(Points[2]).<W/2.0)]  .=  Vint;

            Vx_rot[abs.(Points[1]).<W]          .=   0.0;      # set radial velocity to zero at left boundary
            Vy_rot[abs.(Points[2]).<W]          .=   0.0;      # set radial velocity to zero at left boundary

        elseif  (Type=="SquareDike_TopAccretion") 
                @unpack H,W = dike                          # Dimensions of square dike
                Vint    =  Δ/dt/1.0;                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
                    
                Vz_rot[(Points[3].<0) .& (abs.(Points[1]).<W/2.0) .& (abs.(Points[2]).<W/2.0)]  .= -Vint;
               # Vz_rot[(Points[3].>0) .& (abs.(Points[1]).<W/2.0) .& (abs.(Points[2]).<W/2.0)]  .=  Vint;
    
                Vx_rot[abs.(Points[1]).<W]          .=   0.0;      # set radial velocity to zero at left boundary
                Vy_rot[abs.(Points[2]).<W]          .=   0.0;      # set radial velocity to zero at left boundary
        elseif  (Type=="CylindricalDike_TopAccretion") || (Type=="CylindricalDike_TopAccretion_FullModelAdvection")
                @unpack H,W = dike                          # Dimensions of square dike
                Vint    =  Δ/dt/1.0;                        # open the dike by a maximum amount of Δ in one dt (2=because we open 2 sides)
                    
                Vz_rot[(Points[3] .<= 0.0) .& ( (Points[1].^2 + Points[2].^2).<=(W/2.0).^2) ]  .= -Vint;
               # Vz_rot[(Points[3].>0) .& (abs.(Points[1]).<W/2.0) .& (abs.(Points[2]).<W/2.0)]  .=  Vint;
    
                Vx_rot[abs.(Points[1]).<= W]          .=   0.0;      # set radial velocity to zero at left boundary
                Vy_rot[abs.(Points[2]).<= W]          .=   0.0;      # set radial velocity to zero at left boundary
    
        elseif Type=="ElasticDike"
                @unpack H,W = dike                          # Dimensions of dike
                Vint    =  Δ/dt;                            # open the dike by a maximum amount of Δ in one dt (no 1/2 as that is taken care off inside the routine below)
                    
                Threads.@threads for i=firstindex(Vx_rot):lastindex(Vx_rot)
                 
                    # use elastic dike solution to compute displacement
                    Displacement, Bmax  = DisplacementAroundPennyShapedDike(dike, SVector(Points[1][i], Points[2][i], Points[3][i]), dim);

                    Displacement        .=  Displacement/Bmax;     # normalize such that 1 is the maximum
                    
                    Vz_rot[i]           =   Vint.*Displacement[3];
                    Vy_rot[i]           =   Vint.*Displacement[2];      
                    Vx_rot[i]           =   Vint.*Displacement[1];      

                end

        else
            error("Unknown Dike Type: $Type")
        end

        # "unrotate" vector fields
        RotatePoints_3D!(Vx,Vy,Vz, Vx_rot,Vy_rot,Vz_rot, RotMat')           # rotate velocities back
        RotatePoints_3D!( Points[1], Points[2], Points[3], Points[1]  ,Points[2]  ,Points[3]  , RotMat')           # rotate coordinates back
        
        Points[1]   .=  Points[1] .+ dike.Center[1];
        Points[2]   .=  Points[2] .+ dike.Center[2];
        Points[3]   .=  Points[3] .+ dike.Center[3];

        return (Vx, Vy, Vz);

    end


end

function  RotatePoints_2D!(Xrot,Zrot, X,Z, RotMat)
    @simd for i in eachindex(X) # linear indexing
        pt_rot      =   RotMat*SVector(X[i], Z[i]);
        Xrot[i]     =   pt_rot[1]
        Zrot[i]     =   pt_rot[2]
    end
end

function  RotatePoints_3D!(Xrot,Yrot,Zrot, X,Y,Z, RotMat)
    @simd for i in eachindex(X) # linear indexing
        pt_rot      =   RotMat*SVector(X[i], Y[i], Z[i]);

        Xrot[i]     =   pt_rot[1]
        Yrot[i]     =   pt_rot[2]
        Zrot[i]     =   pt_rot[3]
    end
end

"""
    dike_poly = CreatDikePolygon(dike::Dike)

    Creates a new dike polygon with given orientation and width. 
    This polygon is used for plotting, and described by the struct dike

    Different Types are available:

"""
function CreatDikePolygon(dike::Dike)
    @unpack Type = dike;

    if Type=="SquareDike"

        # unit dike
        poly = StructArray([DikePoly(-1.0, -1.0), DikePoly( 1.0, -1.0),
                            DikePoly( 1.0,  1.0), DikePoly(-1.0,  1.0)]);
        
        push!(poly,poly[1]); # close polygon

        # scale & rotate dike
        @unpack Angle = dike;   
        α      = Angle[end];
        RotMat = [cosd(-α) -sind(-α); sind(-α) cosd(-α)]; 
        for i=1:length(poly)
            @unpack W, H, Center = dike;   
            
            pt      = [LazyRow(poly, i).x*W/2.0; LazyRow(poly, i).z*H/2.0];
            pt_rot  = RotMat*pt;    # rotate
            
            LazyRow(poly, i).x = pt_rot[1] + Center[1];   # shift
            LazyRow(poly, i).z = pt_rot[2] + Center[2];
        end
    
    elseif Type=="ElasticDike"
        error("To be added")
    
    else
        error("Unknown dike type $Type")
    end

    return poly 
end


"""
    in = isinside_dike(pt, dike::Dike)

    Computes if a point [pt] is inside a dike area or not depending on the type of dike

"""
function isinside_dike(pt, dike::Dike)
    # important: this is a "unit" dike, which has the center at [0,0,0] and width given by dike.Size
    dim =   length(pt)
    in  =   false;
    @unpack Type,W,H = dike;
    if Type=="SquareDike"
        if  dim==2
            if  (abs(pt[1])  < W/2.0) & (abs(pt[end])< H/2.0)
                in = true
            end
        elseif dim==3
            if  (abs(pt[1])  < W/2.0) & (abs(pt[end])< H/2.0) & (abs(pt[2])< W/2.0)
                in = true
            end
        end
    elseif  (Type=="SquareDike_TopAccretion") || (Type=="CylindricalDike_TopAccretion") || (Type=="CylindricalDike_TopAccretion_FullModelAdvection")
        if  dim==2
            if  (abs(pt[1])  <= W/2.0) & (abs(pt[end]) <= H/2.0)
                in = true
            end
        elseif dim==3
            error("add 3D case here")
            if  (abs(pt[1])  < W/2.0) & (abs(pt[end])< H/2.0) & (abs(pt[2])< W/2.0)
                in = true
            end
        end
    elseif Type=="ElasticDike"
        eq_ellipse = 100.0;

        if dim==2
            eq_ellipse = (pt[1]^2.0)/((W/2.0)^2.0) + (pt[2]^2.0)/((H/2.0)^2.0); # ellipse
        elseif dim==3
            # radius = sqrt(*)x^2+y^2)
            eq_ellipse = (pt[1]^2.0 + pt[2]^2.0)/((W/2.0)^2.0) + (pt[3]^2.0)/((H/2.0)^2.0); # ellipsoid
        else
            error("Unknown # of dimensions: $dim")
        end

        if eq_ellipse <= 1.0
            in = true;
        end

    else
        error("Unknown dike type $Type")
    end

    return in 
end


"""
    A,V = volume_dike(dike::Dike)

    Returns the area A and volume V of the injected dike
    
    In 2D, the volume is compute by assuming a penny-shaped dike, with length=width
    In 3D, the cross-sectional area in x-z direction is returned 

"""
function volume_dike(dike::Dike)
    # important: this is a "unit" dike, which has the center at [0,0,0] and width given by dike.Size
    @unpack W, H, Type =dike

    if Type=="SquareDike"
        area    = W*H;                  #  (in 2D, in m^2)
        volume  = W*W*H;                #  (equivalent 3D volume, in m^3)
    elseif Type=="SquareDike_TopAccretion"
        area    = W*H;                  #  (in 2D, in m^2)
        volume  = W*W*H;                #  (equivalent 3D volume, in m^3)
    elseif (Type=="CylindricalDike_TopAccretion") || (Type=="CylindricalDike_TopAccretion_FullModelAdvection")
        area    = W*H;                  #  (in 2D, in m^2)
        volume  = pi*(W/2.0)^2*H;       #  (equivalent 3D volume, in m^3)

    elseif Type=="ElasticDike"
        area    = pi*W*H                #   (in 2D, in m^2)
        volume  = 4/3*pi*W*W*H          #   (equivalent 3D volume, in m^3)
    else
        error("Unknown dike type $Type")
    end

    return area,volume;

end

#--------------------------------------------------------------------------
"""
    T, Tracers, dike_poly = AddDike(T,Tracers, Grid, dike,nTr_dike)
    
Adds a dike, described by the dike polygon dike_poly, to the temperature field T (defined at points Grid).
Also adds nTr_dike new tracers randomly distributed within the dike, to the 
tracers array Tracers.

"""
function AddDike(Tfield,Tr, Grid,dike, nTr_dike)

    dim         =   length(Grid);
    @unpack Angle,Center,W,H,T, Phase = dike;
    PhaseDike = Phase;
    
    if dim==2
        α           =    Angle[1];
        RotMat      =    SMatrix{2,2}([cosd(α) -sind(α); sind(α) cosd(α)]); 
        
    elseif dim==3
        α,β             =   Angle[1], Angle[end];
        RotMat_y        =   SMatrix{3,3}([cosd(α) 0.0  -sind(α); 0.0 1.0 0.0; sind(α) 0.0 cosd(α)  ]);                      # perpendicular to y axis
        RotMat_z        =   SMatrix{3,3}([cosd(β) -sind(β) 0.0; sind(β) cosd(β) 0.0; 0.0 0.0 1.0   ]);                      # perpendicular to z axis
        RotMat          =   RotMat_y*RotMat_z;
    end
  
    # Add dike to temperature field
    if dim==2
        x,z = Grid[1], Grid[2];
        for ix=1:length(x)
            for iz=1:length(z)  
                pt      =   SVector(x[ix],z[iz]) - Center;
                pt_rot  =   RotMat*pt;                      # rotate
                in      =   isinside_dike(pt_rot, dike);
                if in
                    Tfield[ix,iz] = T;
                end
            end
        end

    elseif dim==3
        x,y,z = Grid[1], Grid[2], Grid[3]
        for ix=1:length(x)
            for iy=1:length(y)
                for iz=1:length(z)
                    pt      = SVector(x[ix], y[iy], z[iz]) - Center;
                    pt_rot  = RotMat*pt;          # rotate and shift
                    in      = isinside_dike(pt_rot, dike);
                    if in
                        Tfield[ix,iy,iz] = T;
                    end
                end
            end
        end

    end

    # Add new tracers to the dike area
    for iTr=1:nTr_dike
       
        # 1) Randomly initialize tracers to the approximate dike area
        pt      = rand(dim,1) .- 0.5*ones(dim,1);
        if dim==2
            Size = [W; H];
        else
            Size = [W; W; H];
        end

        pt      = pt.*Size;
        pt_rot  = (RotMat')*pt .+ Center[:];          # rotate backwards (hence the transpose!) and shift

        # 2) Make sure that they are inside the dike area   
        in      = isinside_dike(pt, dike);
       
        # 3) Add them to the tracers structure
        if in   # we are inside the dike
            
            if      dim==2; pt_new = [pt_rot[1]; pt_rot[2]];
            elseif  dim==3; pt_new = [pt_rot[1]; pt_rot[2]; pt_rot[3]]; end
            
            if !isassigned(Tr,1)
                number  =   1;
            else     
                number  =   Tr.num[end]+1;  
            end

            new_tracer  =   Tracer(num=number, coord=pt_new, T=T, Phase=PhaseDike);          # Create new tracer
            
            if !isassigned(Tr,1)
                StructArrays.foreachfield(v -> deleteat!(v, 1), Tr)         # Delete first (undefined) row of tracer StructArray. Assumes that Tr is defined as Tr = StructArray{Tracer}(undef, 1)

                Tr      =   StructArray([new_tracer]);                      # Create tracer array
            else
                push!(Tr, new_tracer);                                      # Add new point to existing array
            end
        end

    end

    return Tfield, Tr;


end




"""
    This computes the displacement around a fluid-filled penny-shaped sill that is 
    inserted inside in an infinite elastic halfspace.

    Displacement, Bmax, p = DisplacementAroundPennyShapedDike(dike, CartesianPoint)

    with:

            dike:           Dike structure, containing info about the dike

            CartesianPoint: Coordinate of the point @ which we want to compute displacments
                        2D - [dx;dz]
                        3D - [dx;dy;dz]
            
            Displacement:   Displacements of that point  
                        2D - [Ux;Uz]
                        3D - [Ux;Uy;Uz]

            Bmax:           Max. opening of the dike 
            p:              Overpressure of dike
    
    Reference: 
        Sun, R.J., 1969. Theoretical size of hydraulically induced horizontal fractures and 
        corresponding surface uplift in an idealized medium. J. Geophys. Res. 74, 5995–6011. 
        https://doi.org/10.1029/JB074i025p05995

        Notes:    
            - We employ equations 7a and 7b from that paper, which assume that the dike is in a 
                horizontal (sill-like) position; rotations have to be performed outside this routine
            
            - The center of the dike should be at [0,0,0]
            
            - This does not account for the presence of a free surface. 
            
            - The values are in absolute displacements; this may have to be normalized

"""
function DisplacementAroundPennyShapedDike(dike::Dike, CartesianPoint::SVector, dim)

    # extract required info from dike structure
    @unpack ν,E,W, H = dike;

    Displacement = Vector{Float64}(undef, dim);

    # Compute r & z; note that the Sun solution is defined for z>=0 (vertical)
    if      dim==2; r = sqrt(CartesianPoint[1]^2);                       z = abs(CartesianPoint[2]); 
    elseif  dim==3; r = sqrt(CartesianPoint[1]^2 + CartesianPoint[2]^2); z = abs(CartesianPoint[3]); end

    if r==0; r=1e-3; end

    B::Float64   =  H;                          # maximum thickness of dike
    a::Float64   =  W/2.0;                      # radius

    # note, we can either specify B and a, and compute pressure p and injected volume Q
    # Alternatively, it is also possible to:
    #       - specify p and a, and compute B and Q 
    #       - specify volume Q & p and compute radius and B 
    #
    # What is best to do is to be decided later (and doesn't change the code below) 
    Q   =   B*(2pi*a.^2)/3.0;               # volume of dike (follows from eq. 9 and 10a)
    p   =   3E*Q/(16.0*(1.0 - ν^2)*a^3);    # overpressure of dike (from eq. 10a) = 3E*pi*B/(8*(1-ν^2)*a)

    # Compute displacement, using complex functions
    R1  =   sqrt(r^2. + (z - im*a)^2);
    R2  =   sqrt(r^2. + (z + im*a)^2);

    # equation 7a:
    U   =   im*p*(1+ν)*(1-2ν)/(2pi*E)*( r*log( (R2+z+im*a)/(R1 +z- im*a)) 
                                                - r/2*((im*a-3z-R2)/(R2+z+im*a) 
                                                + (R1+3z+im*a)/(R1+z-im*a)) 
                                                - (2z^2 * r)/(1 -2ν)*(1/(R2*(R2+z+im*a)) -1/(R1*(R1+z-im*a))) 
                                                + (2*z*r)/(1-2ν)*(1/R2 - 1/R1) );
    # equation 7b:
    W   =       2*im*p*(1-ν^2)/(pi*E)*( z*log( (R2+z+im*a)/(R1+z-im*a)) 
                                                - (R2-R1) 
                                                - 1/(2*(1-ν))*( z*log( (R2+z+im*a)/(R1+z-im*a)) - im*a*z*(1/R2 + 1/R1)) );

    # Displacements are the real parts of U and W. 
    #  Note that this is the total required elastic displacement (in m) to open the dike.
    #  If we only want to open the dike partially, we will need to normalize these values accordingly (done externally)  
    Uz   =  real(W);  # vertical displacement should be corrected for z<0
    Ur   =  real(U);
    if (CartesianPoint[end]<0); Uz = -Uz; end
    if (CartesianPoint[1]  <0); Ur = -Ur; end
    
    if      dim==2
        Displacement = [Ur;Uz] 
    elseif  dim==3
        # Ur denotes the radial displacement; in 3D we have to decompose this in x and y components
        x   = abs(CartesianPoint[1]); y = abs(CartesianPoint[2]);
        Ux  = x/r*Ur; Uy = y/r*Ur;

        Displacement = [Ux;Uy;Uz] 
    end

    return Displacement, B, p   
end

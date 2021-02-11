"""

This contains a number of routines that are related to inserting new dikes to the simulation,
defining a velocity field that "opens" the host rock accordingly and to inserting the dike temperature to the
temperature field

"""

"""
    Structure that holds the geometrical parameters of the dike, which are slightly different
    depending on whether we consider a 2D or a 3D case

    General form:
        Dike(Size=[], Center=[], Angle=[], Type="..", T=)

        with:

            Size:   dimensions of the dike area
                    2D - [Width; Thickness]
                    3D - [Width; Length; Thickness]

            Center: center of the dike
                    2D - [x; z]
                    3D - [x; y; z]
            
            Angle:  Dip (and strike) angle of dike
                    2D - [Dip]
                    3D - [Strike; Dip]
            
            Type:   Type of dike
                    "SquareDike"    - square dike area   
            
            T:      Temperature of the dike [Celcius]   
            
    All parameters can be specified through keywords as shown above. 
    If keywords are not given, default parameters are employed         
"""
@with_kw struct Dike    # stores info about dike
    # Note: since we utilize the "Parameters.jl" package, we can add more keywords here w/out breaking the rest of the code 
    Size    ::  Vector{Float64} = [1000.; 2000.]          # Size or aspect ratio of dike 3D: (width, length, thickness), 2D: 
    Center  ::  Vector{Float64} = [20e3 ; -10e3]          # Center
    Angle   ::  Vector{Float64} = [0.]                    # Strike/Dip angle of dike
    Type    ::  String          = "SquareDike"            # Type of dike
    T       ::  Float64         = 950.0                   # Temperature of dike
    E       ::  Float64         = 1.5e10                  # Youngs modulus (only required for elastic dikes)
    ν       ::  Float64         = 0.3                     # Poison ratio of host rocks
end

struct DikePoly    # polygon that describes the geometry of the dike (only in 2D)
    x::Float64          # x-coordinates
    z::Float64          # z-coordinates
end

"""
    Structure that has information about the tracers

    General form:
        Tracer(num, coord, T)

        with:

            num:   number of the tracer (integer)

            coord: coordinates of the tracer
                    2D - [x; z]
                    3D - [x; y; z]
            
            T:      Temperature of the tracer [Celcius]                              
"""
struct Tracer
    num::Int64                  # number
    coord::Vector{Float64}      # holds coordinates [2D or 3D]
    T::Float64                  # temperature
end

"""
    This injects a dike in the computational domain in an instantaneous manner,
    while "pushing" the host rocks to the sides. 

    The orientation and the type of the dike are described by the structure     

    General form:
        T, Velocity, VolumeInjected = InjectDike(Tracers, T, Grid, Spacing, dike, nTr_dike )

    with:
        T:          Temperature grid (will be modified)

        Tracers:    StructArray that contains the passive tracers (will be modified)

        Grid:       regular grid on which the temperature is defined
                    2D - (X,Z)
                    3D - (X,Y,Z)

        Spacing:    (constant) spacing of grid
                    2D - (dx,dz)
                    3D - (dx,dy,dz)

        nTr_dike:   Number of new tracers to be injected into the new dike area

"""

function InjectDike(Tracers, T, Grid, Spacing, dike, nTr_dike )

    # Some notes on the algorithm:
    #   For computational reasons, we do not open the dike at once, but in sufficiently small pseudo timesteps
    #   Sufficiently small implies that the motion per "pseudotimestep" cannot be more than 0.5*{dx|dy|dz}
    
    @unpack Size = dike
    H           =    Size[end];                                         # thickness of dike
    d           =    minimum(Spacing)*0.5;                              # maximum distance the dike can open per pseudotimestep 

    nsteps      =   maximum([ceil(H/d), 10]);                           # the number of steps (>=10)

    # Compute velocity required to create space for dike
    Δ           =   H/(nsteps);
    dt          =   1.0
    Velocity    =   HostRockVelocityFromDike(Grid,Δ, dt,dike);          # compute velocity field
   
    # Move hostrock & already existing tracers to the side to create space for new dike
    Tnew        =   zeros(size(T))
    for ipseudotime=1:nsteps
        Tnew    =   AdvectTemperature(T,        Grid,  Velocity,   Spacing,    dt);    
        if length(Tracers)>0
            Tracers =   AdvectTracers(Tracers, T,   Grid,  Velocity,   Spacing,    dt);
        end
        T       =   Tnew;
    end

    # Insert dike in T profile and add new tracers
    T,Tracers   =   AddDike(T, Tracers, Grid,dike, nTr_dike);                 # Add dike to T-field & insert tracers within dike
    
    # Compute volume of newly injected magma
    Area,InjectedVolume =   volume_dike(dike)

    return T, Tracers, InjectedVolume, Velocity

end

#--------------------------------------------------------------------------
"""
    Host rock velocity obtained during opening of dike

    General form:
        Velocity = HostRockVelocityFromDike( Grid, Δ, dt, dike);

        with:

            Grid: coordinates of regular grid @ which we compute velocity
                    2D - [X; Z]
                    3D - [x; y; z]

            dike: structure that holds info about dike
            dt:   time in which the full dike is opened
            
    Note: the velocity is computed in such a manner that a maximum opening 
        increment of Δ = Vmax*dt is obtained after this timestep


"""
function HostRockVelocityFromDike( Grid, Δ, dt, dike::Dike)

    # Prescibe the velocity field to inject the dike with given orientation
    # 
    dim = length(Grid);
    if dim==2
        X, Z        =   Grid[1], Grid[2];
        
        # Rotate and shift coordinate system into 'dike' reference frame
        @unpack Angle,Type = dike
        α           =   Angle[end];
        RotMat      =   [cosd(α) -sind(α); sind(α) cosd(α)];    # 2D rotation matrix
        Xrot,Zrot   =   zeros(size(X)), zeros(size(Z));

        for i=1:size(Z,1)
            for j=1:size(Z,2)
                pt                      =   [X[i,j]; Z[i,j]] - dike.Center;  # original point, shifted to center of dike
                pt_rot                  =   RotMat*pt;

                Xrot[i,j], Zrot[i,j]    =   pt_rot[1],  pt_rot[2];
            end
        end

        Vx_rot, Vz_rot  = zeros(size(X)), zeros(size(Z));
        Vx, Vz          = zeros(size(X)), zeros(size(Z));
            
        if Type=="SquareDike"
            @unpack Size = dike
            W, H    =  Size[1], Size[end];              # Dimensions of square dike
            Vint    =  Δ/dt;                            # open the dike by a maximum amount of Δ in one dt
                
            Vz_rot[(Zrot.<0) .& (abs.(Xrot).<W/2.0)] .= -Vint;
            Vz_rot[(Zrot.>0) .& (abs.(Xrot).<W/2.0)] .=  Vint;

            Vx_rot[abs.(X).<W]          .=   0.0;      # set radial velocity to zero at left boundary
        else
            error("Unknown Dike Type: $Type")
        end


        # "unrotate" vector fields
        RotMat = [cosd(-α) -sind(-α); sind(-α) cosd(-α)];    # 2D rotation matrix (in opposite direction)
        for i=1:size(Z,1)
            for j=1:size(Z,2)
                pt          =   [Vx_rot[i,j]; Vz_rot[i,j]];  # velocities in rotated space
                pt_rot      =   RotMat*pt;

                Vx[i,j]     =   pt_rot[1];
                Vz[i,j]     =   pt_rot[2];
            end
        end

        return (Vx, Vz);

    else
        @unpack Angle = dike;
        α,β         =   Angle[end], Angle[1];
        RotMat_z    =   [cosd(α) -sind(α) 0.0; sind(α) cosd(α)      0.0;  0.0 0.0       1.0]; 
        RotMat_x    =   [1.0        0.0     0.0; 0.0      cosd(β) -sind(β); 0.0 sind(β) cosd(β)]; 
        RotMat      =   RotMat_z*RotMat_x;

        error("3D implementation to be finished")
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
            @unpack Size, Center = dike;   
            H       =  Size[end];
            W       =  Size[1];
            
            pt      = [LazyRow(poly, i).x*W/2.0; LazyRow(poly, i).z*H/2.0];
            pt_rot  = RotMat*pt;    # rotate
            
            LazyRow(poly, i).x = pt_rot[1] + Center[1];   # shift
            LazyRow(poly, i).z = pt_rot[2] + Center[2];
        end
    
    elseif Type=="Elliptical"
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
    @unpack Type,Size = dike;
    if Type=="SquareDike"
        if  dim==2
            if  (abs(pt[1])  < Size[1]/2.0) & (abs(pt[end])< Size[end]/2.0)
                in = true
            end
        elseif dim==3
            if  (abs(pt[1])  < Size[1]/2.0) & (abs(pt[end])< Size[end]/2.0) & (abs(pt[2])< Size[2]/2.0)
                in = true
            end
        end

    elseif Type=="Ellipse"
        
        if dim==2
            eq_ellipse = pt[1]^2.0/(Size[1]/2.0)^2.0 + pt[2]^2.0/(Size[2]/2.0)^2.0; # ellipse
        else
            eq_ellipse = pt[1]^2.0/(Size[1]/2.0)^2.0 + pt[2]^2.0/(Size[2]/2.0)^2.0 + pt[3]^2.0/(Size[3]/2.0)^2.0; # ellipsoid
        end

        if eq_ellipse < 1.0
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
    @unpack Size, Type =dike
    dim = length(Size)
    if Type=="SquareDike"
        if  dim==2
            area    = Size[1]*Size[2];                        #  (in 2D, in m^2)
            volume  = Size[1]*Size[1]*Size[2];           #  (equivalent 3D volume, in m^3)
     
        elseif dim==3
            area    = Size[1]*Size[3]                         #   (cross-sectional area, in m^2)
            volume  = Size[1]*Size[2]*Size[3]            #   (in 3D, in m^3)
        end

    elseif Type=="Ellipse"
        
        if dim==2
            area    = pi*Size[1]*Size[2]                      #   (in 2D, in m^2)
            volume  = 4/3*pi*Size[1]*Size[1]*Size[2]     #   (equivalent 3D volume, in m^3)
        else
            area    = pi*Size[1]*Size[3]                      #   (cross-sectional area, in m^2)
            volume  = 4/3*pi*Size[1]*Size[2]*Size[3]     #   (in 3D, in m^3)
        end
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
    dike_poly   =   CreatDikePolygon(dike);                            # Polygon that describes the dike 
    @unpack Angle,Center,Size,T = dike;
    
    if dim==2
        α           =    Angle[end];
        RotMat      =    [cosd(α) -sind(α); sind(α) cosd(α)]; 
    elseif dim==3
        α           =    Angle[end];
        β           =    Angle[1];
        RotMat_z    =   [cosd(α) -sind(α) 0.0; sind(α) cosd(α)      0.0;  0.0 0.0       1.0]; 
        RotMat_x    =   [1.0        0.0     0.0; 0.0      cosd(β) -sind(β); 0.0 sind(β) cosd(β)]; 
        RotMat      =   RotMat_z*RotMat_x;
    end
  
    # Add dike to temperature field
    if dim==2
        X,Z = Grid[1], Grid[2];
        for ix=1:size(X,1)
            for iz=1:size(X,2)
                pt = [X[ix,iz];Z[ix,iz]] - Center;
                pt_rot  = RotMat*pt;          # rotate and shift
                in = isinside_dike(pt_rot, dike);
                if in
                    Tfield[ix,iz] = T;
                end
            end
        end

    elseif dim==3
        X,Y,Z = Grid[1], Grid[2], Grid[3]
        for ix=1:size(X,1)
            for iy=1:size(X,2)
                for iz=1:size(X,3)
                    pt      = [X[ix,iy,iz];Y[ix,iy,iz];Z[ix,iy,iz]] - Center;
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
    if dim==2
        α      = Angle[end];
        RotMat = [cosd(-α) -sind(-α); sind(-α) cosd(-α)]; 
    elseif dim==3
        α      = Angle[end];
        β      = Angle[1];
        RotMat_z = [cosd(-α) -sind(-α) 0.0; sind(-α) cosd(-α)      0.0;  0.0 0.0       1.0]; 
        RotMat_x = [1.0        0.0     0.0; 0.0      cosd(-β) -sind(-β); 0.0 sind(-β) cosd(-β)]; 
        RotMat   = RotMat_z*RotMat_x;
    end
    for iTr=1:nTr_dike
       
        # 1) Randomly initialize tracers to the approximate dike area
        pt      = rand(dim,1) .- 0.5*ones(dim,1);
        pt      = pt.*Size;
        pt_rot  = RotMat*pt .+ Center[:];          # rotate and shift

        # 2) Make sure that they are inside the dike area   
        in      = isinside_dike(pt, dike);
       
        # 3) Add them to the tracers structure
        if in   # we are inside dike polygon
            
            if      dim==2; pt_new = [pt_rot[1]; pt_rot[2]];
            elseif  dim==3; pt_new = [pt_rot[1]; pt_rot[2]; pt_rot[3]]; end
            
            if length(Tr)==0;   num     =   1;
            else                num     =   Tr.num[end]+1;  end

            new_tracer  =   Tracer(num, pt_new, T);            # Create new tracer
            if length(Tr)==0
                Tr      =   StructArray([new_tracer]);          # Create tracer array
            else
                push!(Tr, new_tracer);                              # Add new point to existing array
            end
        end

    end

    return Tfield, Tr, dike_poly;


end


"""

This contains a number of routines that are related to inserting new dikes to the simulation,
defining a velocity field that "opens" the host rock accordingly and to inserting the dike temperature to the
temperature field

"""
module Dikes

using StructArrays

export Dike, DikePoly, Tracer
export AddDike, HostRockVelocityFromDike, CreatDikePolygon, volume_dike


"""
    Structure that holds the geometrical parameters of the dike, which are slightly different
    depending on whether we consider a 2D or a 3D case

    General form:
        Dike(Size, Center, Angle, Type, T)

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
"""
struct Dike    # stores info about dike 
    Size::Vector{Float64}       # Size or aspect ratio of dike 3D: (width, length, thickness), 2D: 
    Center::Vector{Float64}     # Center
    Angle::Vector{Float64}      # Strike/Dip angle of dike
    Type::String                # Type of dike
    T::Float64                  # temperature
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

#--------------------------------------------------------------------------
"""
    Host rock velocity obtained during opening of dike

    General form:
        Velocity = HostRockVelocityFromDike( Grid, dt, dike);

        with:

            Grid: coordinates of regular grid @ which we compute velocity
                    2D - [X; Z]
                    3D - [x; y; z]

            dike: structure that holds info about dike
            dt:   time in which the full dike is opened
            
    Note: the velocity is scaled in such a manner that the maximum dike thickness (H) is achieved
            after time dt, so Vmax = H*dt

"""
function HostRockVelocityFromDike( Grid, dt, dike);

    # Prescibe the velocity field to inject the dike with given orientation
    # 
    dim = length(Grid);
    if dim==2
        X = Grid[1];
        Z = Grid[2];

        # Rotate and shift coordinate system into 'dike' reference frame
        α      = dike.Angle[end];
        RotMat = [cosd(α) -sind(α); sind(α) cosd(α)];    # 2D rotation matrix
        Xrot   = zeros(size(X));
        Zrot   = zeros(size(Z));
        for i=1:size(Z,1)
            for j=1:size(Z,2)
                pt          =   [X[i,j]; Z[i,j]] - dike.Center;  # original point, shifted to center of dike
                pt_rot      =   RotMat*pt;

                Xrot[i,j]   =   pt_rot[1];
                Zrot[i,j]   =   pt_rot[2];
            end
        end

        Vz_rot    = zeros(size(Z));
        Vx_rot    = zeros(size(X));
        Vz        = zeros(size(Z));
        Vx        = zeros(size(X));
            
        if dike.Type=="SquareDike"
            # Simple square dike
            H    =  dike.Size[end];
            W    =  dike.Size[1];
            

            Vint = H/dt;       # open the dike in one dt
                
            Vz_rot[(Zrot.<0) .& (abs.(Xrot).<W/2.0)] .= -Vint;
            Vz_rot[(Zrot.>0) .& (abs.(Xrot).<W/2.0)] .=  Vint;

            Vx_rot[abs.(X).<W]          .=   0.0;      # set radial velocity to zero at left boundary
        else
            error("Unknown Dike type field")
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

        α      = dike.Angle[end];
        β      = dike.Angle[1];
        RotMat_z = [cosd(α) -sind(α) 0.0; sind(α) cosd(α)      0.0;  0.0 0.0       1.0]; 
        RotMat_x = [1.0        0.0     0.0; 0.0      cosd(β) -sind(β); 0.0 sind(β) cosd(β)]; 
        RotMat   = RotMat_z*RotMat_x;


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
    
    if dike.Type=="SquareDike"

        # unit dike
        poly = StructArray([DikePoly(-1.0, -1.0), DikePoly( 1.0, -1.0),
                            DikePoly( 1.0,  1.0), DikePoly(-1.0,  1.0)]);
        
        push!(poly,poly[1]); # close polygon

        # scale & rotate dike
        α      = dike.Angle[end];
        RotMat = [cosd(-α) -sind(-α); sind(-α) cosd(-α)]; 
        for i=1:length(poly)
            H       =  dike.Size[end];
            W       =  dike.Size[1];
            
            pt      = [LazyRow(poly, i).x*W/2.0; LazyRow(poly, i).z*H/2.0];
            pt_rot  = RotMat*pt;    # rotate
            
            LazyRow(poly, i).x = pt_rot[1] + dike.Center[1];   # shift
            LazyRow(poly, i).z = pt_rot[2] + dike.Center[2];
        end
    
    elseif dike.Type=="Elliptical"
        error("To be added")
    
    else
        error("Unknown dike type $dike.Type")
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
    if dike.Type=="SquareDike"
        if  dim==2
            if  (abs(pt[1])  < dike.Size[1]/2.0) & (abs(pt[end])< dike.Size[end]/2.0)
                # area = dike.Size[1]*dike.Size[2]  (in 2D, in m^2)
            
                in = true
            end
        elseif dim==3
            if  (abs(pt[1])  < dike.Size[1]/2.0) & (abs(pt[end])< dike.Size[end]/2.0) & (abs(pt[2])< dike.Size[2]/2.0)
                # volume = dike.Size[1]*dike.Size[2]*dike.Size[3]  (in 3D, in m^3)
                in = true
            end
        end

    elseif dike.Type=="Ellipse"
        
        if dim==2
            # area = pi*dike.Size[1]*dike.Size[2]  (in 2D, in m^2)
            eq_ellipse = pt[1]^2.0/(dike.Size[1]/2.0)^2.0 + pt[2]^2.0/(dike.Size[2]/2.0)^2.0; # ellipse
        else
            # volume = 4/3*pi*dike.Size[1]*dike.Size[2]*dike.Size[3]  (in 3D, in m^3)
            eq_ellipse = pt[1]^2.0/(dike.Size[1]/2.0)^2.0 + pt[2]^2.0/(dike.Size[2]/2.0)^2.0 + pt[3]^2.0/(dike.Size[3]/2.0)^2.0; # ellipsoid
        end

        if eq_ellipse < 1.0
            in = true;
        end

    else
        error("Unknown dike type $dike.Type")
    end

    return in 
end


"""
    V = volume_dike(dike::Dike)

    Returns the area (2D) or volume (3D) of the injected dike

"""
function volume_dike(dike::Dike)
    # important: this is a "unit" dike, which has the center at [0,0,0] and width given by dike.Size
    dim =   length(pt)
    if dike.Type=="SquareDike"
        if  dim==2
            area = dike.Size[1]*dike.Size[2];               #  (in 2D, in m^2)
            return area
        elseif dim==3
          volume = dike.Size[1]*dike.Size[2]*dike.Size[3]   # (in 3D, in m^3)
          return volume;
        end

    elseif dike.Type=="Ellipse"
        
        if dim==2
            area = pi*dike.Size[1]*dike.Size[2]             #  (in 2D, in m^2)
            return area
        else
            volume = 4/3*pi*dike.Size[1]*dike.Size[2]*dike.Size[3]  # (in 3D, in m^3)
            return volume;
        end
    else
        error("Unknown dike type $dike.Type")
    end

end

#--------------------------------------------------------------------------
"""
    T, Tracers, dike_poly = AddDike(T,Tracers, Grid, dike,nTr_dike)
    
Adds a dike, described by the dike polygon dike_poly, to the temperature field T (defined at points Grid).
Also adds nTr_dike new tracers randomly distributed within the dike, to the 
tracers array Tracers.

"""
function AddDike(T,Tr, Grid,dike, nTr_dike)

    dim         =   length(Grid);
    dike_poly   =   CreatDikePolygon(dike);                            # Polygon that describes the dike 
    
    if dim==2
        α      = dike.Angle[end];
        RotMat = [cosd(α) -sind(α); sind(α) cosd(α)]; 
    elseif dim==3
        α      = dike.Angle[end];
        β      = dike.Angle[1];
        RotMat_z = [cosd(α) -sind(α) 0.0; sind(α) cosd(α)      0.0;  0.0 0.0       1.0]; 
        RotMat_x = [1.0        0.0     0.0; 0.0      cosd(β) -sind(β); 0.0 sind(β) cosd(β)]; 
        RotMat   = RotMat_z*RotMat_x;
    end
  
    # Add dike to temperature field
    if dim==2
        X = Grid[1];
        Z = Grid[2];
        for ix=1:size(X,1)
            for iz=1:size(X,2)
                pt = [X[ix,iz];Z[ix,iz]] - dike.Center;
                pt_rot  = RotMat*pt;          # rotate and shift
                in = isinside_dike(pt_rot, dike);
                if in
                    T[ix,iz] = dike.T;
                end
            end
        end

    elseif dim==3
        X = Grid[1];
        Y = Grid[2];
        Z = Grid[3];
        for ix=1:size(X,1)
            for iy=1:size(X,2)
                for iz=1:size(X,3)
                    pt      = [X[ix,iy,iz];Y[ix,iy,iz];Z[ix,iy,iz]] - dike.Center;
                    pt_rot  = RotMat*pt;          # rotate and shift
                    in      = isinside_dike(pt_rot, dike);
                    if in
                        T[ix,iy,iz] = dike.T;
                    end
                end
            end
        end

    end

    # Add new tracers to the dike area
    if dim==2
        α      = dike.Angle[end];
        RotMat = [cosd(-α) -sind(-α); sind(-α) cosd(-α)]; 
    elseif dim==3
        α      = dike.Angle[end];
        β      = dike.Angle[1];
        RotMat_z = [cosd(-α) -sind(-α) 0.0; sind(-α) cosd(-α)      0.0;  0.0 0.0       1.0]; 
        RotMat_x = [1.0        0.0     0.0; 0.0      cosd(-β) -sind(-β); 0.0 sind(-β) cosd(-β)]; 
        RotMat   = RotMat_z*RotMat_x;
    end
    for iTr=1:nTr_dike
       
        # 1) Randomly initialize tracers to the approximate dike area
        pt      = rand(dim,1) .- 0.5*ones(dim,1);
        pt      = pt.*dike.Size;
        pt_rot  = RotMat*pt .+ dike.Center[:];          # rotate and shift

        # 2) Make sure that they are inside the dike area   
        in  = isinside_dike(pt, dike);
       
        # 3) Add them to the tracers structure
        if in   # we are inside dike polygon
            if length(Tr)==0
                error("You need to initialize the tracer array outside this routine!")
            else
                num         =   Tr.num[end]+1;  
                if dim==2
                    new_tracer  =   Tracer(num, [pt_rot[1]; pt_rot[2]], dike.T);
                else
                    new_tracer  =   Tracer(num, [pt_rot[1]; pt_rot[2]; pt_rot[3]], dike.T);
                end

                push!(Tr, new_tracer);                  # add new point to list
            end
        end

    end

    return T, Tr, dike_poly;


end




end
"""

This contains a number of routines that are related to inserting new dikes to the simulation,
defining a velocity field that "opens" the host rock accordingly and to inserting the dike temperature to the
temperature field

"""
module Dikes

using StructArrays
using Luxor                                     # for inpolygon

export Dike, DikePoly, Tracer
export AddDike, HostRockVelocityFromDike, CreatDikePolygon

struct Dike    # stores info about dike 
    W::Float64          # Width of dike/sill
    H::Float64          # Thickness
    T::Float64          # temperature
    x0::Float64         # x-coordinate of center of dike
    z0::Float64         # z-coordinate of center of dike
    α::Float64          # Orientation of dike
    Type::String        # Type of dike
end

struct DikePoly    # polygon that describes the geometry of the dike (in 2D)
    x::Float64          # x-coordinates
    z::Float64          # z-coordinates
end

# Create structure that will hold the info about the passive tracers
struct Tracer
    num::Int64                  # number
    x::Float64                  # x-coordinates
    z::Float64                  # z-coordinates
    #coord::Vector{Float64}     # holds coordinates [2D or 3D]
    T::Float64                  # temperature
end

#--------------------------------------------------------------------------
function HostRockVelocityFromDike( Grid, dt, dike);

    # Prescibe the velocity field to inject the dike with given orientation
    # 
    dim = length(Grid);
    if dim==2
        R = Grid[1];
        Z = Grid[2];

        # Rotate and shift coordinate system into 'dike' reference frame
        RotMat = [cosd(dike.α) -sind(dike.α); sind(dike.α) cosd(dike.α)];    # 2D rotation matrix
        Rrot    = zeros(size(R));
        Zrot    = zeros(size(Z));
        for i=1:size(Z,1)
            for j=1:size(Z,2)
                pt          =   [R[i,j]; Z[i,j]] - [dike.x0; dike.z0];  # original point, shifted to center of dike
                pt_rot      =   RotMat*pt;

                Rrot[i,j]   =   pt_rot[1];
                Zrot[i,j]   =   pt_rot[2];
            end
        end

        Vz_rot    = zeros(size(R));
        Vr_rot    = zeros(size(Z));
        Vz        = zeros(size(R));
        Vr        = zeros(size(Z));
            
        if dike.Type=="SquareDike"
            # Simple square dike
            Vint = dike.H/dt;       # open the dike in one dt
                
            Vz_rot[(Zrot.<0) .& (abs.(Rrot).<dike.W/2.0)] .= -Vint;
            Vz_rot[(Zrot.>0) .& (abs.(Rrot).<dike.W/2.0)] .=  Vint;

            Vr_rot[abs.(R).<dike.W]    .=   0.0;      # set radial velocity to zero at left boundary
            Vr_rot                     .=   Vr*10*0;
        else
            error("Unknown Dike type field")
        end


        # "unrotate" vector fields
        RotMat = [cosd(-dike.α) -sind(-dike.α); sind(-dike.α) cosd(-dike.α)];    # 2D rotation matrix
        for i=1:size(Z,1)
            for j=1:size(Z,2)
                pt          =   [Vr_rot[i,j]; Vz_rot[i,j]];  # velocities in rotated space
                pt_rot      =   RotMat*pt;

                Vr[i,j]   =   pt_rot[1];
                Vz[i,j]   =   pt_rot[2];
            end
        end

        return (Vr, Vz);
    else
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
        RotMat = [cosd(-dike.α) -sind(-dike.α); sind(-dike.α) cosd(-dike.α)];    # 2D rotation matrix
        for i=1:length(poly)
            pt      = [LazyRow(poly, i).x*dike.W/2.0; LazyRow(poly, i).z*dike.H/2.0];
            pt_rot  = RotMat*pt;    # rotate
            
            LazyRow(poly, i).x = pt_rot[1] + dike.x0;   # shift
            LazyRow(poly, i).z = pt_rot[2] + dike.z0;
        end

    # to be added:
    #   1) elliptical dike
    #   2) elastic dike (aka 2D analytical elastic solution)    
        
    end

    return poly 
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
        polygon = Point.(dike_poly.x,dike_poly.z);                  # generate polygon (to use with inside)
    elseif dim==3
        polygon = Point.(dike_poly.x,dike_poly.y,dike_poly.z);      # generate polygon (to use with inside)
    end
    
  
    # Add dike to temperature field
    if dim==2
        X = Grid[1];
        Z = Grid[2];
        

        for ix=1:size(X,1)
            for iz=1:size(X,2)
                in = isinside(Point.(X[ix,iz],Z[ix,iz]),                                polygon; allowonedge=true);
                if in
                    T[ix,iz] = dike.T;
                end
            end
        end

    elseif dim==3
        X = Grid[1];
        T = Grid[2];
        Z = Grid[3];
        for ix=1:size(X,1)
            for iy=1:size(X,2)
                for iz=1:size(X,3)
                    in = isinside(Point.(X[ix,iy,iz],Y[ix,iy,iz],Z[ix,iy,iz]),    polygon; allowonedge=true);
                    if in
                        T[ix,iy,iz] = dike.T;
                    end
                end
            end
        end

    end

    # Add new tracers to the dike area
    if dim==2
        RotMat = [cosd(-dike.α) -sind(-dike.α); sind(-dike.α) cosd(-dike.α)];    # 2D rotation matrix
    elseif dim==3
        error("Need to add 3D rotation matrix")
    end
    for iTr=1:nTr_dike
       
        # 1) Randomly initialize tracers to the approximate dike area
        if dim==2
            pt      = (rand(2,1) .- 0.5).*[dike.W; dike.H];
            pt_rot  = RotMat*pt + [dike.x0; dike.z0];                   # rotate and shift
        elseif dim==3
            pt      = (rand(3,1) .- 0.5).*[dike.W; dike.L; dike.H];
            pt_rot  = RotMat*pt + [dike.x0; dike.y0; dike.z0];          # rotate and shift
        end

        # 2) Make sure that they are inside the dike    
        if dim==2
            in      = isinside(Point.(pt_rot[1], pt_rot[2]),            polygon; allowonedge=true);
        else
            in      = isinside(Point.(pt_rot[1], pt_rot[2], pt_rot[3]), polygon; allowonedge=true);
        end
       
        # 3) Add them to the tracers structure
        if in   # we are inside dike polygon
            if length(Tr)==0
                error("You need to initialize the tracer array outside this routine!")
            else
                num         =   Tr.num[end]+1;  
                new_tracer  =   Tracer(num,pt_rot[1],pt_rot[2],dike.T);
                push!(Tr, new_tracer);                  # add new point to list
            end
        end

    end

    return T, Tr, dike_poly;


end


end
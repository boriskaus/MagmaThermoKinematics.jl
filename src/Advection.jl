"""
This contains routines related to advection of temperature and tracers

"""

"""
    Performs a interpolation, either in 2D or 3D, using either linear of cubic interpolation


    General form:
        Interpolate!(Data_interp, Grid, Data_grid, Points_irregular,  InterpolationMethod="Linear")

        with:

            Grid:               Tuple with 1D coordinate vectors that describe the grid
                    2D - (x,z)
                    3D - (x,y,z)

            Spacing:            (constant) spacing of the grid in each direction
                    2D - (dx,dz)
                    3D - (dx,dy,dz)
            
            DataGrid:           Data that is defined on the grid. Can have 1 field or 2 (2D), respectively 3 (3D) fields 
    
            Points_irregular:   Tuple with 2D or 3D arrays with coordinates of irregular points on which we want to interpolate the data
                    2D - (X,Z)
                    3D - (x,y,z)

            InterpolationMethod:   The interpolation method  
                    "Linear"    -   Linear interpolation (default)
                    "Quadratic" -   Quadratic interpolation
                    "Cubic"     -   Cubix interpolation 
    
            Data_interp:   interpolated data field(s) on the irregular points. Same number of fields as Data_grid                          

            Note: we use the Julia package Interpolations.jl to perform the actual interpolation
"""
function Interpolate!(Data_interp, Grid,  Data_grid, Points_irregular, InterpolationMethod="Linear");
    
    dim::Int8           =   length(Grid);           # number of dimensions
    nField ::Int8       =   length(Data_grid);      # number of fields
    iField::Int64       =   0;
    

    CorrectBounds!(Points_irregular, Grid);

    for iField=1:nField
        
        # Select the interpolation method & scale
        if      InterpolationMethod=="Linear"
            #interp      =   LinearInterpolation(Grid, Data_grid[iField],        extrapolation_bc = Throw());    
            itp         =   interpolate(Data_grid[iField], BSpline(Linear()));
        elseif  InterpolationMethod=="Cubic"
            #interp      =   CubicSplineInterpolation(Grid, Data_grid[iField],   extrapolation_bc = Throw());    
            itp         =   interpolate(Data_grid[iField], BSpline(Cubic(Line(OnCell()))));
        elseif  InterpolationMethod=="Quadratic"
            itp         =   interpolate(Data_grid[iField], BSpline(Quadratic(Line(OnCell()))));
        else
            error("Unknown interpolation method $InterpolationMethod")
        end
        if dim==2
            interp  =   scale(itp,Grid[1],Grid[2]);
        else
            interp  =   scale(itp,Grid[1],Grid[2],Grid[3]);
        end

        # do interpolation for all points
        if dim==2
            evaluate_interp_2D(Data_interp[iField], interp,Points_irregular);
        elseif dim==3
            evaluate_interp_3D(Data_interp[iField], interp,Points_irregular);
        end
    
    end

end

# define functions to perform interpolation with as few allocations as possible
function evaluate_interp_2D(s, itp, Points_irregular)
Threads.@threads    for i=firstindex(Points_irregular[1]):lastindex(Points_irregular[1])
                        s[i]    = itp(Points_irregular[1][i],Points_irregular[2][i]);
                    end
end

function evaluate_interp_3D(s, itp, Points_irregular)
Threads.@threads    for i=firstindex(Points_irregular[1]):lastindex(Points_irregular[1])
                        s[i]    = itp(Points_irregular[1][i],Points_irregular[2][i],Points_irregular[3][i]);
                    end
end


"""
    AdvPoints =   AdvectPoints(AdvPoints0, Grid,Velocity,dt, Method="RK2", InterpolationMethod="Linear");
    
Advects irregular points described by the (2D or 3D tuple) AdvPoints0, though a fixed Eulerian
grid (Grid), with constant spacing (Spacing) on which the velocity components (Velocity) are defined.
Advection is done for the time dt, and can use different methods

"""
function AdvectPoints(AdvPoints0, Grid,Velocity,dt, Method="RK2", InterpolationMethod="Linear", VelocityMethod="Interpolation");
    VelocityMethod == "Interpolation" || error("Unknown VelocityMethod: $VelocityMethod; only \"Interpolation\" is supported")
    dim         = length(AdvPoints0);           # number of dimensions
    AdvPoints   = map(x->x.*0, AdvPoints0) ;    # initialize to 0
   
    if dim==2
        Velocity_int    = (zeros(size(AdvPoints0[1])), zeros(size(AdvPoints0[2])));
    elseif dim==3
        Velocity_int    = (zeros(size(AdvPoints0[1])), zeros(size(AdvPoints0[2])),zeros(size(AdvPoints0[3])));
    end

    # Different advection schemes can be used
    if Method=="Euler"
        Interpolate!(Velocity_int, Grid, Velocity, AdvPoints0, InterpolationMethod);

        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end
        CorrectBounds!( AdvPoints , Grid);
        
    elseif Method=="RK2"

        Interpolate!(Velocity_int, Grid, Velocity, AdvPoints0, InterpolationMethod);
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k2

    elseif Method=="RK4"
        
        Interpolate!(Velocity_int, Grid, Velocity, AdvPoints0, InterpolationMethod);
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k2
        
        # Interpolate velocity values on deformed grid
        Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k3

        # Interpolate velocity values on deformed grid
        Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end             
        CorrectBounds!( AdvPoints , Grid);                               # step k4
        
    else
        error("Unknown advection method: $Method")
    end

    return AdvPoints;
end

"""
    CorrectBounds!(Points, Grid);
    
Ensures that the coordinates of Points stay within the bounds
of the regular grid Grid, which is a tuple of 2 or 3 field (for 2D/3D)

 """
function CorrectBounds!(Points, Grid);

    #Points_new  = map(x->x.*0, Points) ; # initialize to 0
    for i=1:length(Grid);
        Points[i][Points[i].<minimum(Grid[i])]      .=      minimum(Grid[i]); 
        Points[i][Points[i].>maximum(Grid[i])]      .=      maximum(Grid[i]); 
    end
end



"""
        Tnew = AdvectTemperature(T, Grid, Velocity, Spacing, dt, Method="RK2",DataInterpolationMethod="Quadratic")

    Advects temperature for one timestep dt, using a semi-lagrangian advection scheme 

        Method: can be "Euler","RK2" or "RK4", for 1th, 2nd or 4th order explicit advection scheme, respectively. 
"""
function AdvectTemperature( T::Array,Grid, PointsAdv0, Velocity, dt, Method="RK2", DataInterpolationMethod="Quadratic", VelocityMethod="Interpolation");
    
    dim  = length(Grid);
    Tnew = tuple(T);
    # 1) Use semi-lagrangian advection to advect temperature
    # Advect regular grid backwards in time
    PointsAdv = AdvectPoints(PointsAdv0, Grid,Velocity,-dt,Method, "Linear", VelocityMethod);

    # 2) Interpolate temperature on deformed points
    Interpolate!( Tnew, Grid, tuple(T), PointsAdv, DataInterpolationMethod);    
    
    return Tnew[1];
end




"""
    AdvectTemperature!(Tnew, T, Grid, Velocity, dt)

Semi-Lagrangian advection of `T` by the displacement field `Velocity` (one array
per direction, in meters, defined on the grid nodes) over pseudo-time `dt`, into
`Tnew`. `Grid` is the tuple of 1D coordinate ranges. The spacing is constant, so
every departure point is index arithmetic; the RK2 velocity sample and the
temperature read are multilinear. This is the RK2/linear scheme of
[`AdvectTemperature`](@ref), threaded and without allocations.

`Tnew` must not alias `T`: the departure point of one node generally reads
values other nodes still need.
"""
function AdvectTemperature!(Tnew::AbstractArray{<:Any,2}, T, Grid, Velocity, dt)
    Tnew === T && error("AdvectTemperature!: Tnew must be a separate array from T")
    Nx, Nz = size(T)
    dx, dz = step(Grid[1]), step(Grid[2])
    u,  w  = Velocity
    Threads.@threads for j in 1:Nz
        for i in 1:Nx
            # RK2, backward in pseudo-time: half a step on the node velocity,
            # then a full step on the velocity sampled where that landed.
            p = (clamp(i - 0.5*dt*u[i,j]/dx, 1.0, Nx),
                 clamp(j - 0.5*dt*w[i,j]/dz, 1.0, Nz))
            q = (clamp(i - dt*_lerp(u, p)/dx, 1.0, Nx),
                 clamp(j - dt*_lerp(w, p)/dz, 1.0, Nz))
            Tnew[i,j] = _lerp(T, q)
        end
    end
    return Tnew
end

function AdvectTemperature!(Tnew::AbstractArray{<:Any,3}, T, Grid, Velocity, dt)
    Tnew === T && error("AdvectTemperature!: Tnew must be a separate array from T")
    Nx, Ny, Nz = size(T)
    dx, dy, dz = step(Grid[1]), step(Grid[2]), step(Grid[3])
    u,  v,  w  = Velocity
    Threads.@threads for k in 1:Nz
        for j in 1:Ny, i in 1:Nx
            p = (clamp(i - 0.5*dt*u[i,j,k]/dx, 1.0, Nx),
                 clamp(j - 0.5*dt*v[i,j,k]/dy, 1.0, Ny),
                 clamp(k - 0.5*dt*w[i,j,k]/dz, 1.0, Nz))
            q = (clamp(i - dt*_lerp(u, p)/dx, 1.0, Nx),
                 clamp(j - dt*_lerp(v, p)/dy, 1.0, Ny),
                 clamp(k - dt*_lerp(w, p)/dz, 1.0, Nz))
            Tnew[i,j,k] = _lerp(T, q)
        end
    end
    return Tnew
end

"Multilinear read of `A` at the fractional *index* position `p`, which must lie within `[1, size(A,d)]` in every direction."
@inline function _lerp(A::AbstractArray{<:Any,2}, p)
    i  = clamp(floor(Int, p[1]), 1, size(A,1)-1); fx = p[1]-i
    j  = clamp(floor(Int, p[2]), 1, size(A,2)-1); fz = p[2]-j
    a  = A[i,j]  *(1-fx) + A[i+1,j]  *fx
    b  = A[i,j+1]*(1-fx) + A[i+1,j+1]*fx
    return a*(1-fz) + b*fz
end

@inline function _lerp(A::AbstractArray{<:Any,3}, p)
    i  = clamp(floor(Int, p[1]), 1, size(A,1)-1); fx = p[1]-i
    j  = clamp(floor(Int, p[2]), 1, size(A,2)-1); fy = p[2]-j
    k  = clamp(floor(Int, p[3]), 1, size(A,3)-1); fz = p[3]-k
    a  = (A[i,j,  k]  *(1-fx) + A[i+1,j,  k]  *fx)*(1-fy) + (A[i,j+1,k]  *(1-fx) + A[i+1,j+1,k]  *fx)*fy
    b  = (A[i,j,  k+1]*(1-fx) + A[i+1,j,  k+1]*fx)*(1-fy) + (A[i,j+1,k+1]*(1-fx) + A[i+1,j+1,k+1]*fx)*fy
    return a*(1-fz) + b*fz
end

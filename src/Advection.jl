"""
This contains routines related to advection of temperature and tracers

"""

"""
    Performs a interpolation, either in 2D or 3D, using either linear of cubic interpolation


    General form:
        Data_interp = Interpolate( Grid, Spacing, Data_grid, Points_irregular,  InterpolationMethod="Linear")

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
                    "Cubic"     -   Cubix interpolation 
    
            Data_interp:   interpolated data field(s) on the irregular points. Same number of fields as Data_grid                          

            Note: we use the Julia package Interpolations.jl to perform the actual interpolation
"""
function Interpolate( Grid, Spacing, Data_grid, Points_irregular, InterpolationMethod="Linear");

    dim                 =   length(Grid);           # number of dimensions
    nField              =   length(Data_grid);      # number of fields
     
    X_irr               =   Points_irregular[1];    # coordinates of irregular point
    
    if dim==2
        Z_irr           =   Points_irregular[2];
    else
        Y_irr           =   Points_irregular[2];
        Z_irr           =   Points_irregular[3];
    end
    
    if nField==1
        Data_interp    = tuple(zeros(size(X_irr))); # initialize to 0
    elseif nField==2
        Data_interp    = (zeros(size(X_irr)), zeros(size(X_irr)));
    elseif nField==3
        Data_interp    = (zeros(size(X_irr)), zeros(size(X_irr)),zeros(size(X_irr)));
    else
        error("Unknown number of fields ($nField)")
    end

    for iField=1:nField

        # Select the interpolation method
        if      InterpolationMethod=="Linear"
            interp          =   LinearInterpolation(Grid, Data_grid[iField],        extrapolation_bc = Flat());    
        elseif  InterpolationMethod=="Cubic"
            interp          =   CubicSplineInterpolation(Grid, Data_grid[iField],   extrapolation_bc = Flat());    
        else
            error("Unknown interpolation method $InterpolationMethod")
        end

        for i=firstindex(X_irr):lastindex(X_irr)
            if dim==2
                Data_interp[iField][i] = interp(X_irr[i], Z_irr[i]); 
            else
                Data_interp[iField][i] = interp(X_irr[i], Y_irr[i], Z_irr[i]); 
            end
        end        
    end

    return Data_interp

end


"""
    AdvPoints =   AdvectPoints(AdvPoints0, Grid,Velocity,Spacing,dt, Method="RK4", InterpolationMethod="Linear");
    
Advects irregular points described by the (2D or 3D tuple) AdvPoints0, though a fixed Eulerian
grid (Grid), with constant spacing (Spacing) on which the velocity components (Velocity) are defined.
Advection is done for the time dt, and can use different methods

"""
function AdvectPoints(AdvPoints0, Grid,Velocity,Spacing,dt, Method="RK2", InterpolationMethod="Linear");

    dim         = length(AdvPoints0);           # number of dimensions
    AdvPoints   = map(x->x.*0, AdvPoints0) ;    # initialize to 0

    # Different advection schemes can be used
    if Method=="Euler"

        Velocity_int = Interpolate( Grid, Spacing, Velocity, AdvPoints0, InterpolationMethod);    
        
        
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end
        AdvPoints  =   CorrectBounds( AdvPoints , Grid);
        
    elseif Method=="RK2"
        Velocity_int = Interpolate( Grid, Spacing, Velocity, AdvPoints0, InterpolationMethod);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate( Grid, Spacing, Velocity, AdvPoints, InterpolationMethod);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end    
        AdvPoints  =   CorrectBounds( AdvPoints , Grid);                               # step k2

    elseif Method=="RK4"

        Velocity_int = Interpolate( Grid, Spacing, Velocity, AdvPoints0, InterpolationMethod);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate( Grid, Spacing, Velocity, AdvPoints,  InterpolationMethod);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds( AdvPoints , Grid);                               # step k2
        
        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate( Grid, Spacing, Velocity, AdvPoints,  InterpolationMethod);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds( AdvPoints , Grid);                               # step k3

        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate( Grid, Spacing, Velocity, AdvPoints,  InterpolationMethod);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end             
        AdvPoints  =   CorrectBounds( AdvPoints , Grid);                               # step k4
        
    end

    return AdvPoints;
end

"""
    Points_new = CorrectBounds(Points, Grid);
    
Ensures that the coordinates of Points stay within the bounds
of the regular grid Grid, which is a tuple of 2 or 3 field (for 2D/3D)

 """
function CorrectBounds(Points, Grid);

    Points_new  = map(x->x.*0, Points) ; # initialize to 0
    for i=1:length(Grid);
        minB    = minimum(Grid[i]);
        maxB    = maximum(Grid[i]);
    
        X               =       Points[i];
        X[X.<minB]      .=      minB; 
        X[X.>maxB]      .=      maxB; 
        Points_new[i]   .=       X;
    end
    
    return Points_new;
end


"""
        Tnew = AdvectTemperature(T, Grid, Velocity, Spacing, dt, Method="RK4",DataInterpolationMethod="Cubic")

    Advects temperature for one timestep dt, using a semi-lagrangian advection scheme 

        Method: can be "Euler","RK2" or "RK4", for 1th, 2nd or 4th order explicit advection scheme, respectively. 
"""
function AdvectTemperature(T::Array,Grid, Velocity, Spacing, dt, Method="RK2", DataInterpolationMethod="Cubic");
    
    dim = length(Grid);

    # 1) Use semi-lagrangian advection to advect temperature
    # Advect regular grid backwards in time

    # Create 2D or 3D grid to be advected backwards    
    if dim==2
        coords      =   collect(Iterators.product(Grid[1],Grid[2]))                             # generate coordinates from 1D coordinate vectors   
        X,Z         =   (x->x[1]).(coords), (x->x[2]).(coords);                      
        PointsAdv   =   (X,Z);
    else
        coords      =   collect(Iterators.product(Grid[1],Grid[2],Grid[3]))                     # generate coordinates from 1D coordinate vectors   
        X,Y,Z       =   (x->x[1]).(coords),(x->x[2]).(coords),(x->x[3]).(coords);               # transfer coords to 3D arrays
        PointsAdv   =   (X,Y,Z);
    end

    PointsAdv   =   AdvectPoints(PointsAdv, Grid,Velocity,Spacing,-dt,Method, "Linear");
 
    # 2) Interpolate temperature on deformed points
    Tnew        =   Interpolate(Grid, Spacing, tuple(T), PointsAdv, DataInterpolationMethod);    

    return Tnew[1];
end


"""
        Tnew = AdvectTracers(Tracers, Grid, Velocity, Spacing, dt, Method="RK4");

        Advects [Tracers] for one timestep (dt) using the [Velocity] defined on the points [Grid]
        that have constant [Spacing].

        Method: can be "Euler","RK2" or "RK4", for 1th, 2nd or 4th order explicit advection scheme, respectively. 
"""
function AdvectTracers(Tracers, Grid, Velocity, Spacing, dt, Method="RK2");
    # Advect tracers forward in time & interpolate T on them 
    
    dim = length(Grid);

    coord = Tracers.coord; coord = hcat(coord...)';       # extract array with coordinates of tracers
    
    x   = coord[:,1];
    z   = coord[:,end];
    if dim==2
        Points_irregular    =   (x,z);
    else
        y                   =   coord[:,2];
        Points_irregular    =   (x,y,z);
    end
    
    # Correct coordinates (to stay withoin bounds of models)
    Points_irregular    =   CorrectBounds(Points_irregular, Grid);

    # Advect
    Points_new          =   AdvectPoints(Points_irregular,  Grid,Velocity,Spacing, dt,Method,  "Linear");     # Advect tracers
    
    for iT = 1:length(Tracers)
        if dim==2
            LazyRow(Tracers, iT).coord = [Points_new[1][iT]; Points_new[2][iT]];
       
        elseif dim==3
            LazyRow(Tracers, iT).coord = [Points_new[1][iT]; Points_new[2][iT]; Points_new[3][iT]];
        end
    end

    return Tracers
end

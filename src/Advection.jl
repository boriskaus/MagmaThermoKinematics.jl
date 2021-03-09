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
function AdvectPoints(AdvPoints0, Grid,Velocity,dt, Method="RK2", InterpolationMethod="Linear", VelocityMethod="Interpolation", DikeStruct=[], Δ=1.0);
    dim         = length(AdvPoints0);           # number of dimensions
    AdvPoints   = map(x->x.*0, AdvPoints0) ;    # initialize to 0
   
    if dim==2
        Velocity_int    = (zeros(size(AdvPoints0[1])), zeros(size(AdvPoints0[2])));
    elseif dim==3
        Velocity_int    = (zeros(size(AdvPoints0[1])), zeros(size(AdvPoints0[2])),zeros(size(AdvPoints0[3])));
    end

    # Different advection schemes can be used
    if Method=="Euler"
        if VelocityMethod=="Interpolation"
            Interpolate!(Velocity_int, Grid, Velocity, AdvPoints0, InterpolationMethod);    
        elseif VelocityMethod=="FromDike"
            Velocity_int    =   HostRockVelocityFromDike(Grid, AdvPoints0, Δ, abs(dt), DikeStruct);          # compute velocity field
        end

        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end
        CorrectBounds!( AdvPoints , Grid);
        
    elseif Method=="RK2"

        if VelocityMethod=="Interpolation"
            Interpolate!(Velocity_int, Grid, Velocity, AdvPoints0, InterpolationMethod);    
        elseif VelocityMethod=="FromDike"
            Velocity_int    =   HostRockVelocityFromDike(Grid, AdvPoints0, Δ, abs(dt), DikeStruct);          # compute velocity field
        end  
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        if VelocityMethod=="Interpolation"
            Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);    
        elseif VelocityMethod=="FromDike"
            Velocity_int    =   HostRockVelocityFromDike(Grid, AdvPoints, Δ, abs(dt), DikeStruct);          # compute velocity field
        end  
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k2

    elseif Method=="RK4"
        
        if VelocityMethod=="Interpolation"
            Interpolate!(Velocity_int, Grid, Velocity, AdvPoints0, InterpolationMethod);    
        elseif VelocityMethod=="FromDike"
            Velocity_int    =   HostRockVelocityFromDike(Grid, AdvPoints0, Δ, abs(dt), DikeStruct);          # compute velocity field
        end   
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        if VelocityMethod=="Interpolation"
            Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);    
        elseif VelocityMethod=="FromDike"
            Velocity_int    =   HostRockVelocityFromDike(Grid, AdvPoints, Δ, abs(dt), DikeStruct);          # compute velocity field
        end   
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k2
        
        # Interpolate velocity values on deformed grid
        if VelocityMethod=="Interpolation"
            Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);    
        elseif VelocityMethod=="FromDike"
            Velocity_int    =   HostRockVelocityFromDike(Grid, AdvPoints, Δ, abs(dt), DikeStruct);          # compute velocity field
        end  
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        CorrectBounds!( AdvPoints , Grid);                               # step k3

        # Interpolate velocity values on deformed grid
        if VelocityMethod=="Interpolation"
            Interpolate!(Velocity_int, Grid, Velocity, AdvPoints, InterpolationMethod);    
        elseif VelocityMethod=="FromDike"
            Velocity_int    =   HostRockVelocityFromDike(Grid, AdvPoints, Δ, abs(dt), DikeStruct);          # compute velocity field
        end    
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
function AdvectTemperature( T::Array,Grid, PointsAdv0, Velocity, dt, Method="RK2", DataInterpolationMethod="Quadratic", VelocityMethod="Interpolation", DikeStruct=[], Δ=1 );
    
    dim  = length(Grid);
    Tnew = tuple(T);
    # 1) Use semi-lagrangian advection to advect temperature
    # Advect regular grid backwards in time
    PointsAdv = AdvectPoints(PointsAdv0, Grid,Velocity,-dt,Method, "Linear", VelocityMethod, DikeStruct, Δ);

    # 2) Interpolate temperature on deformed points
    Interpolate!( Tnew, Grid, tuple(T), PointsAdv, DataInterpolationMethod);    
    
    return Tnew[1];
end




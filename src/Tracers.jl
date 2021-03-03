# This includes routines that deal with tracers

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
@with_kw struct Tracer
    num         ::  Int64     =  0           # number
    coord       ::  Vector{Float64}          # holds coordinates [2D or 3D]
    T           ::  Float64   =  900         # temperature
    Phase       ::  Int64     =  0           # Phase of the particles        
end
#    Chemistry   ::  Vector{Float64} = []    # Could @ some stage hold the evolving chemistry of the magma

"""
    
    Function that updates properties on tracers

        General form:

        Tracers = UpdateTracers(Tracers, Grid, T, Phi, InterpolationMethod);


        with:
            Tracers:   StructArray that contains tracers 

            Grid:   regular grid on which the parameters to be interpolated are defined
                    2D - (X,Z)
                    3D - (X,Y,Z)
            
            T:      Temperature that is defined on the grid. 

            Phi:    Solid fraction defined on grid   

            InterpolationMethod:    Interpolation method from grid->Tracers
                    "Cubic"     -   Cubic interpolation 
                    "Quadratic" -   Quadratic interpolation (default)
                    "Linear"    -   Linear interpolation

        out:
            Tracers:    Tracers structure with updated T field
"""
function UpdateTracers(Tracers, Grid, T, Phi, InterpolationMethod="Quadratic");

    dim = length(Grid)    
    if isassigned(Tracers,1)        # only if the Tracers StructArray is non-empty
        # extract coordinates
        coord = Tracers.coord; coord = hcat(coord...)';       # extract array with coordinates of tracers
    
        x   = coord[:,1];
        z   = coord[:,end];
        if dim==2;                      
            Points_irregular = (x,z);
        else       
            y   = coord[:,2];   
            Points_irregular = (x,y,z);  
        end

        # Correct coordinates (to stay withoin bounds of models)
        CorrectBounds!(Points_irregular, Grid);
 
        # Interpolate temperature from grid to tracers
        T_tracers = tuple(zeros(size(x)));
        Interpolate!(T_tracers, Grid, tuple(T), Points_irregular, InterpolationMethod);
        
        # Update info on tracers
        for iT = 1:length(Tracers)
            LazyRow(Tracers, iT).T = T_tracers[1][iT];
        end
    end

    return Tracers

end


"""
        AdvectTracers!(Tracers, Grid, Velocity, dt, Method="RK2");

        Advects [Tracers] for one timestep (dt) using the [Velocity] defined on the points [Grid].

        Method: can be "Euler","RK2" or "RK4", for 1th, 2nd or 4th order explicit advection scheme, respectively. 
"""
function AdvectTracers!(Tracers, Grid, Velocity, dt, Method="RK2");
    # Advect tracers forward in time & interpolate T on them 
    
    dim             =   length(Grid);
    coord     =   hcat(Tracers.coord...)';    # extract array with coordinates of tracers
    
    x   = coord[:,1];
    z   = coord[:,end];
    if dim==2
        Points_irregular    =   (x,z);
    else
        y                   =   coord[:,2];
        Points_irregular    =   (x,y,z);
    end
    
    # Correct coordinates (to stay withoin bounds of models)
    CorrectBounds!(Points_irregular, Grid);

    # Advect
    Points_new              =   AdvectPoints(Points_irregular,  Grid,Velocity, dt,Method,  "Linear");     # Advect tracers

    # function to assign properties
    function testnoalloc_2D(sarr, val)
        for (Tracer,x,z) in zip(LazyRows(sarr), val[1], val[2])
            Tracer.coord = [x;z];
        end
    end

    function testnoalloc_3D(sarr, val)
        for (Tracer,x,y,z) in zip(LazyRows(sarr), val[1], val[2], val[3])
            Tracer.coord = [x; y; z];
        end
    end

    if dim==2
        testnoalloc_2D(Tracers, Points_new);
    else
        testnoalloc_3D(Tracers, Points_new);
    end

end
# This includes routines that deal with tracers

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
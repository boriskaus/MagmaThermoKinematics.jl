# This includes routines that deal with tracers

"""
    
    Function that updates properties on tracers

        General form:

        UpdateTracers!(Tracers, Grid, Spacing, T, Phi);


        with:
            Tracers:   StructArray that contains tracers 

            Grid:   regular grid on which the parameters to be interpolated are defined
                    2D - (X,Z)
                    3D - (X,Y,Z)

            Spacing: (constant) spacing of grid
                    2D - (dx,dz)
                    3D - (dx,dy,dz)
            
            T:      Temperature that is defined on the grid. 

            Phi:    Solid fraction defined on grid 
"""
function UpdateTracers(Tracers, Grid, Spacing, T, Phi);

    dim = length(Grid)    
    if length(Tracers)>0
        # extract coordinates
        coord = Tracers.coord; coord = hcat(coord...)';       # extract array with coordinates of tracers
    
        x   = coord[:,1];
        z   = coord[:,end];
        
        # Interpolate on tracers
        if dim==2
            T_tracers   = Interpolate_Linear( Grid, Spacing, T, (x,z));
        else
            y           = coord[:,2];
            T_tracers   = Interpolate_Linear( Grid, Spacing, T, (x,y,z));
        end

       
        #T_tracers = Interpolate_Linear( Grid, Spacing, T, Points_irregular);

        # Update info on tracers
        for iT = 1:length(Tracers)
            LazyRow(Tracers, iT).T = T_tracers[1][iT];
        end
    end

    return Tracers

end
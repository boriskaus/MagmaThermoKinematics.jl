"""
This contains routines related to advection of temperature and tracers

"""


"""
    Performs a linear interpolation, either in 2D or 3D


    General form:
        Data_interp = Interpolate_Linear( Grid, Spacing, Data_grid, Points_irregular)

        with:

            Grid:   regular grid in which the parameters to be interpolated are defined
                    2D - (X,Z)
                    3D - (X,Y,Z)

            Spacing: (constant) spacing of grid
                    2D - (dx,dz)
                    3D - (dx,dy,dz)
            
            DataGrid:   Data that is defined on the grid. Can have 1 field or 2 (2D), respectively 3 (3D) fields 
    
            Points_irregular:   coordinates of irregular points on which we want to interpolate the data
                    2D - (x,z)
                    3D - (x,y,z)
    
            Data_interp:   interpolated data field(s) on the irregular points. Same number of fields as Data_grid                          
"""
function Interpolate_Linear( Grid, Spacing, Data_grid, Points_irregular);

    dim = length(Grid);
                            
    if      dim==2
        Data_interp = Interpolate_2D(Spacing, Grid, Data_grid, Points_irregular);

    elseif  dim==3
        Data_interp = Interpolate_3D(Spacing, Grid, Data_grid, Points_irregular);

    else
        error("interpolation for unknown # of dimensions")
    end


    return Data_interp

end




#--------------------------------------------------------------------------
function Interpolate_2D(Spacing,   Grid,    Data_grid,  Points_irregular);
    # This performs a 2D bilinear interpolation from a regular grid with constant
    # spacing (dx, dz) to irregular points [X_irr, Z_irr]. 
    #

    # NOTE: can potentially be parallelized (need to see how to best go about that)
    dx          =   Spacing[1];
    dz          =   Spacing[2];
    X           =   Grid[1];
    Z           =   Grid[2];
    minX        =   minimum(X);
    minZ        =   minimum(Z);
    X_irr       =   Points_irregular[1];
    Z_irr       =   Points_irregular[2];
    
    nX          =   size(X_irr,1); 
    nZ          =   size(X_irr,2);

    nField      = length(Data_grid);
    Data_reg1   = Data_grid[1];
    if nField==1
        Data_irr    = tuple(zeros(size(X_irr))); # initialize to 0
    elseif nField==2
        Data_irr    = (zeros(size(X_irr)), zeros(size(X_irr)));
        Data_reg2   = Data_grid[2];
    end

    if 1==1 # vectorized code

        ix1 = floor.(Int64, (X_irr[:] .- minX)./dx) .+ 1;    ix1[ix1.==nX] .= nX-1;
        iz1 = floor.(Int64, (Z_irr[:] .- minZ)./dz) .+ 1;    iz1[iz1.==nZ] .= nZ-1;

        ind1 = zeros(Int64, nX*nZ,1);
        ind2 = zeros(Int64, nX*nZ,1);
        ind3 = zeros(Int64, nX*nZ,1);
        ind4 = zeros(Int64, nX*nZ,1);

        for i=1:length(ix1); 
            ind1[i] = LinearIndices(X)[ix1[i]  ,iz1[i]  ]; 
            ind2[i] = LinearIndices(X)[ix1[i]+1,iz1[i]  ]; 
            ind3[i] = LinearIndices(X)[ix1[i]  ,iz1[i]+1]; 
            ind4[i] = LinearIndices(X)[ix1[i]+1,iz1[i]+1]; 
        end

        x1          = (X_irr[:] .- X[ind1])./dx;   
        z1          = (Z_irr[:] .- Z[ind1])./dz;   
        α           = [(1.0.-x1).*(1.0.-z1), (    x1).*(1.0.-z1), (1.0.-x1).*(    z1),   (    x1).*(    z1)  ];

        f1          = [Data_reg1[ind1],      Data_reg1[ind2],     Data_reg1[ind3],       Data_reg1[ind4]     ];

        Data_irr1_v  =  zeros(Float64, nX*nZ,1);
        Data_irr1_v  =  α[1].*f1[1] + α[2].*f1[2] + α[3].*f1[3] + α[4].*f1[4];
        if nZ>1
            Data_irr[1]     .=  reshape(Data_irr1_v, (nX,nZ));
        else
            Data_irr[1][:]   =  Data_irr1_v;
        end

        if nField==2
            f2          =   [Data_reg2[ind1],      Data_reg2[ind2],     Data_reg2[ind3],       Data_reg2[ind4]     ];        
            Data_irr2_v =   α[1].*f2[1] + α[2].*f2[2] + α[3].*f2[3] + α[4].*f2[4];
            if nZ>1
                Data_irr[2]     .=  reshape(Data_irr2_v, (nX,nZ));
            else

                Data_irr[2][:]   =  Data_irr2_v;
            end
        end
    end

    if 1==0         # same with a loop

        for ix=1:nX
            for iz=1:nZ
            
                ix1 = floor(Int64, (X_irr[ix,iz] - minX)/dx) + 1;    #ix1[ix1.==nX] .= nX-1;
                iz1 = floor(Int64, (Z_irr[ix,iz] - minZ)/dz) + 1;    #iz1[iz1.==nZ] .= nZ-1;
                if (ix1==nX); ix1 = nX-1;   end
                if iz1==nZ; iz1 = nZ-1;     end

                i    = 1;
                ind1 = LinearIndices(Data_reg1)[ix1[1]  , iz1[1]  ]; 
                ind2 = LinearIndices(Data_reg1)[ix1[1]+1, iz1[1]  ]; 
                ind3 = LinearIndices(Data_reg1)[ix1[1]  , iz1[1]+1]; 
                ind4 = LinearIndices(Data_reg1)[ix1[1]+1, iz1[1]+1]; 

                x1                  = (X_irr[ix,iz] .- X[ix1,iz1])./dx;   
                z1                  = (Z_irr[ix,iz] .- Z[ix1,iz1])./dz;   
                α                   = [(1.0-x1)*(1.0-z1), (    x1)*(1.0-z1), (1.0-x1)*(    z1),   (    x1)*(    z1)  ];
                f1                  = [Data_reg1[ind1],      Data_reg1[ind2],     Data_reg1[ind3],       Data_reg1[ind4]     ];

                Data_irr[1][ix,iz]  = 0.0;
                for i=1:4; Data_irr[1][ix,iz]  =  Data_irr[1][ix,iz] + α[i].*f1[i]; end

                if nField==2
                    f2                  =   [Data_reg2[ind1],      Data_reg2[ind2],     Data_reg2[ind3],       Data_reg2[ind4]     ];        
                    Data_irr[2][ix,iz]  =   0.0;
                    for i=1:4; Data_irr[2][ix,iz]  =  Data_irr[2][ix,iz] + α[i].*f1[i]; end

                end

            end
        end

    end

    return Data_irr

end


#--------------------------------------------------------------------------
function Interpolate_3D(Spacing,   Grid,    Data_grid,  Points_irregular);

    # NOTE: can potentially be parallelized (need to see how to best go about that)
    dx          =   Spacing[1];
    dy          =   Spacing[2];
    dz          =   Spacing[3];
    X           =   Grid[1];
    Y           =   Grid[2];
    Z           =   Grid[3];
    minX        =   minimum(X);
    minY        =   minimum(Y);
    minZ        =   minimum(Z);
    X_irr       =   Points_irregular[1];
    Y_irr       =   Points_irregular[2];
    Z_irr       =   Points_irregular[3];
    
    nX          =   size(X_irr,1); 
    nY          =   size(X_irr,2); 
    nZ          =   size(X_irr,3);

    nField      = length(Data_grid);
    Data_reg1   = Data_grid[1];
    if nField==1
        Data_irr    = tuple(zeros(size(X_irr))); # initialize to 0
    elseif nField==3
        Data_irr    = (zeros(size(X_irr)), zeros(size(X_irr)), zeros(size(X_irr)));
        Data_reg2   = Data_grid[2];
        Data_reg3   = Data_grid[3];
        
    else
        error("unknown number of data fields")
    end


    for ix=1:nX
        for iy=1:nY
            for iz=1:nZ
            
                ix1 = floor(Int64, (X_irr[ix,iy,iz] - minX)/dx) + 1;   
                iy1 = floor(Int64, (Y_irr[ix,iy,iz] - minY)/dy) + 1;   
                iz1 = floor(Int64, (Z_irr[ix,iy,iz] - minZ)/dz) + 1;   
                if ix1==nX; ix1 = nX-1;   end
                if iy1==nY; iy1 = nY-1;   end
                if iz1==nZ; iz1 = nZ-1;   end

                ind1 = LinearIndices(Data_reg1)[ix1[1]  , iy1[1]  , iz1[1]  ]; 
                ind2 = LinearIndices(Data_reg1)[ix1[1]+1, iy1[1]  , iz1[1]  ]; 
                ind3 = LinearIndices(Data_reg1)[ix1[1]  , iy1[1]+1, iz1[1]  ]; 
                ind4 = LinearIndices(Data_reg1)[ix1[1]+1, iy1[1]+1, iz1[1]  ]; 
                ind5 = LinearIndices(Data_reg1)[ix1[1]  , iy1[1]  , iz1[1]+1]; 
                ind6 = LinearIndices(Data_reg1)[ix1[1]+1, iy1[1]  , iz1[1]+1]; 
                ind7 = LinearIndices(Data_reg1)[ix1[1]  , iy1[1]+1, iz1[1]+1]; 
                ind8 = LinearIndices(Data_reg1)[ix1[1]+1, iy1[1]+1, iz1[1]+1]; 
                
                x1                  = (X_irr[ix,iy,iz] .- X[ix1,iy1,iz1])./dx;   
                y1                  = (Y_irr[ix,iy,iz] .- Y[ix1,iy1,iz1])./dy;   
                z1                  = (Z_irr[ix,iy,iz] .- Z[ix1,iy1,iz1])./dz;   
                α                   = [ (1.0-x1)*(1.0-y1)*(1.0-z1), (    x1)*(1.0-y1)*(1.0-z1), (1.0-x1)*(    y1)*(1.0-z1),   (    x1)*(    y1)*(1.0-z1),  
                                        (1.0-x1)*(1.0-y1)*(    z1), (    x1)*(1.0-y1)*(    z1), (1.0-x1)*(    y1)*(    z1),   (    x1)*(    y1)*(    z1)];
                f1                  = [Data_reg1[ind1],      Data_reg1[ind2],     Data_reg1[ind3],       Data_reg1[ind4],
                                       Data_reg1[ind5],      Data_reg1[ind6],     Data_reg1[ind7],       Data_reg1[ind8]      ];

                Data_irr[1][ix,iy,iz]  = 0.0;
                for i=1:8; Data_irr[1][ix,iy,iz]  =  Data_irr[1][ix,iy,iz] + α[i].*f1[i]; end

                if nField==3
                    f2                  = [ Data_reg2[ind1],      Data_reg2[ind2],     Data_reg2[ind3],       Data_reg2[ind4],
                                            Data_reg2[ind5],      Data_reg2[ind6],     Data_reg2[ind7],       Data_reg2[ind8]      ];
                    Data_irr[2][ix,iy,iz]  = 0.0;
                    for i=1:8; Data_irr[2][ix,iy,iz]  =  Data_irr[2][ix,iy,iz] + α[i].*f2[i]; end

                    f3                  = [ Data_reg3[ind1],      Data_reg3[ind2],     Data_reg3[ind3],       Data_reg3[ind4],
                                            Data_reg3[ind5],      Data_reg3[ind6],     Data_reg3[ind7],       Data_reg3[ind8]      ];
                    Data_irr[3][ix,iy,iz]  = 0.0;
                    for i=1:8; Data_irr[3][ix,iy,iz]  =  Data_irr[3][ix,iy,iz] + α[i].*f3[i]; end

                end

            end
        end
    end
    
    return Data_irr

end


"""
    AdvPoints =   AdvectPoints(AdvPoints0, Grid,Velocity,Spacing,dt, Method="RK4");
    
Advects irregular points described by the (2D or 3D tuple) AdvPoints0, though a fixed Eulerian
grid (Grid), with constant spacing (Spacing) on which the velocity components (Velocity) are defined.
Advection is done for the time dt, and can use different methods

"""
function AdvectPoints(AdvPoints0, Grid,Velocity,Spacing,dt, Method="RK4");
    
    
    dim         = length(AdvPoints0);           # number of dimensions
    AdvPoints   = map(x->x.*0, AdvPoints0) ;    # initialize to 0

    # Different advection schemes can be used
    if Method=="Euler"
        Velocity_int = Interpolate_Linear( Grid, Spacing, Velocity, AdvPoints0);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end
        AdvPoints  =   CorrectBounds!( AdvPoints , Grid);
        
    elseif Method=="RK2"
        Velocity_int = Interpolate_Linear( Grid, Spacing, Velocity, AdvPoints0);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds!( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate_Linear( Grid, Spacing, Velocity, AdvPoints);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end    
        AdvPoints  =   CorrectBounds!( AdvPoints , Grid);                               # step k2

    elseif Method=="RK4"
        Velocity_int = Interpolate_Linear( Grid, Spacing, Velocity, AdvPoints0);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds!( AdvPoints , Grid);                               # step k1
        
        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate_Linear( Grid, Spacing, Velocity, AdvPoints0);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds!( AdvPoints , Grid);                               # step k2
        
        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate_Linear( Grid, Spacing, Velocity, AdvPoints0);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt/2.0;  
        end    
        AdvPoints  =   CorrectBounds!( AdvPoints , Grid);                               # step k3

        # Interpolate velocity values on deformed grid
        Velocity_int = Interpolate_Linear( Grid, Spacing, Velocity, AdvPoints0);    
        for i=1:dim; 
            AdvPoints[i]  .= AdvPoints0[i] .+ Velocity_int[i].*dt;  
        end             
        AdvPoints  =   CorrectBounds!( AdvPoints , Grid);                               # step k4
        
    end


    return AdvPoints;

end

"""
    Points_new = CorrectBounds!(Points, Grid);
    
Ensures that the coordinates of Points stay within the bounds
of the regular grid Grid, which is a tuple of 2 or 3 field (for 2D/3D)

"""
function CorrectBounds!(Points, Grid);

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
        Tnew = AdvectTemperature(T, Grid, Velocity, Spacing, dt, Method="RK4")

    Advects temperature for one timestep dt, using a semi-lagrangian advection scheme 

        Method: can be "Euler","RK2" or "RK4", for 1th, 2nd or 4th order explicit advection scheme, respectively. 
"""
function AdvectTemperature(T::Array,Grid, Velocity, Spacing, dt, Method="RK4");
    
    # 1) Use semi-lagrangian advection to advect temperature
    # Advect regular grid backwards in time
    PointsAdv = Grid;
    PointsAdv = AdvectPoints(PointsAdv, Grid,Velocity,Spacing,-dt,Method);
 
    # 2) Interpolate temperature on deformed points
    Tnew = Interpolate_Linear(Grid, Spacing, tuple(T), PointsAdv);    

    return Tnew[1];
end


"""
        Tnew = AdvectTracers(Tracers, T::Array,Grid, Velocity, Spacing, dt, Method="RK4");

        Advects [Tracers] for one timestep (dt) using the [Velocity] defined on the points [Grid]
        that have constant [Spacing].

        Method: can be "Euler","RK2" or "RK4", for 1th, 2nd or 4th order explicit advection scheme, respectively. 
"""
function AdvectTracers(Tracers, T::Array,Grid, Velocity, Spacing, dt, Method="RK4");
    # Advect tracers forward in time & interpolate T on them 
    
    dim = length(Grid);

    coord = Tracers.coord; coord = hcat(coord...)';       # extract array with coordinates of tracers
    
    x   = coord[:,1];
    z   = coord[:,end];
    if dim==2
        Points_new  =   AdvectPoints((x,z),    Grid,Velocity,Spacing, dt,Method);     # Advect tracers
    else
        y           =   coord[:,2];
        Points_new  =   AdvectPoints((x,y,z),  Grid,Velocity,Spacing, dt,Method);     # Advect tracers
    end
 
    for iT = 1:length(Tracers)
        if dim==2
            LazyRow(Tracers, iT).coord = [Points_new[1][iT]; Points_new[2][iT]];
       
        elseif dim==3
            LazyRow(Tracers, iT).coord = [Points_new[1][iT]; Points_new[2][iT]; Points_new[3][iT]];
        end
    end

    return Tracers
end

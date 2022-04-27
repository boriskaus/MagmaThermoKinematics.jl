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
            
            T:          Temperature of the tracer [Celcius]     
            time_vec:   Vector with time
            T_vec :     Vector with temperature values                         
"""
@with_kw mutable struct Tracer
    num         ::  Int64     =  0           # number
    coord       ::  Vector{Float64}          # holds coordinates [2D or 3D]
    T           ::  Float64   =  900         # temperature
    Phase       ::  Int64     =  1           # Phase (aka rock type) of the Tracer      
    Phi_melt    ::  Float64   =  0           # Melt fraction on Tracers
    time_vec    ::  Vector{Float64} = []     # Time vector
    T_vec       ::  Vector{Float64} = []     # Temperature vector
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
            Tracers:    Tracers structure with updated T and melt fraction fields
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

        # Correct coordinates (to stay within bounds of models)
        CorrectBounds!(Points_irregular, Grid);
 
        # Interpolate temperature from grid to tracers
        T_tracers           = tuple(zeros(size(x)));
        Interpolate!(T_tracers,         Grid, tuple(T), Points_irregular, InterpolationMethod);
        
        Phi_melt_tracers    = tuple(zeros(size(x)));
        Interpolate!(Phi_melt_tracers,  Grid, tuple(1.0 .- Phi), Points_irregular, InterpolationMethod);    # 1-Phi, as Phi=solid fraction
      
        # Update info on tracers
        for iT = 1:length(Tracers)
            LazyRow(Tracers, iT).T          = T_tracers[1][iT];             # Temperature
            LazyRow(Tracers, iT).Phi_melt   = Phi_melt_tracers[1][iT];      # Melt fraction
        end
        

        #=
        Bound_min = minimum.(Grid)
        Bound_max = maximum.(Grid)

        # Create linear interpolation objects
        itp_T     =   interpolate(T, BSpline(Linear()));
        interp_T  =   scale(itp_T,Grid...);
        
        itp_Phi   =   interpolate(Phi, BSpline(Linear()));
        interp_ϕ  =   scale(itp_Phi,Grid...);
        
        for iT = 1:length(Tracers)  
            Trac = Tracers[iT];
            x = Trac.coord[1];
            z = Trac.coord[2];

            # correct points for bounds
            if x<Bound_min[1]; x=Bound_min[1]; end
            if x>Bound_max[1]; x=Bound_max[1]; end
            if z<Bound_min[2]; z=Bound_min[2]; end
            if z>Bound_max[2]; z=Bound_max[2]; end

            if dim==2;                      
                Points_irregular = (x,z);
            else       
                y   = coord[:,3];   
                if z<Bound_min[3]; z=Bound_min[3]; end
                if z>Bound_max[3]; z=Bound_max[3]; end
                Points_irregular = (x,z,y);  
            end

            # Interpolate:
            Trac_T              = interp_T(Points_irregular...);
            Trac_ϕ              = interp_ϕ(Points_irregular...);

            # Update values on tracer
            LazyRow(Tracers, iT).T        = Trac_T;
            LazyRow(Tracers, iT).Phi_melt = Trac_ϕ;
        end
        
    =#
    end
    
    return Tracers

end


"""
    UpdateTracers_T_ϕ!(Tracers, Grid::Tuple, T, Phi);
    
In-place function that interpolates `T` & `Phi`, defined on the `Grid`, to `Tracers`.

- Tracers:  StructArray that contains tracers, where we want to update  
- Grid:     Regular grid on which the parameters to be interpolated are defined
                2D - (X,Z)
                3D - (X,Y,Z)
- T:  `T` field that is defined on the grid, to be interpolated to tracers 
- Phi:  `Phi` field that is defined on the grid, to be interpolated to tracers 

Note that we employ linear interpolation using custom functions
"""
function UpdateTracers_T_ϕ!(Tracers, Grid::Tuple, T::AbstractArray{_T,dim}, Phi::AbstractArray{_T,dim}) where {_T, dim}

    if isassigned(Tracers,1)        # only if the Tracers StructArray is non-empty
        
        # Boundaries of the grid
        Bound_min = minimum.(Grid)
        Bound_max = maximum.(Grid)
        
        if dim==2
            Δx = Grid[1][2]-Grid[1][1]
            Δz = Grid[2][2]-Grid[2][1]
        elseif dim==3
            Δx = Grid[1][2]-Grid[1][1]
            Δy = Grid[2][2]-Grid[2][1]
            Δz = Grid[3][2]-Grid[3][1]
        end

        for iT = 1:length(Tracers)  
            Trac = Tracers[iT];
            pt   = Trac.coord

            # correct point for bounds:
            for i=1:dim
                if pt[i]<Bound_min[i]; pt[i] = Bound_min[i]; end
                if pt[i]>Bound_max[i]; pt[i] = Bound_max[i]; end
            end                
            
            # Linear interpolation:
            if dim==2
                Trac_T = interpolate_linear_2D(pt[1], pt[2], Bound_min, Δx, Δz, T   )
                Trac_ϕ = interpolate_linear_2D(pt[1], pt[2], Bound_min, Δx, Δz, Phi )
                if Trac_T>1001
                    @show pt, Bound_min, Δx, Δz, Trac_T
                end
            elseif dim==3
                Trac_T = interpolate_linear_3D(pt[1], pt[2], pt[3], Bound_min, Δx, Δy, Δz, T   )
                Trac_ϕ = interpolate_linear_3D(pt[1], pt[2], pt[3], Bound_min, Δx, Δy, Δz, Phi )
            end

            # Update values on tracers
            LazyRow(Tracers, iT).T        = Trac_T;
            LazyRow(Tracers, iT).Phi_melt = Trac_ϕ;

        end
        
    end

    @show maximum(Tracers.T)
    
    return nothing

end

""" 

Implements 2D bilinear interpolation 
"""
function interpolate_linear_2D(pt_x, pt_z, Bound_min, Δx, Δz, Field )

    ix = floor(Int64, (pt_x - Bound_min[1])/Δx)
    iz = floor(Int64, (pt_z - Bound_min[2])/Δz)
    fac_x = (pt_x - ix*Δx - Bound_min[1])/Δx     # distance to lower left point
    fac_z = (pt_z - iz*Δz - Bound_min[2])/Δz     # distance to lower left point
    if fac_x<0.0 || fac_x>1.0
        @show fac_x, pt_x, ix, Δx,  Bound_min[1]
    end
    if fac_z<0.0 || fac_z>1.0
        @show fac_z, pt_z, iz, Δz,  Bound_min[2]
    end

    
    # interpolate in x    
    val_x_bot =  (1.0-fac_x)*Field[ix+1,iz+1] +  ( fac_x)*Field[ix+2,iz+1]
    val_x_top =  (1.0-fac_x)*Field[ix+1,iz+2] +  ( fac_x)*Field[ix+2,iz+2]
     
    # Interpolate value in z
    val    = (1.0-fac_z)*val_x_bot + fac_z*val_x_top

    return val 
end

""" 

Implements 3D trilinear interpolation 
"""
function interpolate_linear_3D(pt_x, pt_y, pt_z, Bound_min, Δx, Δy, Δz, Field )

    ix = floor(Int64, (pt_x - Bound_min[1])/Δx)
    ix = floor(Int64, (pt_y - Bound_min[2])/Δy)
    iz = floor(Int64, (pt_z - Bound_min[3])/Δz)
    fac_x = (pt_x - ix*Δx - Bound_min[1])/Δx     # distance to lower left point
    fac_y = (pt_y - iy*Δy - Bound_min[2])/Δy     # distance to lower left point
    fac_z = (pt_z - iz*Δz - Bound_min[3])/Δz     # distance to lower left point

    # Interpolate in x    
    val_x_bot_left  =  (1.0-fac_x)*Field[ix+1,iy+1,iz+1] +  ( fac_x)*Field[ix+2,iy+1,iz+1]
    val_x_top_left  =  (1.0-fac_x)*Field[ix+1,iy+1,iz+2] +  ( fac_x)*Field[ix+2,iy+1,iz+2]
    val_x_bot_right =  (1.0-fac_x)*Field[ix+1,iy+2,iz+1] +  ( fac_x)*Field[ix+2,iy+2,iz+1]
    val_x_top_right =  (1.0-fac_x)*Field[ix+1,iy+2,iz+2] +  ( fac_x)*Field[ix+2,iy+2,iz+2]
    
    # Interpolate in y    
    val_y_bot       =  (1.0-fac_y)*val_x_bot_left + fac_y*val_x_bot_right
    val_y_top       =  (1.0-fac_y)*val_x_top_left + fac_y*val_x_top_right
    
    # Interpolate value in z
    val             = (1.0-fac_z)*val_y_bot + fac_z*val_y_top

    return val 
end

"""
    This evenly populates the grid with tracers

        General form:

        Tracers     = InitializeTracers(Grid, NumTracersDir=3, RandomPertur=true);

        with:
            Grid:   2D or 3D arrays that describe the grid coordinates
                    2D - (X,Z)
                    3D - (X,Y,Z)
            
            NumTracersDir:    The number of tracers per direction 

            RandomPertur:     Add slight random noise on tracer location?

        out:
            Tracers:    Tracers structure 
"""
function InitializeTracers(Grid, NumTracersDir=3, RandomPertur=true);

    dim             =   length(Grid)  
    R               =   CartesianIndices(Grid[1])
    Ifirst, Ilast   =   first(R), last(R)
    I1              =   oneunit(Ifirst)
    numTr           =   0;
    cen,d           =   zeros(dim), zeros(dim)
    coord_loc       =   zeros(NumTracersDir^dim,dim);
    Tracers         =   StructArray{Tracer}(undef, 1)                                    # Initialize Tracers structure

    # create a 'basic' Tracers struct
    t               =   Tracer(num=numTr, coord=zeros(dim), T=0.0, Phase=1);
    Tracers0        =   StructArray([t]);
    for i=1:NumTracersDir^dim-1
        append!(Tracers0,[t])
    end
    
    # The main assumption is that the grid coordinates specified in GRID are the corner points 
    # of the cells. The control is given between the cells, as illustrated in the sketch below
    #
    #  X(ix,iz+1)
    #       | o  o  o  |
    #       x----------x  
    #       |  o   o   |
    #       |   o    o |
    #  X(ix,iz)       X(ix+1,iz)
    #       x----------x
    #
    #
    # Note that the CartesianIndices in julia allow writing quite general code that works in any dimension
    
    # Determine center coordinate of current cell & width in every direction
    for idim=1:dim
        d[idim]    = (Grid[idim][Ifirst+I1] -   Grid[idim][Ifirst]);         # spacing of grid cells 
    end
    d1      =   d/(NumTracersDir); 
    xl      =   (-d[  1]/2. + d1[  1]/2.)
    zl      =   (-d[end]/2. + d1[end]/2.)
    xs      =   xl:d1[  1]: (xl+(NumTracersDir-1)*d1[  1]);
    zs      =   zl:d1[end]: (zl+(NumTracersDir-1)*d1[end]);
    if dim==3
        yl  =   (-d[2]/2. + d1[2]/2.)
        ys  =   yl:d1[2]: (yl+(NumTracersDir-1)*d1[2]);
    end
    
    # Generate local, regular, coordinate arrays for the new tracers
    if dim==2
        coord_loc0 = [ [x,z]     for x=xs for z=zs];             # Creates a tuple with coords
    elseif dim==3
        coord_loc0 = [ [x,y,z]   for x=xs for y=ys for z=zs];    # Creates a tuple with coords
    end
    coord_loc = coord_loc0;

    for I = Ifirst:Ilast-I1         
       
      
        for idim=1:dim
            cen[idim]  = (Grid[idim][I+I1] +   Grid[idim][I])/2.0;     # center of control volume
            d[idim]    = (Grid[idim][I+I1] -   Grid[idim][I]);         # spacing of grid cells 
        end

       
        ## THIS SECTION ALLOCATES A LOT EVEN WHEN WE HAVE IT FALSE  ---
        # add random perturbation if requested
        if RandomPertur

            if dim==2
                randm       = [(map(rand,(Float64,Float64)) .- 0.5).*2.0 .* (d1[1],d1[2]) for i=eachindex(coord_loc)];        #
                coord_loc   = [ coord_loc0[i] .+ randm[i] for i=eachindex(coord_loc0)];       # This can potentially be done in-place?   
            elseif dim==3
                randm       = [(map(rand,(Float64,Float64, Float64)) .- 0.5).*2.0 .* (d1[1],d1[2],d1[3]) for i=eachindex(coord_loc)];        #
                coord_loc   = [ coord_loc0[i] .+ randm[i] for i=eachindex(coord_loc0)];       # This can potentially be done in-place?
            end
        end
     
        #if 1==0
        
        # Add new tracers with perturbed coords to struct
        
        ## THIS LINE ALLOCATES EVEN MORE, BECAUSE IT IS CALLED MANY TIMES
        ReplaceTracerFields!(Tracers0, coord_loc,cen,size(Tracers,1));     # Replace coord & num and add cen to coordinate
        
        ## THIS LINE IS SLOW BUT DOESN'T ALLOCATE ALL THAT MUCH:
        append!(Tracers, Tracers0);                                             # Extend the Tracers structure and add new fields to it
       # end
    end

    # delete first field, which was empty   
    StructArrays.foreachfield(v -> deleteat!(v, 1), Tracers);    


    return Tracers

end

function ReplaceTracerFields!(Tracers0, coord_loc::Array, cen::Array, num_start::Int)
    for i=1:length(coord_loc)
        Tracers0.coord[i]  = coord_loc[i] + cen;
        Tracers0.num[i]    = i+num_start-1; 
    end
end


"""
    This computes the PhaseRatio on the grid specified by Grid

        General form:

        PhaseRatio,NumTracers = PhaseRatioFromTracers(Grid, Tracers, InterpolationMethod, BackgroundPhase=2, RequestNumTracers=false; BackgroundPhase=2);

        with:
            Grid:   2D or 3D arrays that describe the grid coordinates
                    2D - (X,Z)
                    3D - (X,Y,Z)
            
            Tracers:    Tracers structure 

            InterpolationMethod:    Interpolation method used to go from Tracers ->  Grid
                    "Constant"          -   All particles within a distance [dx,dy,dz] around the grid point 
                    "DistanceWeighted"  -   Particles closer to the grid point have a stronger weight.
                                            This follows what is described in:
                                                Duretz, T., May, D.A., Gerya, T.V., Tackley, P.J., 2011. Discretization errors and 
                                                free surface stabilization in the finite difference and marker-in-cell method for applied geodynamics: 
                                                A numerical study: Geochem. Geophys. Geosyst. 12, https://doi.org/10.1029/2011GC00356
                                                
            BackgroundPhase:        The background phase (used for places that don't have cells, nor surrounding cells )

            RequestNumTracers:      Return the number of tracers on every grid cell (default=false)

        optional parameters:

            BackgroundPhase:        if this is defined, points where no tracers are present, will get the number BackgroundPhase.
                                    You need to define this with a keywords as: (, BackgroundPhase=2) 

        out:
            PhaseRatio:    Phase ratio on the gridpoints defined by Grid

            NumTracers:    The number of tracers per grid point
            
"""
function PhaseRatioFromTracers(FullGrid, Grid, Tracers, InterpolationMethod="Constant", RequestNumTracers=false; BackgroundPhase=nothing);

    numPhases       =   maximum(Tracers.Phase);
    if ~isnothing(BackgroundPhase)
        if numPhases<BackgroundPhase; numPhases=BackgroundPhase; end
    end
    dim             =   length(FullGrid)  
    Nx              =   size(FullGrid[1],1);
    if dim==2
        Nz          =   size(FullGrid[1],2);
        siz         =   [Nx,Nz,numPhases];
    else
        Ny, Nz      =   size(FullGrid[1],2), size(FullGrid[1],3);
        siz         =   [Nx,Ny,Nz,numPhases];
    end
    PhaseRatio      =   zeros(Tuple(siz));                  # will hold phase ratio @ the end for every phase 
    
    R               =   CartesianIndices(FullGrid[1])
    Ifirst, Ilast   =   first(R), last(R)
    I1              =   oneunit(Ifirst)

    # We assume that spacing is constant in all directions; 
    #   If that is not the case the algorithm becomes a bit more complicated (not implemented)
    d               =   zeros(dim)
    for idim=1:dim
        d[idim]     =   (FullGrid[idim][Ifirst+I1] -   FullGrid[idim][Ifirst]);         # spacing of grid cells 
    end
    coord           =   Tracers.coord; coord = hcat(coord...)';                         # extract array with coordinates of all tracers

    # Correct coordinates of tracers (to stay within bounds of grid), to not mess up the interpolation below
    CorrectBounds_Array!(coord, Grid);
  
    iPhase          =   Tracers.Phase;  
    indX            =   CartesianIndices( FullGrid[  1]);

    IndPoints       =   zeros(Int,length(Tracers),1);
    if dim==2
        itp         =   interpolate(LinearIndices(FullGrid[1]), BSpline(Constant()));   # 2D interpolation that has indices
        sitp        =   scale(itp, Grid[1], Grid[2]);
        evaluate_interp_Int_2D(IndPoints, sitp,coord);                                  # this gives the full 2D/3D index
    else
        itp         =   interpolate(LinearIndices(FullGrid[1]), BSpline(Constant()));   # 2D interpolation that has indices
        sitp        =   scale(itp, Grid[1], Grid[2], Grid[3]);
        evaluate_interp_Int_3D(IndPoints, sitp,coord);                                  # this gives the full 2D/3D index
    end

    NumTracers  =   zeros(Int64,Tuple(siz[1:end-1]));   # Keep track of # of tracers around every point
    indNum      =   CartesianIndices(NumTracers);
    
    if InterpolationMethod=="DistanceWeighted"
        # In case we use a distance based weighting, compute the weight factor here
        X_g             =   FullGrid[1  ][IndPoints];  # Gridpoint to which the particle belongs 
        Z_g             =   FullGrid[end][IndPoints];  # Gridpoint to which the particle belongs 
        Weight      =   zeros(size(coord,1),1)
        if dim==2
            evaluate_weight_2D(Weight, coord, X_g, Z_g, d[1], d[end]);
        elseif dim==3
            Y_g         =   FullGrid[2  ][IndPoints];   
            evaluate_weight_3D(Weight, coord, X_g, Y_g, Z_g, d[1], d[2], d[end]);
        end
    
    elseif InterpolationMethod=="Constant"
        Weight          =   ones(length(Tracers),1); 
    else
        error("Unknown InterpolationMethod=$InterpolationMethod. Choose: [Constant] or [DistanceWeighted]. ")
    end
    
    indPhase             =   CartesianIndices(PhaseRatio);
    if dim==2
        Add_Phase_2D(PhaseRatio, NumTracers, length(Tracers), IndPoints, indPhase, indX, iPhase, indNum, Weight)
    elseif dim==3
        Add_Phase_3D(PhaseRatio, NumTracers, length(Tracers), IndPoints, indPhase, indX, iPhase, indNum, Weight)
    end


    if ~isnothing(BackgroundPhase)
        # All "empty" cells will be set to the background phase
        #
        # This is particularly useful in case we have models where Tracers do not 'fill'
        # the full model space 

        ind_empty       = findall(x->x==0, NumTracers);
        for I in ind_empty
            # Use phase ratios from nearby point with most tracers
            if dim==2;  PhaseRatio[:,:  ,BackgroundPhase]    .=   1.0;     
            else        PhaseRatio[:,:,:,BackgroundPhase]    .=   1.0;
            end

            NumTracers[I] = 1;
        end

    else 
        # This is for cases where we do cover the full model domain     
        # 
        # Deal with points that have zero tracers @ this stage 
        #  We do this by a query of the surrounding (9/27 in 2D/3D) points and use the properties of the closest point
        #
        # TODO: Deal with cases in which tracers are not defined in the full domain, but only used to track dikes 
        #       and their properties, while otherwise having a fixed background 
        ind_empty       = findall(x->x==0, NumTracers);
        R               = CartesianIndices(NumTracers)
        Ifirst, Ilast   = first(R), last(R)
        I1              = oneunit(Ifirst)
        for I in ind_empty
            @show I, NumTracers[I]

            # query surrounding points & store the index of the point with the most particles
            smax = 0;   Imax = I;
            for J in max(Ifirst, I-I1):min(Ilast, I+I1)
                if NumTracers[J]>smax
                    Imax = J;
                    smax = NumTracers[J];
                end
            end
            
            # Use phase ratios from nearby point with most tracers
            for iPhase=1:numPhases
                if dim==2;  PhaseRatio[I[1],I[2],     iPhase]   =   PhaseRatio[Imax[1],Imax[2],iPhase];     
                else        PhaseRatio[I[1],I[2],I[3],iPhase]   =   PhaseRatio[Imax[1],Imax[2],Imax[3],iPhase];
                end
            end

            NumTracers[I] = 1;
        end

      # in case we still have problematic cells, at least spit out a warning
      if any(NumTracers .== 0 )
        warning("We still have cells that have no tracers and we did not manage to fix this by using nearby cells. ")
        end

    end
    sumPhaseRatio   =   sum(PhaseRatio,dims=dim+1);

    # Normalize phase ratio, such that sum=1
    for iPhase=1:numPhases
        if dim==2;  PhaseRatio[:,:  ,iPhase]    =   PhaseRatio[:,:  ,iPhase]./sumPhaseRatio;
        else        PhaseRatio[:,:,:,iPhase]    =   PhaseRatio[:,:,:,iPhase]./sumPhaseRatio; 
        end
    end

    if RequestNumTracers
        return PhaseRatio, NumTracers
    else
        return PhaseRatio
    end
end


"""
    This averages a certain property from tracers -> grid. The data will ONLY be set on the grid if we have at least x tracers!
        if not, the origibal data will be kept

        General form:   

            TracersToGrid!(Data, FullGrid, Grid, Tracers, Property="T", InterpolationMethod="Constant", RequestNumTracers=false);

        with:
            Data:       2D or 3D arrays that have the grid coordinates
            
            FullGrid:   2D or 3D arrays that describe the grid coordinates
                        2D - (X,Z)
                        3D - (X,Y,Z)
            
            Grid:       1D vectors describing the cartesian grid            
            
            Tracers:    Tracers structure 

            Property:   Property to be interpolated from Tracers2Grid 
                        "T" - temperature

            InterpolationMethod:    Interpolation method used to go from Tracers ->  Grid
                    "Constant"          -   All particles within a distance [dx,dy,dz] around the grid point 
                    "DistanceWeighted"  -   Particles closer to the grid point have a stronger weight.
                                            This follows what is described in:
                                                Duretz, T., May, D.A., Gerya, T.V., Tackley, P.J., 2011. Discretization errors and 
                                                free surface stabilization in the finite difference and marker-in-cell method for applied geodynamics: 
                                                A numerical study: Geochem. Geophys. Geosyst. 12, https://doi.org/10.1029/2011GC003567

            RequestNumTracers:    Return the number of tracers on every grid cell (default=false)

        out:
            PhaseRatio:    Phase ratio on the gridpoints defined by Grid

            NumTracers:    The number of tracers per grid point
            
"""
function TracersToGrid!(Data, FullGrid, Grid, Tracers, Property="T", InterpolationMethod="Constant", RequestNumTracers=false);

    numPhases       =   maximum(Tracers.Phase);
    dim             =   length(FullGrid)  
    Nx              =   size(FullGrid[1],1);
    if dim==2
        Nz          =   size(FullGrid[1],2);
    else
        Ny, Nz      =   size(FullGrid[1],2), size(FullGrid[1],3);
    end
    Data_local      =   copy(Data) .* 0.0;                  # will hold the data @ the end
    
    R               =   CartesianIndices(FullGrid[1])
    Ifirst, Ilast   =   first(R), last(R)
    I1              =   oneunit(Ifirst)

    # We assume that spacing is constant in all directions; 
    #   If that is not the case the algorithm becomes a bit more complicated (not implemented)
    d               =   zeros(dim)
    for idim=1:dim
        d[idim]     =   (FullGrid[idim][Ifirst+I1] -   FullGrid[idim][Ifirst]);         # spacing of grid cells 
    end
    coord           =   Tracers.coord; coord = hcat(coord...)';                         # extract array with coordinates of all tracers

    # Correct coordinates of tracers (to stay within bounds of grid), to not mess up the interpolation below
    CorrectBounds_Array!(coord, Grid);
  
    indX            =   CartesianIndices( FullGrid[  1]);
    IndPoints       =   zeros(Int,length(Tracers),1);
    if dim==2
        itp         =   interpolate(LinearIndices(FullGrid[1]), BSpline(Constant()));   # 2D interpolation that has indices
        sitp        =   scale(itp, Grid[1], Grid[2]);
        evaluate_interp_Int_2D(IndPoints, sitp,coord);                                  # this gives the full 2D/3D index
    else
        itp         =   interpolate(LinearIndices(FullGrid[1]), BSpline(Constant()));   # 2D interpolation that has indices
        sitp        =   scale(itp, Grid[1], Grid[2], Grid[3]);
        evaluate_interp_Int_3D(IndPoints, sitp,coord);                                  # this gives the full 2D/3D index
    end

    NumTracers  =   zeros(Int64,  size(Data));   # Keep track of # of tracers around every point
    TotalWeight =   zeros(Float64,size(Data)) 
    indNum      =   CartesianIndices(NumTracers);
    
    if InterpolationMethod=="DistanceWeighted"
        # In case we use a distance based weighting, compute the weight factor here
        X_g             =   FullGrid[1  ][IndPoints];   # Gridpoint to which the particle belongs 
        Z_g             =   FullGrid[end][IndPoints];   # Gridpoint to which the particle belongs 
        Weight      =   zeros(size(coord,1),1)
        if dim==2
            evaluate_weight_2D(Weight, coord, X_g, Z_g, d[1], d[end]);
        elseif dim==3
            Y_g         =   FullGrid[2  ][IndPoints];   
            evaluate_weight_3D(Weight, coord, X_g, Y_g, Z_g, d[1], d[2], d[end]);
        end
    
    elseif InterpolationMethod=="Constant"
        Weight          =   ones(length(Tracers),1); 
    else
        error("Unknown InterpolationMethod=$InterpolationMethod. Choose: [Constant] or [DistanceWeighted]. ")
    end

    if  Property=="T"
        DataTracers          =   Tracers.T;     # temperature
    elseif Property=="Phi"
        DataTracers          =   Tracers.Phi;   # solid fraction
    else
        error("Property $(Property) not yet implemented")
    end
    
    SumData(Data_local, NumTracers, TotalWeight, length(Tracers), IndPoints, DataTracers, Weight);   # sum data from ever point

    # normalize based on weight & set non-empty data pounts
    Threads.@threads for i=eachindex(Data_local)
        if TotalWeight[i]>0.0
            Data_local[i] = Data_local[i]/TotalWeight[i];
        end
        if NumTracers[i]>0
            Data[i] = Data_local[i];
        end
    end


    if RequestNumTracers
        return NumTracers
    else
        return nothing
    end

end



# define functions to speed up key calculations above
function SumData(DataLocal, NumTracers, TotalWeight, nT, IndPoints, DataTracers, Weight)
    Threads.@threads    for iT=1:nT
                                ind = IndPoints[iT];
@inbounds                       DataLocal[ind]      +=  Weight[iT]*DataTracers[iT];  
@inbounds                       TotalWeight[ind]    +=  Weight[iT];  
@inbounds                       NumTracers[ind]     +=  1;               
                        end
    end


function Add_Phase_2D(PhaseRatio, NumTracers, nT, IndPoints, indPhase, indX, iPhase, indNum, Weight)
    Threads.@threads    for iT=1:nT
                                iX = indX[IndPoints[iT]][1]
                                iZ = indX[IndPoints[iT]][2]
@inbounds                       PhaseRatio[   indPhase[iX,iZ,iPhase[iT]]]    +=  Weight[iT];  
@inbounds                       NumTracers[indNum[iX,iZ             ]]       +=  1;               
                        end
    end

function Add_Phase_3D(PhaseRatio, NumTracers, nT, IndPoints, indPhase, indX, iPhase, indNum, Weight)
        Threads.@threads    for iT=1:nT
                                    iX = indX[IndPoints[iT]][1]
                                    iY = indX[IndPoints[iT]][2]
                                    iZ = indX[IndPoints[iT]][3]
    @inbounds                       PhaseRatio[   indPhase[iX,iY,iZ,iPhase[iT]]]    +=  Weight[iT];  
    @inbounds                       NumTracers[indNum[iX,iY,iZ             ]]       +=  1;               
                            end
        end

function evaluate_weight_2D(w, coord, X_g, Z_g, dx, dz)
    Threads.@threads    for i=1:size(coord,1)
                           wx = 1.0 - abs.(coord[i,1  ] - X_g[i])/(dx/2.0);
                           wz = 1.0 - abs.(coord[i,2  ] - Z_g[i])/(dz/2.0);
        @inbounds          w[i]    = wx*wz;
                        end
    end

function evaluate_weight_3D(w, coord, X_g, Y_g, Z_g, dx, dy, dz)
        Threads.@threads    for i=1:size(coord,1)
                               wx = 1.0 - abs.(coord[i,1  ] - X_g[i])/(dx/2.0);
                               wy = 1.0 - abs.(coord[i,2  ] - Y_g[i])/(dy/2.0);
                               wz = 1.0 - abs.(coord[i,3  ] - Z_g[i])/(dz/2.0);
            @inbounds          w[i]    = wx*wy*wz;
                            end
        end
function evaluate_interp_Int_2D(s, itp, Points_irregular)
    Threads.@threads    for i=1:size(Points_irregular,1)
                            s[i]    = Int(itp(Points_irregular[i,1],Points_irregular[i,2]));
                        end
    end
    
function evaluate_interp_Int_3D(s, itp, Points_irregular)
    Threads.@threads    for i=1:size(Points_irregular,1)
                            s[i]    = Int(itp(Points_irregular[i,1],Points_irregular[i,2],Points_irregular[i,3]));
                        end
    end



"""
    CorrectBounds_Array!(Points, Grid);
    
Ensures that the coordinates of Points stay within the bounds
of the regular grid Grid. Points is an array of size [nPoints, dim]

 """
function CorrectBounds_Array!(Points, Grid);
    
    minC = [minimum(Grid[i]) for i=1:length(Grid)];
    maxC = [maximum(Grid[i]) for i=1:length(Grid)];
    
    Threads.@threads   for iT=1:size(Points,1)
        for i=1:length(Grid);
            if Points[iT,i] < minC[i]
                Points[iT, i ]  =   minC[i]; 
            end
            if Points[iT,i] > maxC[i]
                Points[iT, i ]  =   maxC[i]; 
            end
        end
    end
end

"""
        RockType = RockAssemblage(PhaseRatio)

        Computes the most abundant rock assemblage @ every point 
 
"""
function RockAssemblage(PhaseRatio);
    # Most abundant rock-type @ every point 
    dim         =   length(size(PhaseRatio))-1
    dummy       =   findmax(PhaseRatio,dims=dim+1);
    RockType    =   getindex.(dummy[2], dim+1)          # extract last of Cartesian index, which is the phase
    if dim==2
        RockType=RockType[:,:,1];
    end
    return RockType
end

"""
        CorrectTracersForTopography!(Tracers, Topo, PhaseAir=1, PhaseRock=2)

        Corrects tracers for topography, such that 'rock' tracers above the topography
            are set to "air" and vice-versa

        Input:
            Tracers     -       Tracers structure
            
            Topo        -       
                                (Topo_x, Topo_y, Topo_z)  : topography in 3D
                                (Topo_x, Topo_z)  :         topography in 2D
            PhaseAir    -       Phase of air
            PhaseRock   -       Phase of rock
 
"""
function CorrectTracersForTopography!(Tracers, Topo, PhaseAir=1, PhaseRock=2)
    # Correct Tracers for topography: 
    #       all Tracers with rock phase>1 are set to "air",
    #       all Tracesr with phase==1 is set to phase=2 ("rock")

    dim = length(Topo)
    if dim==2
        interp_linear = LinearInterpolation((Topo[1]), Topo[end],  extrapolation_bc = Line());
    elseif dim==3
        interp_linear = LinearInterpolation((Topo[1], Topo[2]), Topo[end],  extrapolation_bc = Line());
    end

    minT,maxT     = minimum(Topo[end]), maximum(Topo[end])
    
    for iT=1:length(Tracers)
        if      (Tracers[iT].coord[end]>maxT) & (Tracers.Phase[iT]>PhaseAir)
            Tracers.Phase[iT] = PhaseAir;               # air
        elseif (Tracers[iT].coord[end]<minT)  & (Tracers.Phase[iT]==PhaseAir)
            Tracers.Phase[iT] = PhaseRock;               # rock
        else
            # need to do interpolation
            pt      = Tracers[iT].coord;
            
            if dim==2
                z_topo  = interp_linear(pt[1]);
            elseif dim==3
                z_topo  = interp_linear(pt[1],pt[2]);
            end

            if      (pt[end]>z_topo) & (Tracers[iT].Phase>PhaseAir)
                Tracers.Phase[iT] = PhaseAir;          #
            elseif  (pt[end]<z_topo) & (Tracers[iT].Phase==PhaseAir)
                Tracers.Phase[iT] = PhaseRock; 
            end
 
        end
    end
  
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

"""
    update_Tvec!(Tracers::StructArray, time)

Updates temperature & time vector on every tracer  
"""
function update_Tvec!(Tracers::StructArray, time_val::Float64)

    if isassigned(Tracers,1) 
        for iT = 1:length(Tracers)
            LazyRow(Tracers, iT).time_vec = push!(LazyRow(Tracers, iT).time_vec, time_val);             
            LazyRow(Tracers, iT).T_vec    = push!(LazyRow(Tracers, iT).T_vec,     LazyRow(Tracers, iT).T);     
        end
    end

    return Tracers
end

"""
    SolidFraction computes the solid fraction (= (1-phi) as a function of T

"""
function SolidFraction(T::Array, Phi_o::Array,dt::Float64; P_bar=[], MeltFrac=[], PhaseRat=[],PhaseDiagrams=[])
    
    Phi_new =   Phi_o.*0.0;
    dPhi_dt =   Phi_o.*0.0;
    if isempty(P_bar)
        # in case we use a parameterized method only (default)
        SolidFraction_Parameterized!(T, Phi_o, Phi_new, dPhi_dt, dt);

    else
        # melt fraction is computed in a phase-wise manner
        #   which can be:
        #               "none"              -   no melting 
        #               "parameterized"     -   parameterized mekting model (aka, our default)
        #               "PD"                -   we use a phase diagram to compute the melt fraction, as specified in PhaseDiagramData
        numPhases   =   size(PhaseRat)[end];  
        dim         =   length(size(PhaseRat))-1;
        Phi_melt    =   Phi_o.*0.0;
        for iPhase  =   1:numPhases 
            Phi_melt_av = Phi_o.*0.0;
            if  MeltFrac[iPhase]=="parameterized"
                Phi_melt_av_temp = Phi_o.*0.0;
                
                # Use parameterized melting diagram
                SolidFraction_Parameterized!(T, Phi_o, Phi_melt_av_temp, dPhi_dt, dt);
                Phi_melt_av .= 1.0 .- Phi_melt_av_temp;
                
            elseif MeltFrac[iPhase]=="PD"
                # Interpolate melt fraction from the phase diagram
                interp_meltWt       =   PhaseDiagrams[iPhase].meltWt;
                
                for i=eachindex(P_bar)
                    meltWt          =   interp_meltWt(T[i],P_bar[i]);
                    
                    Phi_melt_av[i]  =  meltWt;
                end

            elseif MeltFrac[iPhase]=="none"
                # remains zero

            else
                error("Unknown melt fraction type for phase $iPhase: namely $(MeltFrac[iPhase]). Choose from: [none, PD, parameterized] ")
            end
            
            if dim==2
                Phi_melt +=  Phi_melt_av.*PhaseRat[:,:,iPhase]
            elseif dim==3
                Phi_melt +=  Phi_melt_av.*PhaseRat[:,:,:,iPhase]
            end



        end
        Phi_new   .=   1.0 .- Phi_melt;    # Phi=solid fraction
        dPhi_dt    =   (Phi_new .- Phi_o)./dt;  
        
    end
    
    return Phi_new, dPhi_dt
end


function SolidFraction_Parameterized!(T::Array, Phi_o::Array, Phi::Array, dPhi_dt::Array, dt::Float64)
   # Compute the melt fraction of the domain, assuming T=Celcius
    # Taken from L.Caricchi (pers. comm.)

    # Also compute dPhi/dt, which is used to compute latent heat

    #Theta      =   (800.0 .- T)./23.0;
    #Phi        =   1.0 .- 1.0./(1.0 .+ exp.(Theta)); 


    Phi       .=   1.0 .- 1.0./(1.0 .+ exp.((800.0 .- T)./23.0)); 
   
    dPhi_dt   .=   (Phi .- Phi_o)./dt;  
    Phi_o     .=   Phi;

end



"""
    This computes the lithostatic pressure from a given densit matrix

        General form:

        P = ComputeLithostaticPressure(Rho, Grid);

        with:
            Grid:   2D or 3D arrays that describe the 1D grid coordinates
                    2D - (x,z)
                    3D - (x,y,z)
            
            Rho:    2D or 3D matrix with density distribution

        out:
            P:      2D or 3D matrix with pressure [in bar!]
"""
function ComputeLithostaticPressure(Rho, Grid);
    # This computes the lithostatic pressure [in bar] from a given density matrix
    g   =   9.81;                   # m/s2
    z   =   Grid[end]               # z coordinates
    dim =   length(size(Rho))
    
    dz  =   z[2]-z[1];              # note that we assume a constant spacing in z
    P   =   Rho*g*dz; 
    
    if dim==2; 
        P[end,:]    .= 0.0;
    elseif dim==3
        P[end,:,:]  .= 0.0;
    end    
    P   =   reverse(P,dims=dim);        # reverse array as we go from top->bottom
    P   =   cumsum( P/1e5, dims=dim);   # sum
    P   =   reverse(P,dims=dim);        # same

    return P;
end


struct PhaseDiagram
    meltRho
    meltWt
    rockRho
    rockVp
    rockVs
    rockVpVs
    meltVp
    meltVs
    meltVpVs
end


"""
    This preloads phase diagram data from disk 

        General form:

        PhaseDiagramData = LoadPhaseDiagrams(PhaseDiagramNames, PlotDiagrams=false);

        with:
            PhaseDiagramNames:  Array with names that are either [""] or contain the name & directory of the phase diagram
            PlotDiagrams:       Plot the diagrams [warning; to be removed]

        out:
            PhaseDiagramData:   Array with interpolation objects that describe the phase diagrams

"""
function LoadPhaseDiagrams(PhaseDiagramNames, PlotDiagrams=false);
    # This pre-loads phase diagram and creates the interpolation objects to 
    #   efficiently query them @ a later stage.
    #   
    #   The phase diagram input format is the LaMEM input format, which is 
    #   to a large extend similar to what Perple_X gives as an output 
    #   (with more comments @ the beginning)

    PhaseData   =   [PhaseDiagram([],[],[],  [], [], [], [], [], []) for i=1:length(PhaseDiagramNames)];    # initialize
    i           =   0;
    for PhaseDiagramName in PhaseDiagramNames
        i   +=   1;
 
        if length(PhaseDiagramName)>0
           
            # open diagram and read info about the size of the diagram
            fid             =   open(PhaseDiagramName,"r")
            line            =   readline(fid)
            numFields       =   parse(Int64,line)                       # the # of fields encoded in the diagram
            for i=1:48; line=readline(fid);  end                        # skip to line 50, where the first "real" input starts
            Tmin            =   parse(Float64,readline(fid))-273.15     # minimum T in C
            dT              =   parse(Float64,readline(fid))            # step-size in T [C]
            numT            =   parse(Int64,readline(fid))              # number of T points
            Pmin            =   parse(Float64,readline(fid))            # minimum P in bar
            dP              =   parse(Float64,readline(fid))            # step-size in P [bar]
            numP            =   parse(Int64,readline(fid))              # number of P points
            close(fid)
            
            # Read the full data from the file
            data_diagram    =   CSV.File(PhaseDiagramName, skipto=56, header=false);    # read all diagram data
            
            Pvec            =   Pmin:dP:dP*(numP-1)+Pmin;
            Tvec            =   Tmin:dT:dT*(numT-1)+Tmin;
            meltRho         =   reshape(data_diagram.Column1, (numT,numP));              # density of melt
            meltWt          =   reshape(data_diagram.Column2, (numT,numP));              # melt fraction
            rockRho         =   reshape(data_diagram.Column3, (numT,numP));              # density of solid rock
            
        
            intp_meltRho    =   LinearInterpolation((Tvec, Pvec), meltRho,  extrapolation_bc = Line()); 
            intp_meltWt     =   LinearInterpolation((Tvec, Pvec), meltWt,   extrapolation_bc = Line()); 
            intp_rockRho    =   LinearInterpolation((Tvec, Pvec), rockRho,  extrapolation_bc = Line()); 
            PhaseData[i]    =   PhaseDiagram(intp_meltRho,intp_meltWt,intp_rockRho, [], [], [], [], [], []);

            if numFields>5
                # Seismic velocities are encoded as well
                rockVp          =   reshape(data_diagram.Column6, (numT,numP));              # Vp velocity of rock [km/s]
                rockVs          =   reshape(data_diagram.Column7, (numT,numP));              # Vs velocity of rock [km/s]
                rockVpVs        =   reshape(data_diagram.Column8, (numT,numP));              # Vp/Vs velocity of rock [-]
                
                meltVp          =   reshape(data_diagram.Column9, (numT,numP));              # Vp velocity of melt [km/s]
                meltVs          =   reshape(data_diagram.Column10,(numT,numP));              # Vs velocity of melt [km/s]
                meltVpVs        =   reshape(data_diagram.Column11,(numT,numP));              # Vp/Vs velocity of melt [-]
                
                # add interpolation objects to PhaseDiagramData struct
                intp_rockVp     =   LinearInterpolation((Tvec, Pvec), rockVp,       extrapolation_bc = Line());
                intp_rockVs     =   LinearInterpolation((Tvec, Pvec), rockVs,       extrapolation_bc = Line());
                intp_rockVpVs   =   LinearInterpolation((Tvec, Pvec), rockRho,      extrapolation_bc = Line());

                intp_meltVp     =   LinearInterpolation((Tvec, Pvec), meltVp,       extrapolation_bc = Line());
                intp_meltVs     =   LinearInterpolation((Tvec, Pvec), meltVs,       extrapolation_bc = Line());
                intp_meltVpVs   =   LinearInterpolation((Tvec, Pvec), meltVpVs,     extrapolation_bc = Line());

                # store data in a single structure
                PhaseData[i]    =   PhaseDiagram(intp_meltRho,intp_meltWt,intp_rockRho, intp_rockVp, intp_rockVs, intp_rockVpVs, intp_meltVp, intp_meltVs, intp_meltVpVs);

            elseif numFields==5
                # nothing to be done
            
            elseif numFields<5
                error("The phase diagram should at least have the collumns: meltRho | meltWt | rockRho | T [K] | P [bar] ")

            end

            # plot phase diagram if requested
            if PlotDiagrams # note that this requires the Plots package to be installed - move this to a separate routine?
                Rho = meltWt.*meltRho .+ (1.0 .- meltWt).*rockRho
                
                # density 
                p2 = heatmap(Tvec,Pvec/1e3,meltWt',  title=PhaseDiagramName,  c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="Melt fraction")
                p3 = heatmap(Tvec,Pvec/1e3,meltRho',                          c=:lajolla, xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="ρ melt [kg/m3]")
                p1 = heatmap(Tvec,Pvec/1e3,rockRho',                          c=:lajolla, xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="ρ rock [kg/m3]")
                p4 = heatmap(Tvec,Pvec/1e3,Rho',                              c=:lajolla, xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="ρ combined[kg/m3]")
                plot(p1,p2,p3,p4); png("PhaseDiagram_DensitiesMeltfraction")

                if numFields>5
                    # Seismic velocities for rock
                    p1 = heatmap(Tvec,Pvec/1e3,rockVp',                         c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="rock Vp [km/s]")
                    p2 = heatmap(Tvec,Pvec/1e3,rockVs',  title=PhaseDiagramName,c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="rock Vs [km/s]")
                    p3 = heatmap(Tvec,Pvec/1e3,rockVpVs',                       c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="rock Vp/Vs []")
                    plot(p1,p2,p3); png("PhaseDiagram_RockSeismicVelocities")

                    # Seismic velocities for melt
                    p1 = heatmap(Tvec,Pvec/1e3,meltVp',                         c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="melt Vp [km/s]")
                    p2 = heatmap(Tvec,Pvec/1e3,meltVs',  title=PhaseDiagramName,c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="melt Vs [km/s]")
                    p3 = heatmap(Tvec,Pvec/1e3,meltVpVs',                       c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="melt Vp/Vs []")
                    plot(p1,p2,p3); png("PhaseDiagram_MeltSeismicVelocities")

                    # Combined velocities
                    Vp      =  meltWt.*meltVp   .+ (1.0 .- meltWt).*rockVp;
                    Vs      =  meltWt.*meltVs   .+ (1.0 .- meltWt).*rockVs;
                    VpVs    =  meltWt.*meltVpVs .+ (1.0 .- meltWt).*rockVpVs;
                    p1 = heatmap(Tvec,Pvec/1e3,Vp',                         c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="melt Vp [km/s]")
                    p2 = heatmap(Tvec,Pvec/1e3,Vs',  title=PhaseDiagramName,c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="melt Vs [km/s]")
                    p3 = heatmap(Tvec,Pvec/1e3,VpVs',                       c=:batlow,  xlabel="T [C]",   ylabel="P [kbar]", dpi=400, fontsize=6, colorbar_title="melt Vp/Vs []")
                    plot(p1,p2,p3); png("PhaseDiagram_CombinedSeismicVelocities")
        
                end 
                
            end

        else
            PhaseData[i]    =   PhaseDiagram([],[],[],  [], [], [], [], [], []);
        end

            
    
    end

    println("Preloaded phase diagram data from disk ...")

    return PhaseData # Send back the interpolation objects 

end



"""
    This computes the density distribution and pressure in the domain

        General form:

        Rho_new, P_bar = ComputeDensityAndPressure(Rho, T, FullGrid, Grid, Tracers, ρ, PhaseDiagramData);

        with:
          
        out:
          
"""
function ComputeDensityAndPressure(Rho, T, FullGrid, Grid, Tracers, ρ, PhaseDiagramData);
    # This computes density and pressure for the domain. 
    # Nonlinear iterations are employed, as (lithostatic) pressure depends on density

    dim         =   length(Grid);

    # compute phase ratio @ every point
    PhaseRatio  =   PhaseRatioFromTracers(FullGrid, Grid, Tracers, "DistanceWeighted");       
    numPhases   =   size(PhaseRatio)[end];  

    if length(PhaseDiagramData) < numPhases
        error("The array PhaseDiagrams has less entries than the maximum detected phase")
    end

    # Compute pressure 
    P_bar_new   =   ComputeLithostaticPressure(Rho, Grid);  #
    Error       =   1.0;
    P_bar       =   P_bar_new;
    Rho_new     =   P_bar .* 0.0;
    it          =   1;
    while (Error>1e-3)  & (it<20)           # density changes P, which changes density (if using phase diagrams), which is why we use iterations
        it          +=  it;         
        P_bar       =   P_bar_new;

        Rho_new     =   P_bar .* 0.0;
        Rho_Phase   =   ones(size(Rho_new));
        for iPhase  =   1:numPhases 

            if isa(ρ[iPhase], Number)
                # We have a constant density
                Rho_Phase = Rho_Phase*0.0 .+ ρ[iPhase];
                
            elseif ρ[iPhase]=="PD"
                # we interpolate density from the phase diagram
                interp_meltRho      =   PhaseDiagramData[iPhase].meltRho;       
                interp_rockRho      =   PhaseDiagramData[iPhase].rockRho;       
                interp_meltWt       =   PhaseDiagramData[iPhase].meltWt;
                
                for i=eachindex(P_bar)
                    rockRho         =   interp_rockRho(T[i],P_bar[i]);
                    meltRho         =   interp_meltRho(T[i],P_bar[i]);
                    meltWt          =   interp_meltWt(T[i],P_bar[i]);
                    
                    Rho_Phase[i]    =   (1.0 - meltWt)*rockRho + meltRho*meltWt;
                end
            end
            
            if dim==2
                Rho_new +=  Rho_Phase.*PhaseRatio[:,:,iPhase]
            elseif dim==3
                Rho_new +=  Rho_Phase.*PhaseRatio[:,:,:,iPhase]
            end

        end

        P_bar_new     =   ComputeLithostaticPressure(Rho_new, Grid);  #

        Error = maximum( abs.(P_bar_new-P_bar) )/maximum(abs.(P_bar_new))
        #@show Error
    end

    return Rho_new, P_bar, PhaseRatio

end


"""
    This computes the average of a certain property

        General form:

            PhaseRatioAverage!(Average, prop_vec,  PhaseRatio)

        with:
            PhaseRatio - PhaseRatio matrix for the current grid  
          
"""
function PhaseRatioAverage!(Average::Array, prop_vec,  PhaseRatio)

    numPhases   =   size(PhaseRatio)[end];  
    dim         =   length(size(PhaseRatio))-1;
    
    # Catch errors
    if numPhases>length(prop_vec)
        error("you did not define sufficient properties.")
    end

    Average     =   Average.*0.0;
    for iPhase  =   1:numPhases 
        
        if      dim==2
            Average +=  prop_vec[iPhase].*PhaseRatio[:,:,iPhase]
        elseif  dim==3
            Average +=  prop_vec[iPhase].*PhaseRatio[:,:,:,iPhase]
        end

    end
    
    return Average
end


"""
    This computes seismic velocities 

        General form:

            Vp,Vs,VpVs = ComputeSeismicVelocities(Grid, T, P_bar, PhaseRatio, PhaseDiagramData)

        with:
            
"""
function ComputeSeismicVelocities(Grid, T, P_bar, PhaseRatio, PhaseDiagramData)
    # This computes seismic velocities on the given grid, taking melt fraction into account
    # This requires phase diagrams that list Vp,Vs etc. as a function of P and T 
    
    dim         =   length(Grid);
    Vp          =   zeros(size(T));
    Vs          =   zeros(size(T));
    VpVs        =   zeros(size(T));
    
    numPhases   =   size(PhaseRatio)[end]; 
    for iPhase  =   1:numPhases 
    
            Vp_av               =   zeros(size(T));
            Vs_av               =   zeros(size(T));
            VpVs_av             =   zeros(size(T));
            
            interp_meltVp       =   PhaseDiagramData[iPhase].meltVp;       
            interp_meltVs       =   PhaseDiagramData[iPhase].meltVs;       
            interp_meltVpVs     =   PhaseDiagramData[iPhase].meltVpVs;       
            
            interp_rockVp       =   PhaseDiagramData[iPhase].rockVp;       
            interp_rockVs       =   PhaseDiagramData[iPhase].rockVs;   
            interp_rockVpVs     =   PhaseDiagramData[iPhase].rockVpVs;       

            interp_meltWt       =   PhaseDiagramData[iPhase].meltWt;

            if !isempty(interp_meltVp)
                for i=eachindex(P_bar)
                    Phi_melt        =   interp_meltWt(  T[i],   P_bar[i]);

                    meltVp          =   interp_meltVp(  T[i],   P_bar[i]);
                    rockVp          =   interp_rockVp(  T[i],   P_bar[i]);
                    
                    meltVs          =   interp_meltVs(  T[i],   P_bar[i]);
                    rockVs          =   interp_rockVs(  T[i],   P_bar[i]);
                    
                    rockVpVs        =   interp_rockVpVs(T[i],   P_bar[i]);
                    meltVpVs        =   interp_meltVpVs(T[i],   P_bar[i]);

                    Vp_av[i]        =   rockVp*(1.0 - Phi_melt)     + meltVp*Phi_melt;
                    Vs_av[i]        =   rockVs*(1.0 - Phi_melt)     + meltVs*Phi_melt;
                    VpVs_av[i]      =   rockVpVs*(1.0 - Phi_melt)   + meltVpVs*Phi_melt;
                    
                end

            end

            if dim==2
                Vp      += Vp_av  .*PhaseRatio[:,:,  iPhase];
                Vs      += Vs_av  .*PhaseRatio[:,:,  iPhase];
                VpVs    += VpVs_av.*PhaseRatio[:,:,  iPhase];
            else
                Vp      += Vp_av  .*PhaseRatio[:,:,:,iPhase];
                Vs      += Vs_av  .*PhaseRatio[:,:,:,iPhase];
                VpVs    += VpVs_av.*PhaseRatio[:,:,:,iPhase];
            end
    end

    return Vp,Vs,VpVs

end
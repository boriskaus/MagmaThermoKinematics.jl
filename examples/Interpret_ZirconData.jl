using JLD2, MagmaThermoKinematics, MAT, CSV, Tables, Statistics, StatsBase
using CairoMakie

#dirname = "ZASSy_Geneva_isoT_variable_k_run8_withlatent_depth_smooth_dt0_2_Geotherm30_Flux9_1e-6"
#dirname = "ZASSy_Geneva_9_1e_6_reference"

#dirname = "ZASSy_Geneva_14_9e_6_v2"
dirname = "ZASSy_UCLA_10_7e_6_v2"

#dirname = "ZASSy_Geneva_10_7e_6_reference_DikeTracers300"


filename = dirname*"/Tracers_SimParams.jld2";
Time_vec,Melt_Time,Tav_magma_Time, Tav_3D_magma_Time, Tracers, Tracers_grid, Tnew, Phases = JLD2.load(filename, "Time_vec", "Melt_Time", "Tav_magma_Time","Tav_3D_magma_Time","Tracers","Tracers_grid", "Tnew_cpu","Phases_float")
@show dirname

useTracersOnGrid = true

Write_R_input = true

#SecYear = 3600*24*365.25;
if Write_R_input

    # Prepare the input for the R-script of Gregor Weber 
    if !useTracersOnGrid
        # Advected tracers
        time_vec     = Tracers.time_vec*1e6*SecYear;
        T_vec        = Tracers.T_vec;
        coord        = Tracers.coord;
        Phi          = Tracers.Phi; 
        T            = Tracers.T;
    else
        
        # The ones that stayed on the grid points
        time_vec     = Tracers_grid.time_vec*1e6*SecYear;
        T_vec        = Tracers_grid.T_vec;
        coord        = Tracers_grid.coord;
        Phi          = Tracers_grid.Phi; 
        T            = Tracers_grid.T;
    end
    
    # Sample tracers, based on their probability 
    # This is done, as the axisymmetric geometry should be expanded to a 3D volume from which melt would be extracted.
    # This implies that the sampling probability scales with 2πR, with 
    Prob_Tracers     = [coord[i][1] for i = 1:length(coord)]; Prob_Tracers = Prob_Tracers./maximum(Prob_Tracers)

    if 1==1
        # All Tracers that have final T>700

        ind      = [ T_vec[i][end]>700 for i=1:length(T_vec) ];     # Tracers that are still molten @ the end
        
        # All Tracers that are still partially molten at the end
        #ind         = findall(Phi .> 0.0)
        
        time_vec     = time_vec[ind]
        T_vec        = T_vec[ind]
        Prob_Tracers = Prob_Tracers[ind]
        T = T[ind]
    end

    SampleTracers = true
    nSample = 5000
    if SampleTracers
        # Sample the tracers with the probability that they are being extracted:
        id = 1:length(Prob_Tracers)
        wt = ProbabilityWeights(Prob_Tracers)

        indSample = zeros(Int64,nSample)
        for i=1:nSample
            indSample[i] = sample(id,wt)
        end

        time_vec     = time_vec[indSample]
        T_vec        = T_vec[indSample]
        Prob_Tracers = Prob_Tracers[indSample]
    end

    if 1==1
        # Only consider the most recent time the tracer was molten
        for iT = 1:nSample
            ind             = findall(T_vec[iT] .< 700) # 
            if length(ind)>0

                T_vec[iT]       = T_vec[iT][maximum(ind):end]
                time_vec[iT]    = time_vec[iT][maximum(ind):end]
            end
        end
    end
  
    if 1==1
        filename_small = dirname*"/Tracers_SimParams_small.jld2";

        # Save it into a different (smaller) file
        jldsave(filename_small; Time_vec, Melt_Time, Tav_magma_Time, time_vec, T_vec,Prob_Tracers)


        #load with:
       # Time_vec,Melt_Time,Tav_magma_Time, time_vec, T_vec, Prob_Tracers = JLD2.load(filename_small, "Time_vec", "Melt_Time", "Tav_magma_Time","time_vec","T_vec","Prob_Tracers")

    end

    # Convert T-t path as text-file input to R-script:
    time_sec, Tt_paths_Temp = compute_zircons_convert_vecs2mat(time_vec, T_vec);

    if useTracersOnGrid
        Tracer_str = "TracersOnGrid";
    else
        Tracer_str = "Tracers";
    end


    time_sec    = range(round(time_sec[1]),round(time_sec[end]),length(time_sec));       # avoids round-off errors
    
    step        = 1;
    Data_TXT                = zeros(length(time_sec), length(1:step:size(Tt_paths_Temp,2))+1);

    Data_TXT[:,1]           = time_sec;

    Data_TXT[:,2:end]       = Tt_paths_Temp[:,1:step:end];

    CSV.write("$(dirname)/$(dirname)_$(Tracer_str).txt", Tables.table(Data_TXT),writeheader=false)

    # Compute average T of points that have T>0 
    #  note that the likelihood that something is sampled in 3D is already taken care off above
    T_average   = [ mean(Tt_paths_Temp[i,Tt_paths_Temp[i,:].>0]) for i=1:size(Tt_paths_Temp,1) ]
    time_years  = time_sec/SecYear
    time_Ma     = (time_years[end] .- time_years)/1e6

    # save to CSV file
    ArrayOut = zeros(length(time_Ma),4);
    ArrayOut[:,1] = time_Ma;
    ArrayOut[:,2] = T_average;
    CSV.write("$(dirname)/$(dirname)_$(Tracer_str)_Taverage_julia.csv", Tables.table(ArrayOut),writeheader=false)

    #Plot average T

    fig = Figure(fontsize = 25, resolution = (800,800))
    ax  = Axis(fig[1, 1], 
        xlabel="Age [ka]",
        ylabel="T_average magma [ᵒC]")
    lines!(ax,time_Ma*1000,T_average, linewidth=0.5)
    #limits!(ax,0,1500,600,1000)
    save("$(dirname)/$(dirname)_Taverage_$(Tracer_str).png", fig)


end

# use our julia implementation to compute the zircon age distribution
Compute_with_julia = true
if Compute_with_julia
    # perform the same zircon age calculations but using julia

    if 1==0   
        filename_small = dirname*"/Tracers_SimParams_small.jld2";
        # Load 'small' file
        Time_vec, Melt_Time, Tav_magma_Time,time_vec,T_vec,Prob_Tracers = JLD2.load(filename_small, "Time_vec", "Melt_Time", "Tav_magma_Time","time_vec","T_vec","Prob_Tracers")
    end 

    ZirconData  	=   ZirconAgeData(Tsat=820, Tmin=700, Tsol=700, Tcal_max=800, Tcal_step=1.0, max_x_zr=0.001, zircon_number=100);	 # data as used in the R-script of Gregor   
    
    time_years, prob, ages_eruptible, number_zircons, T_av_time, T_sd_time = compute_zircons_Ttpath(time_vec/SecYear, T_vec, ZirconData=ZirconData)
    
    zircon_cumulativePDF = (1.0 .- cumsum(prob))*100;

    # The large numbers in seconds can cause roundoff errors: make sure it is equally spaced again:    
    #time_yrs = range(round(time_sec[1]/SecYear),round(time_sec[end]/SecYear),length(time_sec));
    #time_years, prob, ages_eruptible, number_zircons, T_av_time, T_sd_time, zircon_cumulativePDF = compute_zircons_Ttpath(Vector(time_yrs), Tt_paths_Temp, ZirconData=ZirconData)
    
    ArrayOut = zeros(length(time_sec),5);

    ArrayOut[:,1] = time_sec/SecYear;
    ArrayOut[:,2] = T_av_time;
    ArrayOut[:,3] = number_zircons;
    ArrayOut[:,4] = prob;
    ArrayOut[:,5] = zircon_cumulativePDF;
    
    # save output to the same directory
    CSV.write("$(dirname)/ZirconPDF_Tav_$(Tracer_str)_$(dirname)_julia.csv", Tables.table(ArrayOut),writeheader=true,header=["time[yrs]","T_average[C]","#zircons","probability","cumulativePDF"])
    time_Ma_vec = (time_years[end] .- time_years)/1e6;

    fig = Figure(fontsize = 25, resolution = (800,1600))
    ax2  = Axis(fig[1, 1], 
    xlabel="Age [ka]",
    ylabel="cumulative PDF ")
    lines!(ax2,time_Ma_vec*1e3,zircon_cumulativePDF[:], linewidth=2)



    limits!(ax2,0,1500,0,100)

    ax  = Axis(fig[2, 1], 
        xlabel="Age [ka]",
        ylabel="T_average magma [ᵒC]")
    lines!(ax,time_Ma*1000,T_average, linewidth=0.5, label="from tracers")

    Time_Vec_ka =  (Time_vec[end] .- Time_vec[:])/SecYear/1e3
    lines!(ax,Time_Vec_ka,Tav_3D_magma_Time[:], linewidth=0.5, label="from grid")
    
    axislegend(ax, position=:lt,labelsize=15)


    limits!(ax,0,1500,600,1000)
    
    save("$(dirname)/Tav_Zircon_cumPDF_$(dirname)_$(Tracer_str).png", fig)

    # Save data as CSV files 
    ArrayOut = zeros(length(time_sec),2);
    ArrayOut[:,1] = time_sec/SecYear;
    ArrayOut[:,2] = T_av_time;
    CSV.write("$(dirname)/$(dirname)_$(Tracer_str)_julia_Taverage.csv", Tables.table(ArrayOut),writeheader=true,header=["time[yrs]","T_average_all_above700[C]"])

    
    ArrayOut = zeros(length(time_sec),2);
    ArrayOut[:,1] = time_sec/SecYear;
    ArrayOut[:,2] = number_zircons;
    CSV.write("$(dirname)/$(dirname)_$(Tracer_str)_julia_ZirconAges.csv", Tables.table(ArrayOut),writeheader=true,header=["time[yrs]","#zircons"])


end



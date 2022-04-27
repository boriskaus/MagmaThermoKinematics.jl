
 using Statistics: mean

"""
    Process_ZirconAges(dirname; ZirconData=ZirconAgeData()) 

Performs postprocessing which computes zircon ages from given Tt-paths of a simulation and saves the result in a JLD2 file

Example
======
```julia
julia> using MagmaThermoKinematics, MagmaThermoKinematics.GeoParams
julia> ZirconData = ZirconAgeData(time_zr_growth=1e5);
julia> dirname = "Zassy_Geneva_zeroFlux_variable_k_2";
julia> Process_ZirconAges(dirname, ZirconData=ZirconData )
Save processed zircon age data to file: Zassy_Geneva_zeroFlux_variable_k_2/ZirconAges.jld2
```
Once this is done, you can load the data with
```julia
julia> using MagmaThermoKinematics.JLD2
julia> Age_Ma, cum_PDF, norm_PDF, T_av_time  = JLD2.load("Zassy_Geneva_zeroFlux_variable_k_2/ZirconAges.jld2","Age_Ma","cum_PDF","prob","T_av_time")
```

"""
function Process_ZirconAges(dirname; ZirconData=ZirconAgeData())
    filename = dirname*"/Tracers_SimParams.jld2"
    Tracers, Tav_magma_Time, Time_vec = JLD2.load(filename,"Tracers","Tav_magma_Time","Time_vec")

    # Compute average T in magma (that is all tracers with T>700)
    time_years, Ttpaths_mat = compute_zircons_convert_vecs2mat(Tracers.time_vec, Tracers.T_vec)
    Ttpaths_mat[Ttpaths_mat.<700] .= 0;
    T_average_magma_time = zero(time_years)
    for i = 1:length(time_years)
        T_vec = Ttpaths_mat[i,:];
        T_average_magma_time[i] = mean(T_vec[T_vec .> 10])
    end

    # Compute zircon ages
    time_Ma, PDF_zircons, time_Ma_average, PDF_zircon_average, time_years, prob, ages_eruptible, number_zircons, T_av_time, T_sd_time = 
        compute_zircon_age_PDF(Tracers.time_vec*1e6, Tracers.T_vec; ZirconData = ZirconData, bandwidth=1e5, n_analyses=300);




    # compute useful vectors (for plotting): 
    Age_Ma = time_years[end:-1:1]/1e6;   # age of the rocks (opposite of model time)
    cum_PDF  = 1.0 .- cumsum(prob);      # cumulative probability density function from 0-1
    norm_PDF = prob;                     # normal probability density function from 0-1

    # save stuff as jld2 file (easier to plot later)
    filename_save = dirname*"/ZirconAges.jld2"
    jldsave(filename_save; time_Ma, PDF_zircons, time_Ma_average, PDF_zircon_average, 
                            time_years, prob, ages_eruptible, number_zircons, T_av_time, T_sd_time, Tav_magma_Time, 
                            Age_Ma, cum_PDF, norm_PDF, Time_vec, T_average_magma_time)
    println("Save processed zircon age data to file: $filename_save")

    return nothing
end

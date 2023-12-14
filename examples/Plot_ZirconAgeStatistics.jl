using GLMakie
using JLD2

# Specify directory names
dirnames = ["Zassy_UCLA_ellipticalIntrusion_constant_k_radioactiveheating",
            "Zassy_Geneva_zeroFlux_variable_k_4thordermelt",
            "Zassy_Geneva_zeroFlux_variable_k_CaricchiMelting"]

col = [:red,:blue,:green]
leg = ["central injection","sill underaccretion","sill underaccretion Caricchi Melting"]

# Create figure
fig = Figure(resolution = (2000,1000))
ax1=Axis(fig[1, 1],xlabel = "Age [Ma]", ylabel = "T_{average} magma")
xlims!(ax1, 0, 1.5)    
ylims!(ax1, 400, 1000)    

ax2=Axis(fig[1, 2],xlabel = "Age [Ma]", ylabel = "Zircon age cummulative probability %")
limits!(ax2, 0, 1.5, 0,1)

for i = 1:length(dirnames) 

    # Load file
    filename = dirnames[i]*"/ZirconAges.jld2";
    Age_Ma,T_av_time,cum_PDF,norm_PDF = JLD2.load(filename, "Age_Ma","T_av_time","cum_PDF","norm_PDF");

    # Plot T
    lines!(ax1,Age_Ma, T_av_time, color=col[i])

     # Plot PDF
     lines!(ax2,Age_Ma, cum_PDF, color=col[i], label=leg[i])

end
axislegend(position=:rb)

fig

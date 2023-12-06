# This creates a plot of the MTK result vs the Geneva result run by Gregor Weber
#
#
using MAT
using CairoMakie
#using GLMakie
using DelimitedFiles

#SimName_UCLA_final  = "UCLA_data/isot1_01_000.dat"
#SimName_UCLA_final  = "UCLA_data/isot1_03_075.dat"
SimName_UCLA_final  = "UCLA_data/isot1_05_113.dat"
SimName_UCLA_inter  = "UCLA_data/isot1_02_050.dat"


#SimName             =   "Zassy_UCLA_ellipticalIntrusion"     # name of simulation
#SimName             =   "Zassy_UCLA_ellipticalIntrusion_2D"     # name of simulation
#SimName             =   "Zassy_UCLA_ellipticalIntrusion_constant_k"     # name of simulation
SimName             = "Zassy_UCLA_ellipticalIntrusion_variable_k"
#SimName             =   "Zassy_UCLA_ellipticalIntrusion_constantkcp"     # name of simulation
#SimName = "Zassy_UCLA_ellipticalIntrusion_variable_k_radioactiveheating"
SimName = "Zassy_UCLA_ellipticalIntrusion_constant_k_radioactiveheating"

#it_final            =   28800 
it_final            =   43584 
it_inter            =   19200
#title_str           =   "isothermal lower BC, UCLA setup, k=3.35W/m/K"
#save_str            =   "Zassy_UCLA_ellipticalIntrusion_constant_k"

#title_str           =   "flux lower BC with qm=76mW/m^2, UCLA setup, k=k(T), no radioactive heating"
#save_str            =   "Zassy_UCLA_ellipticalIntrusion_variable_k"

#title_str           =   "UCLA setup, qm_bottom=76mW/m2, k=k(T), radioactive heating, H0=1e-6W/m3"
#save_str            =   "Zassy_UCLA_ellipticalIntrusion_variable_k_radioactiveheating"

title_str           =   "UCLA setup, qm_bottom=134mW/m2, k=3.35W/K/m, radioactive heating: H0=1e-6W/m3, hr=10km"
save_str            =   "ZASSy_UCLA_ellipticalIntrusion_constant_k_radioactiveheating"


# Load Oscar's final results:
T_full    =  readdlm(SimName_UCLA_final)
T_o_final =  T_full[:,301:end]';
z_o_ref   =  Vector(0:-(20e3/(size(T_o_final,2)-1)):-20e3)
x_o_ref   =  Vector(0:(30e3/(size(T_o_final,1)-1)):30e3)

T_full    =  readdlm(SimName_UCLA_inter)
T_o_inter =  T_full[:,301:end]';

# Load MTK results
T_MTK  = matopen("$(SimName)/$(SimName)_$(it_final).mat");
    Tnew_final  =  read(T_MTK, "Tnew")
    #Tnew_final  =  read(T_MTK, "T")
    x           =  read(T_MTK, "x")
    z           =  read(T_MTK, "z")
    time_final  =  read(T_MTK, "time")
    dike_poly   =  read(T_MTK, "dike_poly")
close(T_MTK)


T_MTK  = matopen("$(SimName)/$(SimName)_$it_inter.mat");
    Tnew_inter  =  read(T_MTK, "Tnew")
    x           =  read(T_MTK, "x")
    z           =  read(T_MTK, "z")
    time_inter  =  read(T_MTK, "time")
close(T_MTK)
 
T_lin = -(801.119-25)/20e3*z .+ 25


fig = Figure(resolution = (2400,800))


# 1D figure with cross-sections
SecYear = 3600*24*365.25

# 2D temperature plot UCLA
ax2=Axis(fig[1, 1],xlabel = "Width [km]", ylabel = "Depth [km]", title = "UCLA model")
Tcon = T_o_final;

Tcon[Tcon.>= 999.9] .= 999.9
co = contourf!(fig[1, 1], x_o_ref/1e3, z_o_ref/1e3, Tcon, levels = 0:50:1000,colormap = :jet)
if maximum(T_o_final)>691
    co1 = contour!(fig[1, 1], x_o_ref/1e3, z_o_ref/1e3, Tcon, levels = 690:691, color=:black)       # solidus
end
limits!(ax2, 0, 30, -20, 0)

# 2D temperature plot MTK
time_Myrs_rnd       = round(time_final/SecYear/1e6,digits=3)
time_Myrs_int_rnd   = round(time_inter/SecYear/1e6,digits=3)

ax2=Axis(fig[1, 2],xlabel = "Width [km]",  title = "MagmaThermoKinematics.jl, t=$time_Myrs_rnd Myrs")
co = contourf!(fig[1, 2], x/1e3, z/1e3, Tnew_final, levels = 0:50:1000,colormap = :jet)
if maximum(Tnew_final)>691
    co1 = contour!(fig[1, 2], x/1e3, z/1e3, Tnew_final, levels = 690:691, color=:black)       # solidus
end
pl = lines!(fig[1, 2], dike_poly[1]/1e3, dike_poly[2]/1e3,   color = :black, linestyle=:dot, linewidth=1.5)
limits!(ax2, 0, 30, -20, 0)
Colorbar(fig[1, 3], co, label = "Temperature [ᵒC]")



ax3 = Axis(fig[1,4], xlabel = "Temperature [ᵒC]", ylabel = "Depth [km]", title = "$title_str")
#lines!(fig[1,4],T_lin,z/1e3,label="t=0",  color=:black)

lines!(fig[1,4],Tnew_inter[1,:],  z/1e3,label="Center, MTK, t=$time_Myrs_int_rnd", color=:orange)
lines!(fig[1,4],T_o_inter[1,:],  z_o_ref/1e3,label="Center, UCLA",linestyle=:dot, color=:orange)
lines!(fig[1,4],Tnew_inter[end,:],z/1e3,label="Side, MTK, t=$time_Myrs_int_rnd", color=:green)
lines!(fig[1,4],T_o_inter[end,:],z_o_ref/1e3,label="Side, UCLA",linestyle=:dot, color=:green)

lines!(fig[1,4],Tnew_final[1,:],  z/1e3,label="Center, MTK, t=$time_Myrs_rnd", color=:red)
lines!(fig[1,4],T_o_final[1,:],  z_o_ref/1e3,label="Center, UCLA",linestyle=:dot, color=:red)
lines!(fig[1,4],Tnew_final[end,:],z/1e3,label="Side, MTK, t=$time_Myrs_rnd", color=:blue)
lines!(fig[1,4],T_o_final[end,:],z_o_ref/1e3,label="Side, UCLA",linestyle=:dot, color=:blue)



axislegend(ax3, position = :lb)
limits!(ax3, 0, 1100, -20, 0)

#save("Comparison_Geneva_MTK_zero_flux_$(time_Myrs_rnd).png", fig)
#save("Comparison_Geneva_MTK_isothermal_$(time_Myrs_rnd).png", fig)

save("$(save_str)_$(time_Myrs_rnd)Myrs.png", fig)

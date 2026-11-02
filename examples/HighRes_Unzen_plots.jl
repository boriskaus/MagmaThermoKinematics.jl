# Movie and zircon ages of the high-resolution 2D showcase (HighRes_Unzen_2D.jl).
#
# Usage:  julia --project HighRes_Unzen_plots.jl HighRes2D_50m          (the output directory of the run)
# Writes <dir>/movie.mp4, <dir>/movie.gif, <dir>/final_frame.png and <dir>/zircon_ages.png (age PDF, CDF and the eruptions).
using CairoMakie, JLD2, Printf, Statistics
using MagmaThermoKinematics, ZirconGrowth
dir = isempty(ARGS) ? "HighRes2D_100m" : ARGS[1]
snaps = sort(filter(f -> startswith(f, "snap_"), readdir(dir)))
set_theme!(fonts = (regular = "Liberation Sans", bold = "Liberation Sans Bold"), fontsize = 16)

# --- Movie: temperature, melt, deformed layering, surface and eruptions -----------------------------------------------
s0 = load(joinpath(dir, snaps[1])); x, z = s0["x"] ./ 1.0e3, s0["z"] ./ 1.0e3
last = load(joinpath(dir, snaps[end])); ev_all = last["events"]
i_snap = Observable(1)
snap = @lift load(joinpath(dir, snaps[$i_snap]))
air(s) = s["Phases"] .== 0
Tplot = @lift (T = Float32.($snap["T"]); T[air($snap)] .= NaN; T)
ϕplot = @lift (ϕ = Float32.($snap["ϕ"]) ./ 255; ϕ[air($snap)] .= 0; ϕ)          # stored as 0–255
band(z0) = mod(z0 > -10.0e3 ? floor(z0 / 1.0e3) : floor(z0 / 2.0e3), 2)                  # layers of 1 km in the upper 10 km, 2 km below
bands = @lift (B = Float32.(band.($snap["Z0"])); B[air($snap)] .= NaN; B)
sill(s) = s["Phases"] .== 4                                                                 # intruded magma (phase 4)
old_sills = @lift (S = Float32.(sill($snap)); S[S .== 0] .= NaN; S)
times = [load(joinpath(dir, f), "time_kyr") for f in snaps]
SILLS = isfile(joinpath(dir, "sills.jld2")) ? load(joinpath(dir, "sills.jld2"), "sills") : NamedTuple[]
new_sills = @lift [Point2f(q.x + f * q.R * cosd(q.angle), q.z - f * q.R * sind(q.angle)) ./ 1.0f3 for q in SILLS if times[max($i_snap - 1, 1)] < q.time_kyr <= times[$i_snap] for f in (-1, 1)]   # intruded since the previous frame (the angle is clockwise)
per_kyr(ev) = (c = 0.5:1.0:149.5; (c, [sum((e.V_km3 for e in ev if abs(-e.time_kyr - ci) <= 0.5); init = 0.0) for ci in c]))   # erupted volume in bins of 1 kyr
fig = Figure(size = (1600, 900))
title = @lift(@sprintf("%.1f kyr before present", abs($snap["time_kyr"])))
ax = Axis(fig[1, 1], xlabel = "Distance W–E [km]", ylabel = "Height [km]", title = title, aspect = DataAspect())
axz = Axis(fig[1, 2], xlabel = "Distance W–E [km]", title = "Shallow intrusion zone", aspect = DataAspect())
const ZOOM = ((0.5, 9.5), (-8.0, -2.0))                                               # [km]
for a in (ax, axz)
    global hm = heatmap!(a, x, z, Tplot, colormap = :lajolla, colorrange = (0, 1000))
    heatmap!(a, x, z, bands, colormap = [(:white, 0.0), (:white, 0.12)], colorrange = (0, 1))    # the layering of the crust, deformed by sills and eruptions
    heatmap!(a, x, z, old_sills, colormap = [(:deepskyblue, 0.2), (:deepskyblue, 0.2)])          # older sills and dikes, faint so the temperature shows through
    linesegments!(a, new_sills, color = :cyan, linewidth = a === ax ? 1.5 : 2.5)
    contour!(a, x, z, ϕplot, levels = [0.5], color = :white, linewidth = 2)
    lines!(a, x, @lift($snap["Zsurf"] ./ 1.0e3), color = :black, linewidth = 2)
end
ylims!(ax, -20, 4); limits!(axz, ZOOM...)
poly!(ax, Rect(ZOOM[1][1], ZOOM[2][1], ZOOM[1][2] - ZOOM[1][1], ZOOM[2][2] - ZOOM[2][1]), color = :transparent, strokecolor = :white, strokewidth = 1.5, linestyle = :dash)
Colorbar(fig[1, 3], hm, label = "Temperature [°C]")
axe = Axis(fig[2, 1], xlabel = "Time before present [kyr]", ylabel = "Erupted [km³/kyr]", height = 180)
tb, vb = per_kyr(ev_all)
xlims!(axe, 150, 0); ylims!(axe, 0, 1.1 * maximum(vb; init = 0.1))
barplot!(axe, tb, @lift(ifelse.(tb .>= -$snap["time_kyr"], vb, 0.0)), gap = 0, color = :firebrick)
vlines!(axe, @lift(-$snap["time_kyr"]), color = :gray)
Legend(
    fig[2, 2:3], [LineElement(color = :cyan, linewidth = 2.5), PolyElement(color = (:deepskyblue, 0.4)), LineElement(color = :white, linewidth = 2), PolyElement(color = (:gray, 0.2)), LineElement(color = :black, linewidth = 2)],
    ["new sills and dikes (since the last frame)", "older sills and dikes", "melt fraction 0.5", "rock layers (initially 1 km, 2 km below 10 km)", "surface"], framecolor = :transparent, backgroundcolor = (:gray, 0.35)
)
colsize!(fig.layout, 2, Relative(0.3))
record(fig, joinpath(dir, "movie.mp4"), eachindex(snaps); framerate = 15) do i
    i_snap[] = i
end
ffmpeg = CairoMakie.Makie.FFMPEG_jll.ffmpeg()                                         # gif for the documentation: 1000 px wide, shared palette
filt = "fps=10,scale=1000:-1:flags=lanczos"
run(`$ffmpeg -y -loglevel error -i $(joinpath(dir, "movie.mp4")) -vf "$filt,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" -loop 0 $(joinpath(dir, "movie.gif"))`)
save(joinpath(dir, "final_frame.png"), fig)

# --- Zircon ages from the temperature–time paths of the tracers, and the eruptions ------------------------------------
isfile(joinpath(dir, "final.jld2")) || (println("no final.jld2 yet: movie only"); exit())
fin = load(joinpath(dir, "final.jld2"))
ages, _ = simulate_zircon_growth_from_tracers(fin["tracers"])
a = sort(filter(isfinite, ages) ./ 1.0e3)                                           # [kyr]
h = 1.06 * std(a) * length(a)^(-1 / 5)                                              # bandwidth of the Gaussian kernel density (Silverman)
t = range(0, 150, length = 601)
pdf = [sum(exp(-0.5 * ((ti - ai) / h)^2) for ai in a) / (length(a) * h * sqrt(2π)) for ti in t]
f2 = Figure(size = (1500, 450))
a1 = Axis(f2[1, 1], xlabel = "Zircon age [kyr]", ylabel = "Probability density [1/kyr]", title = "Zircon ages ($(length(a)) zircons)")
band!(a1, t, zeros(length(t)), pdf, color = t, colormap = :viridis, colorrange = (0, 150))
lines!(a1, t, pdf, color = :black)
a2 = Axis(f2[1, 2], xlabel = "Zircon age [kyr]", ylabel = "Cumulative fraction", title = "Cumulative distribution")
lines!(a2, a, (1:length(a)) ./ length(a), color = :black, linewidth = 2)
ev = fin["events"]; te = [-e.time_kyr for e in ev]; ve = [e.V_km3 for e in ev]
a3 = Axis(f2[1, 3], xlabel = "Time before present [kyr]", ylabel = "Erupted [km³/kyr]", title = "Eruptions ($(length(ev)))")
barplot!(a3, per_kyr(ev)..., gap = 0, color = :firebrick)
a3r = Axis(f2[1, 3], ylabel = "Cumulative erupted volume [km³]", yaxisposition = :right, ygridvisible = false, xgridvisible = false)
hidespines!(a3r); hidexdecorations!(a3r)
o = sortperm(te, rev = true); lines!(a3r, te[o], cumsum(ve[o]), color = :black, linewidth = 2)
for ax in (a1, a2, a3, a3r)
    xlims!(ax, 150, 0)
end
save(joinpath(dir, "zircon_ages.png"), f2)
println("written to $dir")

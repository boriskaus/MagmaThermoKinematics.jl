# Summary figure of the high-resolution 2D and 3D runs below Unzen (HighRes_Unzen_2D.jl, HighRes_Unzen_3D.jl), e.g. for a paper:
# the temperature at present in 2D and 3D, and the eruptions and zircon ages of the 3D run.
#
# Usage:  PVBATCH=/path/to/pvbatch julia -t auto --project HighRes_Unzen_figure.jl HighRes2D_2m HighRes3D_50m [HighRes_Unzen.png]
# TMAP sets the colour map of the temperature (default lajolla, one of the scientific colour maps of F. Crameri; e.g. TMAP=lipari).
# The 3D panel is rendered by ParaView (HighRes_Unzen_3D_Paraview.py) from the final state of the 3D run at its full resolution; the
# zircon ages are computed from the tracers once and cached in <dir>/zircon_ages.jld2.
using CairoMakie, JLD2, Printf, Statistics, WriteVTK
using CairoMakie.Makie.Colors: red, green, blue
using MagmaThermoKinematics, ZirconGrowth
dir2, dir3 = ARGS[1], ARGS[2]
out = length(ARGS) > 2 ? ARGS[3] : "HighRes_Unzen.png"
set_theme!(fonts = (regular = "Liberation Sans", bold = "Liberation Sans Bold"), fontsize = 20)
const C_ERUPT = :firebrick
const TMAP = Symbol(get(ENV, "TMAP", "lajolla"))                                      # temperature colour map
const SILL = (:deepskyblue, 0.2)                                                 # the intrusive complex (all sills and dikes)
const T_NEW = 1.0                                                                   # sills and dikes younger than this are drawn as lines [kyr]
const MELT = cgrad([RGBf(1.0, 0.95, 0.6), RGBf(1.0, 0.6, 0.1), RGBf(0.9, 0.2, 0.05), RGBf(0.55, 0.0, 0.05)], [0.0, 0.3, 0.6, 1.0])   # as in the 3D movie
const LAYERS = [-1.0e3 .* (1:10); -1.0e3 .* (12:2:18)]                            # lines of equal initial height [m], as in the 3D movie

function zircon_ages(dir)                                                           # zircon ages [kyr], cached
    f = joinpath(dir, "zircon_ages.jld2")
    isfile(f) || jldsave(f; ages_yr = first(simulate_zircon_growth_from_tracers(load(joinpath(dir, "final.jld2"), "tracers"))))
    return filter(isfinite, load(f, "ages_yr")) ./ 1.0e3
end

function render_3D(dir)                                                             # ParaView image of the final 3D state at full resolution
    png = joinpath(dir, "final_render_$TMAP", "frame_0000.png")
    isfile(png) && return png
    vdir = mkpath(joinpath(dir, "vtk_final"))
    isfile(joinpath(vdir, "fields.pvd")) || write_vtk(load(joinpath(dir, "final.jld2")), vdir)
    cmap = joinpath(vdir, "$TMAP.txt")                                            # the same colours in ParaView
    write(cmap, join((@sprintf("%.4f %.4f %.4f", red(c), green(c), blue(c)) for c in cgrad(TMAP).colors), "\n"))
    run(`$(ENV["PVBATCH"]) --force-offscreen-rendering $(joinpath(@__DIR__, "HighRes_Unzen_3D_Paraview.py")) $vdir $(dirname(png)) 2.5 $cmap`)
    return png
end

function write_vtk(f, vdir)                                                          # the final state as one time step of fields.pvd and surface.pvd
    x, y, z = f["x"] ./ 1.0e3, f["y"] ./ 1.0e3, f["z"] ./ 1.0e3
    fields, surface = paraview_collection(joinpath(vdir, "fields")), paraview_collection(joinpath(vdir, "surface"))
    vtk_grid(joinpath(vdir, "fields_1"), x, y, z) do vtk
        vtk["Temperature"] = f["T"]; vtk["MeltFraction"] = f["ϕ"]; vtk["Phases"] = Float32.(f["Phases"]); vtk["InitialDepth"] = f["Z0"] ./ 1.0f3
        fields[0.0] = vtk
    end
    Zs = f["Zsurf"] ./ 1.0e3
    vtk_grid(joinpath(vdir, "surface_1"), [xx for xx in x, yy in y, _ in 1:1], [yy for xx in x, yy in y, _ in 1:1], reshape(Zs, size(Zs)..., 1)) do vtk
        vtk["Elevation"] = reshape(Zs, size(Zs)..., 1); surface[0.0] = vtk
    end
    vtk_save(fields); vtk_save(surface)
end

"Every `s`-th point of `A`, and whether any point of each `s × s` block is a sill (phase 4), so that thin sills stay visible."
coarse(A, s) = A[1:s:end, 1:s:end]
coarse_sill(P, s) = [any(==(4), view(P, i:min(i + s - 1, size(P, 1)), j:min(j + s - 1, size(P, 2)))) for i in 1:s:size(P, 1), j in 1:s:size(P, 2)]

function section!(ax, f, s; ix = Colon(), iz = Colon())                             # temperature, layers, sills, melt and surface of the 2D run
    x, z = f["x"][ix][1:s:end] ./ 1.0e3, f["z"][iz][1:s:end] ./ 1.0e3
    P = f["Phases"][ix, iz]; air = coarse(P, s) .== 0
    T = coarse(f["T"][ix, iz], s); T[air] .= NaN
    Z0 = coarse(f["Z0"][ix, iz], s); Z0[air] .= NaN
    S = Float32.(coarse_sill(P, s)); S[S .== 0] .= NaN
    hm = heatmap!(ax, x, z, T, colormap = TMAP, colorrange = (0, 1000))
    heatmap!(ax, x, z, S, colormap = [SILL, SILL])
    contour!(ax, x, z, Z0, levels = LAYERS, color = (:white, 0.35), linewidth = 1)
    contour!(ax, x, z, coarse(f["ϕ"][ix, iz], s), levels = [0.5], color = :white, linewidth = 2.5)
    return hm
end

kde(a, t) = (h = 1.06 * std(a) * length(a)^(-1 / 5); [sum(exp(-0.5 * ((ti - ai) / h)^2) for ai in a) / (length(a) * h * sqrt(2π)) for ti in t])

# --- data ---------------------------------------------------------------------------------------------------------------------
f2 = load(joinpath(dir2, "final.jld2"))
f3 = jldopen(joinpath(dir3, "final.jld2")) do f
    Dict(k => f[k] for k in ("events", "x", "y", "z"))
end
png3 = render_3D(dir3)
ages = zircon_ages(dir3)
events = f3["events"]

# --- figure ---------------------------------------------------------------------------------------------------------------------
fig = Figure()                                                                      # sized to its content (resize_to_layout! below)
panel_label(pos, l) = Label(pos, l, font = :bold, fontsize = 28, padding = (0, 10, 5, 0), halign = :right)

# (a, b) 2D section at present
top = fig[1, 1] = GridLayout()
ax = Axis(top[1, 1], xlabel = "Distance W–E [km]", ylabel = "Height [km]", width = 1100, height = 1100 * 23 / 40, title = "2D: 2.5 m grid, $(length(f2["x"])) × $(length(f2["z"])) points")
hm = section!(ax, f2, 4)
x2 = f2["x"] ./ 1.0e3
lines!(ax, x2, f2["Zsurf"] ./ 1.0e3, color = :black, linewidth = 2)
limits!(ax, -20, 20, -20, 3)
const ZOOM = ((0.5, 9.5), (-8.0, -2.0))                                               # [km]
poly!(ax, Rect(ZOOM[1][1], ZOOM[2][1], ZOOM[1][2] - ZOOM[1][1], ZOOM[2][2] - ZOOM[2][1]), color = :transparent, strokecolor = :white, strokewidth = 2, linestyle = :dash)
panel_label(top[1, 1, TopLeft()], "a")
axz = Axis(top[1, 2], xlabel = "Distance W–E [km]", width = 420, height = 280, title = "Shallow intrusion zone, full resolution")
ix = findall(x -> ZOOM[1][1] <= x <= ZOOM[1][2], x2); iz = findall(z -> ZOOM[2][1] <= z <= ZOOM[2][2], f2["z"] ./ 1.0e3)
section!(axz, f2, 1; ix, iz)
limits!(axz, ZOOM...)
SILLS = load(joinpath(dir2, "sills.jld2"), "sills")                                 # the sills and dikes of the last kyr (their position at intrusion)
new = [Point2f(q.x + f * q.R * cosd(q.angle), q.z - f * q.R * sind(q.angle)) ./ 1.0f3 for q in SILLS if q.time_kyr > -T_NEW for f in (-1, 1)]
linesegments!(ax, new, color = :cyan, linewidth = 1.5)
linesegments!(axz, new, color = :cyan, linewidth = 3)
text!(axz, ZOOM[1][1] + 0.15, ZOOM[2][1] + 0.15, text = "cyan: sills and dikes of the last $(round(Int, T_NEW)) kyr", color = :white, fontsize = 16)
panel_label(top[1, 2, TopLeft()], "b")
Colorbar(top[1, 3], hm, label = "Temperature [°C]", height = Relative(0.7))

# (c) 3D at present
function crop(img)                                                                  # without its white margins
    ink = map(c -> Float32(red(c)) + green(c) + blue(c) < 2.9, img)     # (in Float32: N0f8 sums wrap around)
    rows, cols = findall(vec(any(ink, dims = 2))), findall(vec(any(ink, dims = 1)))
    return rotr90(img[first(rows):last(rows), first(cols):last(cols)])
end
img = load(png3); m = size(img, 2) ÷ 2                                              # the two views, without the line between them
mid = fig[2, 1] = GridLayout()
Label(mid[0, 1:2], "3D: 50 m grid, $(length(f3["x"])) × $(length(f3["y"])) × $(length(f3["z"])) points", font = :bold)
for (k, (half, cmap, range, label)) in enumerate(((img[:, 1:(m - 10)], TMAP, (0, 1000), "Temperature [°C]"), (img[:, (m + 10):end], MELT, (0, 1), "Melt fraction")))
    v = crop(half)
    a = Axis(mid[1, k], width = 780, height = 780 * size(v, 2) / size(v, 1), aspect = DataAspect())
    image!(a, v); hidedecorations!(a); hidespines!(a)
    Colorbar(mid[2, k], colormap = cmap, colorrange = range, label = label, vertical = false, width = 400, flipaxis = false)
end
panel_label(mid[1, 1, TopLeft()], "c")

# (d–g) eruptions and zircon ages of the 3D run
bot = fig[3, 1] = GridLayout()
te = [-e.time_kyr for e in events]; ve = [e.V_km3 for e in events]; o = sortperm(te, rev = true)
axd = Axis(bot[1, 1], xlabel = "Time before present [kyr]", ylabel = "Erupted volume [km³]", title = "Cumulative eruptions")
stairs!(axd, [150.0; te[o]; 0.0], [0.0; cumsum(ve[o]); sum(ve)], step = :post, color = C_ERUPT, linewidth = 3)
text!(axd, 145, 0.95 * sum(ve), text = "$(length(ve)) eruptions\n$(round(sum(ve), digits = 1)) km³ in total", align = (:left, :top), fontsize = 18)
axe = Axis(bot[1, 2], xlabel = "Time before present [kyr]", ylabel = "Volume of eruption [km³]", title = "Eruptions")
linesegments!(axe, [Point2f(t, v) for (t, v0) in zip(te, ve) for v in (0, v0)], color = C_ERUPT, linewidth = 2.5)   # one line per eruption
ylims!(axe, 0, nothing)
t = range(0, 150, length = 601); a = sort(ages)
axf = Axis(bot[1, 3], xlabel = "Zircon age [kyr]", ylabel = "Probability density [1/kyr]", title = "Zircon ages ($(length(a)) zircons)")
band!(axf, t, zeros(length(t)), kde(a, t), color = t, colormap = :viridis, colorrange = (0, 150))
lines!(axf, t, kde(a, t), color = :black, linewidth = 2)
axg = Axis(bot[1, 4], xlabel = "Zircon age [kyr]", ylabel = "Cumulative fraction", title = "Cumulative distribution")
lines!(axg, a, (1:length(a)) ./ length(a), color = :black, linewidth = 3)
for (g, l) in zip(1:4, "defg")
    panel_label(bot[1, g, TopLeft()], string(l))
end
for ax in (axd, axe, axf, axg)
    xlims!(ax, 150, 0)
    ax.width, ax.height = 300, 280
end
resize_to_layout!(fig)
save(out, fig, px_per_unit = 2)
println("written to $out")

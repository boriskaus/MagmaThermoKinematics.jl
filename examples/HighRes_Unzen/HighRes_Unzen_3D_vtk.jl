# Converts the snapshots of the 3D showcase (HighRes_Unzen_3D.jl) to VTK files for ParaView, and renders the movie with
# HighRes_Unzen_3D_Paraview.py if ParaView's `pvbatch` is found.
#
# Usage:  julia --project HighRes_Unzen_3D_vtk.jl HighRes3D_200m          (the output directory of the run)
# Writes <dir>/vtk/{fields,surface}.pvd (unless they exist: delete <dir>/vtk to write them again) and, with ParaView, <dir>/frames/,
# <dir>/movie.mp4 and <dir>/movie.gif.
using JLD2, WriteVTK
dir = isempty(ARGS) ? "HighRes3D_200m" : ARGS[1]
snaps = sort(filter(f -> startswith(f, "snap_"), readdir(dir)))
vdir = mkpath(joinpath(dir, "vtk"))
isfile(joinpath(vdir, "fields.pvd")) || let fields = paraview_collection(joinpath(vdir, "fields")), surface = paraview_collection(joinpath(vdir, "surface"))
    for (i, f) in enumerate(snaps)
        s = load(joinpath(dir, f)); x, y, z = s["x"] ./ 1.0e3, s["y"] ./ 1.0e3, s["z"] ./ 1.0e3; t = s["time_kyr"]
        vtk_grid(joinpath(vdir, "fields_$i"), x, y, z) do vtk
            vtk["Temperature"] = s["T"]; vtk["MeltFraction"] = Float32.(s["ϕ"]) ./ 255; vtk["Phases"] = Float32.(s["Phases"])
            vtk["InitialDepth"] = s["Z0"] ./ 1.0f3                                      # initial height of the rock [km]: its isosurfaces are the deformed layers
            fields[t] = vtk
        end
        Zs = s["Zsurf"] ./ 1.0e3
        vtk_grid(joinpath(vdir, "surface_$i"), [xx for xx in x, yy in y, _ in 1:1], [yy for xx in x, yy in y, _ in 1:1], reshape(Zs, size(Zs)..., 1)) do vtk
            vtk["Elevation"] = reshape(Zs, size(Zs)..., 1); surface[t] = vtk
        end
    end
    vtk_save(fields); vtk_save(surface)
end
pvbatch = Sys.which("pvbatch")                                                     # or the path to pvbatch, if it is not on the PATH
if pvbatch !== nothing
    using CairoMakie                                                                   # for its ffmpeg and the colour map
    cmap = joinpath(vdir, "lajolla.txt")                                               # temperature in LaJolla (F. Crameri), as in 2D
    write(cmap, join(("$(c.r) $(c.g) $(c.b)" for c in cgrad(:lajolla).colors), "\n"))
    run(`$pvbatch --force-offscreen-rendering $(joinpath(@__DIR__, "HighRes_Unzen_3D_Paraview.py")) $vdir $(joinpath(dir, "frames")) 1 $cmap`)
    ffmpeg = CairoMakie.Makie.FFMPEG_jll.ffmpeg(); fr = joinpath(dir, "frames", "frame_%04d.png")
    run(`$ffmpeg -y -loglevel error -framerate 12 -i $fr -pix_fmt yuv420p $(joinpath(dir, "movie.mp4"))`)
    run(`$ffmpeg -y -loglevel error -framerate 10 -i $fr -vf "scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" -loop 0 $(joinpath(dir, "movie.gif"))`)
end
println("written to $dir")

module MTKMakieExt

import Makie
import MagmaThermoKinematics: save_phase_diagram

function save_phase_diagram(path, Tvec, Pvec, fields)
    fig = Makie.Figure(size = (900, 800))
    for (i, (values, title, colormap)) in enumerate(fields)
        row, col = divrem(i - 1, 2)
        ax = Makie.Axis(fig[row + 1, 2col + 1]; xlabel = "T [°C]", ylabel = "P [kbar]", title)
        hm = Makie.heatmap!(ax, Tvec, Pvec ./ 1.0e3, values; colormap)
        Makie.Colorbar(fig[row + 1, 2col + 2], hm)
    end
    return Makie.save(path, fig)
end

end

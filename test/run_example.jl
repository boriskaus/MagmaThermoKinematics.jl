# Runs one example script, unchanged except that the time loop is capped.
# Usage: julia --project=<env> run_example.jl <example.jl> [nsteps]
example = ARGS[1]
const NSTEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6

using MagmaThermoKinematics, GeophysicalModelGenerator
import MagmaThermoKinematics: MTK_GMG, NumericalParameters, SillParameters

"Cap the run at `NSTEPS` time steps, with output saved on the last one and no figures."
function cap!(Num)
    Num.nt = min(Num.nt, NSTEPS)
    Num.SaveOutput_steps = Num.nt
    Num.CreateFig_steps = typemax(Int)
    return Num
end

# More specific than the package's method, so it wraps rather than replaces it.
MagmaThermoKinematics.MTK_GeoParams(Mat::Tuple, Num::NumParam, Dikes::SillParams; kw...) =
    invoke(MagmaThermoKinematics.MTK_GeoParams, Tuple{Tuple, NumericalParameters, SillParameters}, Mat, cap!(Num), Dikes; kw...)
# CartData setups recompute `nt` inside the solver.
MTK_GMG.Setup_Model_CartData(d::CartData, Num::NumParam, Mat::Tuple) =
    cap!(invoke(MTK_GMG.Setup_Model_CartData, Tuple{CartData, NumericalParameters, Tuple}, d, Num, Mat))

include(abspath(example))
println("RUNNER: OK ", basename(example))

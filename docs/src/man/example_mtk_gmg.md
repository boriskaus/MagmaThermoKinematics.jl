# MTK_GMG Style

This set of examples demonstrates the MTK_GMG coding style: integrating MagmaThermoKinematics with [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) inputs and user-overridden runtime hooks. 

This is a nice feature as it allows you to model natural magmatic systems, and take observables such as seismic tomography and the actual topography into account while constructing your model. The `GeophysicalModelGenerator` package allows you to even import screenshots from published papers and compare them with the modelling results. Likewise, you can use MTK output along with [MAGEMin](https://computationalthermodynamics.github.io/MAGEMin_C.jl/dev/) phase diagrams (and seismic velocities) to compute synthetic seismic tomography output from your simulation.

MTK is fully customisable, and allows you to inject a deeper magmatic system for some period of time, while later switching to shallower injections. Thanks to its integration with [GeoParams](https://github.com/JuliaGeodynamics/GeoParams.jl), it is also straightforward to change the material parameters (and test the effect of temperature-dependent thermal conductivity vs. constant conductivity, for example). At the same time, your whole simulation is contained in a single file, which makes it highly reproducible and easy to share — as required by the FAIR data principles of many journals.

The Unzen and Lanin 3D examples show how different geological datasets and initialization workflows can be plugged into one consistent MTK_GMG simulation pipeline.


## Core Pattern
The general workflow is as follows:

1. Initialize backend and MTK modules.
2. Optionally construct or import a CartData model using the [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl).
3. Override selected functions in the MTK_GMG namespace for custom output, visualization, initialization, or time-dependent diagnostics, or dike injection.
4. Define NumParam and DikeParam inputs.
5. Define material parameters as tuples of SetMaterialParams entries.
6. Run MTK_GeoParams_2D or MTK_GeoParams_3D.

## Overriding custom settings

The MTK_GMG style commonly overrides these functions:

- MTK_GMG.MTK_initialize!
- MTK_GMG.MTK_visualize_output
- MTK_GMG.MTK_print_output
- MTK_GMG.MTK_update_TimeDepProps!

## Dedicated Examples
The following examples illustrate how to use this workflow:

- [MTK_GMG Example 1](example_mtk_gmg1.md)
- [MTK_GMG Example 2 (Unzen)](example_mtk_gmg2.md)
- [MTK_GMG 3D Examples (Unzen3D, Lanin3D)](example_mtk_gmg_3d.md)

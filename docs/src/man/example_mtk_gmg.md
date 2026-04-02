# MTK_GMG Style

This set of examples demonstrates the MTK_GMG coding style: integrating MagmaThermoKinematics with [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) inputs and user-overridden runtime hooks.

## Core Pattern

1. Initialize backend and MTK modules.
2. Optionally construct or import a CartData model using [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl).
3. Override selected functions in the MTK_GMG namespace for custom output, visualization, initialization, or time-dependent diagnostics.
4. Define NumParam and DikeParam inputs.
5. Define material parameters as tuples of SetMaterialParams entries.
6. Run MTK_GeoParams_2D.

## Typical Hook Overrides

The MTK_GMG style commonly overrides these functions:

- MTK_GMG.MTK_initialize!
- MTK_GMG.MTK_visualize_output
- MTK_GMG.MTK_print_output
- MTK_GMG.MTK_update_TimeDepProps!

## Dedicated Examples

- [MTK_GMG Example 1](example_mtk_gmg1.md)
- [MTK_GMG Example 2 (Unzen)](example_mtk_gmg2.md)

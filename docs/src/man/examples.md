# Examples

The repository contains ready-to-run examples in the top-level examples directory.

For dedicated walkthrough pages, see:

- [2D Example](example2d.md)
- [3D Example](example3d.md)
- [MTK_GMG Style](example_mtk_gmg.md)
- [MTK_GMG Example 1](example_mtk_gmg1.md)
- [MTK_GMG Example 2 (Unzen)](example_mtk_gmg2.md)
- [MTK_GMG 3D Examples (Unzen and Lanin)](example_mtk_gmg_3d.md)

## Main Scripts

- Example2D.jl: 2D thermal evolution with repeated intrusions.
- Example3D.jl: 3D run with VTK output for ParaView.
- Example2D_ZASSy.jl: benchmark-style setup used in published comparisons.
- MTK_GMG_2D_example1.jl and MTK_GMG_2D_example2.jl: workflows integrating [GeophysicalModelGenerator](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) input.

## Visual Outputs

2D example:

![](../assets/movies/Example2D.gif)

3D example:

![](../assets/movies/Example3D.gif)

MTK_GMG 3D examples:

![](../assets/movies/Unzen3D.gif)

![](../assets/movies/Lanin3D.gif)

## Running an Example

From the repository root:

```julia
include("examples/Example2D.jl")
```

or

```julia
include("examples/Example3D.jl")
```

## Code Snippets

The full README-equivalent code snippets are available in dedicated tabs:

- [2D Example](example2d.md)
- [3D Example](example3d.md)

## Notes

- GPU execution typically requires loading CUDA before backend initialization.
- `environment!(...)` initializes package internals; if an example script uses ParallelStencil macros directly (`@zeros`, `@parallel`, etc.), it should also call `@init_parallel_stencil(...)` in script scope.
- 3D output is commonly explored with ParaView using VTK and PVD files.

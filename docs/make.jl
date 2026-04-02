using Documenter
using DocumenterVitepress
using MagmaThermoKinematics

repo_root = dirname(@__DIR__)
docs_src = joinpath(@__DIR__, "src")
man_dir = joinpath(docs_src, "man")
assets_movies = joinpath(docs_src, "assets", "movies")

mkpath(man_dir)
mkpath(assets_movies)

# Copy media used by README so links render in docs.
for movie in ("Example2D.gif", "Example3D.gif")
    src = joinpath(repo_root, "examples", "movies", movie)
    dst = joinpath(assets_movies, movie)
    isfile(src) && cp(src, dst; force = true)
end

# Mirror repository license in docs.
license_src = joinpath(repo_root, "LICENSE")
if isfile(license_src)
    write(joinpath(man_dir, "license.md"), read(license_src, String))
end

makedocs(;
    sitename = "MagmaThermoKinematics.jl",
    authors = "MagmaThermoKinematics contributors",
    modules = [MagmaThermoKinematics],
    checkdocs = :none,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/boriskaus/MagmaThermoKinematics.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => Any[
            "Installation" => "man/installation.md",
            "Dependencies" => "man/dependencies.md",
            "Quick Start" => "man/quickstart.md",
            "Examples" => Any[
                "Overview" => "man/examples.md",
                "2D Example" => "man/example2d.md",
                "3D Example" => "man/example3d.md",
                "MTK_GMG Style" => "man/example_mtk_gmg.md",
                "MTK_GMG Example 1" => "man/example_mtk_gmg1.md",
                "MTK_GMG Example 2 (Unzen)" => "man/example_mtk_gmg2.md",
            ],
            "Numerics and Physics" => "man/numerics.md",
            "Ongoing Development" => "man/development.md",
        ],
        "API" => Any[
            "Function Reference" => "man/listfunctions.md",
        ],
        "References" => Any[
            "Benchmarking" => "man/benchmarking.md",
            "Related Work" => "man/related_work.md",
            "Citing" => "man/citing.md",
            "License" => "man/license.md",
        ],
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/boriskaus/MagmaThermoKinematics.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)

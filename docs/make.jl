using Documenter
using DocumenterVitepress
using MagmaThermoKinematics
using GeoParams

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
license = read(joinpath(repo_root, "LICENSE"), String)
write(joinpath(man_dir, "license.md"), license)

security = read(joinpath(repo_root, "SECURITY.md"), String)
write(joinpath(man_dir, "security.md"), security)

# Copy list of authors to not need to synchronize it manually
authors_text = read(joinpath(repo_root, "AUTHORS.md"), String)
# authors_text = replace(authors_text, "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
write(joinpath(man_dir, "authors.md"), authors_text)


# Copy some files from the repository root directory to the docs and modify them as necessary
# Based on: https://github.com/ranocha/SummationByPartsOperators.jl/blob/0206a74140d5c6eb9921ca5021cb7bf2da1a306d/docs/make.jl#L27-L41
open(joinpath(man_dir, "license.md"), "w") do io
    # Point to source license file
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/LICENSE"
        ```
        """
    )
    # Write the modified contents
    println(io, "# [License](@id license)")
    println(io, "")
    for line in eachline(joinpath(dirname(@__DIR__), "LICENSE"))
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, "> ", line)
    end
end

open(joinpath(man_dir, "code_of_conduct.md"), "w") do io
    # Point to source license file
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/CODE_OF_CONDUCT.md"
        ```
        """
    )
    # Write the modified contents
    println(io, "# [Code of Conduct](@id code-of-conduct)")
    println(io, "")
    for line in eachline(joinpath(dirname(@__DIR__), "CODE_OF_CONDUCT.md"))
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, "> ", line)
    end
end

open(joinpath(man_dir, "contributing.md"), "w") do io
    # Point to source license file
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/boriskaus/MagmaThermoKinematics.jl/blob/main/CONTRIBUTING.md"
        ```
        """
    )
    # Write the modified contents
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, line)
    end
end

makedocs(;
    sitename = "MagmaThermoKinematics.jl",
    authors = "Boris Kaus, Pascal Aellig, Albert de Montserrat",
    modules = [MagmaThermoKinematics],
    checkdocs = :none,
    warnonly = Documenter.except(:footnote),
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
            "GeoParams Parameterisations" => Any[
                "Melting" => "man/MeltingParameterisations.md",
                "Conductivity" => "man/ConductivityParameterisations.md",
            ]
        ],
        "References" => Any[
            "Benchmarking" => "man/benchmarking.md",
            "Related Work" => "man/related_work.md",
            "Citing" => "man/citing.md",
        ],
        "Authors" => "man/authors.md",
        "Contributing" => "man/contributing.md",
        "Code of Conduct" => "man/code_of_conduct.md",
        "Security" => "man/security.md",
        "License" => "man/license.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/boriskaus/MagmaThermoKinematics.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)

using PubBot, Test, JSON, Dates
using PubBot: acknowledged, normalize_doi, extract_dois, title_similarity, initials, entry_line, load_config,
    Page, list_range, page_dois, insert_entry!, remove_entry!, Work, Stack, Package, assess,
    word_pattern, select_candidates, team_authors

# The engine tests use the configuration of GPU4GEO, the project this bot comes from (test/gpu4geo);
# the "MagmaThermoKinematics.jl" testset uses the real one.
const CFG = load_config(joinpath(@__DIR__, "gpu4geo"))
const MTK = load_config()
fixture(name) = JSON.parse(read(joinpath(@__DIR__, "fixtures", "csl_$name.json"), String))

@testset "DOIs" begin
    @test normalize_doi("https://doi.org/10.1029/2025JB031240") == "10.1029/2025jb031240"
    @test normalize_doi("doi: 10.5194/gmd-15-5757-2022.") == "10.5194/gmd-15-5757-2022"
    line = "- X (2024). [https://doi.org/10.21105/joss.09365](https://joss.theoj.org/papers/10.21105/joss.09365)"
    @test extract_dois(line) == ["10.21105/joss.09365"]
    @test extract_dois("see 10.1234/abc?x=\"1\" and <script>") == ["10.1234/abc"]
end

@testset "Titles" begin
    @test title_similarity(
        "Automatic tuning of iterative pseudo-transient solvers for modelling the deformation of heterogeneous media",
        "Automatic tuning of iterative pseudo-transient solvers for modeling the deformation of heterogeneous media"
    ) >= 0.8
    @test title_similarity("Ice flow, Part I", "Ice flow, Part II") == 0
    @test title_similarity("Räss et al.", "rass et al") == 1
    # Versions of the same paper found by the backtest.
    @test PubBot.similar_titles(
        "Overcoming the numerical challenges owing to rapid ductile localization",
        "Overcoming the numerical challenges owing to rapid ductile localization with DEDLoc (version 1.0.0)"
    )
    @test PubBot.similar_titles(
        "Graphics processing unit accelerated ice flow solver for unstructured meshes using the Shallow Shelf Approximation",
        "Graphics-processing-unit-accelerated ice flow solver for unstructured meshes using the Shallow-Shelf Approximation (FastIceFlo v1.0.1)"
    )
    @test !PubBot.similar_titles("GPU solvers", "GPU solvers for ice flow in Julia")  # too short to extend
    @test !PubBot.similar_titles(
        "Ice flow modelling on GPUs, Part I: theory and numerics",
        "Ice flow modelling on GPUs, Part II: theory and numerics"
    )
end

@testset "Formatting" begin
    @test initials("Boris J. P.") == "B. J. P."
    @test initials("Jean-Pierre") == "J.-P."
    @test initials("A.") == "A."
    @test PubBot.display_name_citation("Albert de Montserrat") == "de Montserrat, A."
    @test PubBot.display_name_citation("Boris J. P. Kaus") == "Kaus, B. J. P."
    @test PubBot.display_name_citation("Plato") == "Plato"
    # DOI metadata without authors (early JuliaCon proceedings): OpenAlex names instead.
    anonymous = delete!(copy(fixture("10_21105_jcon_00137")), "author")
    @test startswith(
        entry_line(anonymous, CFG; author_names = ["Samuel Omlin", "Michael Schlottke‐Lakemper", "Ivan Utkin"]),
        "- Omlin, S., Schlottke-Lakemper, M., and Utkin, I. (2024). **Distributed Parallelization"
    )
    @test PubBot.clean_text("Thermo‐Mechanical <i>Controls</i>, 2025 &amp; 10") == "Thermo-Mechanical Controls, 2025 & 10"
    @test entry_line(fixture("10_5194_gmd-19-5343-2026"), CFG) ==
        "- Duretz, T., de Montserrat, A., Sevilla, R., Räss, L., Utkin, I., and Spang, A. (2026). **Automatic tuning of iterative pseudo-transient solvers for modeling the deformation of heterogeneous media.** Geoscientific Model Development, 19(12), 5343–5362. [https://doi.org/10.5194/gmd-19-5343-2026](https://doi.org/10.5194/gmd-19-5343-2026)"
    @test entry_line(fixture("10_5194_egusphere-2025-5641"), CFG) ==
        "- Duretz, T., de Montserrat, A., Sevilla, R., Räss, L., Utkin, I., and Spang, A. (2025). **Automatic tuning of iterative pseudo-transient solvers for modelling the deformation of heterogeneous media.** EGUsphere (preprint). [https://doi.org/10.5194/egusphere-2025-5641](https://doi.org/10.5194/egusphere-2025-5641)"
    @test entry_line(fixture("10_21105_joss_06763"), CFG) ==
        "- Kaus, B. J. P., Thielmann, M., Aellig, P., de Montserrat, A., de Siena, L., Frasunkiewicz, J., Fuchs, L., Piccolo, A., Ranocha, H., Riel, N., Schuler, C., Spang, A., and Weiler, T. (2024). **GeophysicalModelGenerator.jl: A Julia package to visualise geoscientific data and create numerical model setups.** Journal of Open Source Software, 9(103), 6763. [https://doi.org/10.21105/joss.06763](https://doi.org/10.21105/joss.06763)"
    @test entry_line(fixture("10_1029_2025jb031240"), CFG) ==
        "- Spang, A., Thielmann, M., de Montserrat, A., and Duretz, T. (2025). **Transient Propagation of Ductile Ruptures by Thermal Runaway.** Journal of Geophysical Research: Solid Earth, 130(6), e2025JB031240. [https://doi.org/10.1029/2025jb031240](https://doi.org/10.1029/2025jb031240)"
    @test entry_line(fixture("10_48550_arxiv_2211_02740"), CFG) ==
        "- Churavy, V., Godoy, W. F., Bauer, C., Ranocha, H., Schlottke-Lakemper, M., Räss, L., Blaschke, J., Giordano, M., Schnetter, E., Omlin, S., Vetter, J. S., and Edelman, A. (2022). **Bridging HPC Communities through the Julia Programming Language.** arXiv (preprint). [https://doi.org/10.48550/ARXIV.2211.02740](https://doi.org/10.48550/ARXIV.2211.02740)"
    @test entry_line(fixture("10_1016_j_compgeo_2025_107189"), CFG) ==
        "- Huo, Z., Alkhimenkov, Y., Jaboyedoff, M., Podladchikov, Y., Räss, L., Wyser, E., and Mei, G. (2025). **A high-performance backend-agnostic Material Point Method solver in Julia.** Computers and Geotechnics, 183, 107189. [https://doi.org/10.1016/j.compgeo.2025.107189](https://doi.org/10.1016/j.compgeo.2025.107189)"
end

const PAGE_TEXT = """
+++
header = "Publications"
+++

### 2026

- A (2026). [https://doi.org/10.1234/a](https://doi.org/10.1234/a)

### 2024

- B (2024). [https://doi.org/10.1234/b](https://doi.org/10.1234/b)

## Conferences & Events

### 2025

- [talk](https://doi.org/10.1234/talk)
"""

page() = Page("", split(chomp(PAGE_TEXT), "\n"))
text(p) = join(p.lines, "\n") * "\n"

@testset "Page editing" begin
    p = page()
    @test page_dois(p) == Set(["10.1234/a", "10.1234/b"])  # conference section ignored
    insert_entry!(p, 2026, "- new2026")
    @test occursin("### 2026\n\n- new2026\n- A (2026)", text(p))
    insert_entry!(p, 2025, "- new2025")
    @test occursin("- A (2026). [https://doi.org/10.1234/a](https://doi.org/10.1234/a)\n\n### 2025\n\n- new2025\n\n### 2024", text(p))
    insert_entry!(p, 2027, "- new2027")
    @test occursin("+++\n\n### 2027\n\n- new2027\n\n### 2026", text(p))
    insert_entry!(p, 2020, "- new2020")
    @test occursin("- B (2024). [https://doi.org/10.1234/b](https://doi.org/10.1234/b)\n\n### 2020\n\n- new2020\n\n## Conferences", text(p))
    @test occursin("## Conferences & Events\n\n### 2025\n\n- [talk]", text(p))

    p = page()
    @test remove_entry!(p, "10.1234/b")
    @test text(p) == replace(PAGE_TEXT, "### 2024\n\n- B (2024). [https://doi.org/10.1234/b](https://doi.org/10.1234/b)\n\n" => "")
    @test !remove_entry!(p, "10.1234/talk")

    # The real page: parsing finds every entry's DOI, and the introduction names none.
    real = PubBot.load_page()
    r = list_range(real)
    @test startswith(real.lines[1], "# Publications") && last(r) == length(real.lines)
    @test length(page_dois(real)) == count(l -> startswith(l, "- "), real.lines[r])
end

function work(;
        title = "A paper", authors = [("Ludovic Räss", "0000-0002-1136-899X")], doi = "10.1234/x",
        refs = Set{String}(), abstract = "", type = "article"
    )
    return Work(; id = "W1", doi, title, date = Date(2026, 1, 1), type, authors, refs, abstract)
end

@testset "Classification" begin
    stack = Stack(
        CFG.packages, Dict(d => p for p in CFG.packages for d in p.dois),
        Dict("W99" => only(filter(p -> p.name == "JustRelax.jl", CFG.packages)))
    )
    kaus = [("Boris J. P. Kaus", "0000-0002-0247-8660")]   # co-PI, no acknowledgment required
    schuler = [("Christian Schuler", "")]                  # team, matched by name not ORCID
    outsider = [("Jane Doe", "0000-0000-0000-0000")]

    @test team_authors(work(; authors = [("Albert de Montserrat", ""), ("A. B. Utkin", "")]), CFG) |> length == 1
    @test assess(work(), CFG, stack, "We use ParallelStencil.jl on GPUs.").bucket == :strong
    @test assess(work(; refs = Set(["W99"])), CFG, stack, nothing).bucket == :strong
    @test assess(work(; doi = "10.21105/joss.09365"), CFG, stack, nothing).bucket == :strong
    # General Julia GPU packages count for any team author.
    @test assess(work(; authors = kaus), CFG, stack, "Differentiated with Enzyme.jl").bucket == :strong
    @test assess(work(; authors = schuler), CFG, stack, "Runs with CUDA.jl").bucket == :strong
    # Trixi.jl only for GPU work: GPU in title/abstract, or repeatedly in the body.
    @test assess(work(; authors = kaus), CFG, stack, "Implemented in Trixi.jl.").bucket == :none
    @test assess(work(; authors = kaus, abstract = "Trixi.jl on NVIDIA GPUs"), CFG, stack, "Trixi.jl").bucket == :strong
    gpu_body = "Trixi.jl runs on GPUs. " * "GPU kernels, GPU memory. "
    @test assess(work(; authors = kaus), CFG, stack, gpu_body).bucket == :strong
    filler = repeat("Lorem ipsum. ", 200)
    cites = "We use Trixi.jl, which may one day run on a GPU. $filler References [1] GPU paper [2] GPU paper [3] GPU paper"
    @test assess(work(; authors = kaus), CFG, stack, cites).bucket == :none
    @test PubBot.body_text("Intro $filler References [1] x") == "Intro $filler "
    @test PubBot.body_text("References to GPUs early on. $filler") == "References to GPUs early on. $filler"
    # Papers with no team author are at most "review".
    @test assess(work(; authors = outsider), CFG, stack, "Built on JustRelax.jl").bucket == :review
    # Keywords only.
    @test assess(work(), CFG, stack, "Julia code, see Foo.jl, runs on GPUs with GPU kernels and GPU arrays").bucket == :review
    @test assess(work(), CFG, stack, "Julia code, see Foo.jl, runs on a GPU").bucket == :none
    @test assess(work(; abstract = "A Julia solver"), CFG, stack, nothing).bucket == :review
    @test assess(work(), CFG, stack, "Julia Smith helped with the GPU runs").bucket == :none
    @test assess(work(), CFG, stack, "Nothing relevant; FastIcer and GeoParamsX are not packages").bucket == :none

    # People who must acknowledge the project (Tackley, Moses, Schlottke-Lakemper, Ranocha, Omlin).
    ranocha = [("Hendrik Ranocha", "0000-0002-3456-2277")]
    omlin = [("Samuel Omlin", "0000-0001-8969-5619")]
    @test assess(work(; authors = omlin), CFG, stack, "Runs with CUDA.jl").bucket == :none
    @test assess(work(; authors = omlin), CFG, stack, "CUDA.jl. Acknowledgements: GPU4GEO.").bucket == :strong
    @test assess(work(; authors = ranocha), CFG, stack, "We use ParallelStencil.jl on GPUs.").bucket == :review
    @test assess(
        work(; authors = ranocha), CFG, stack,
        "We use ParallelStencil.jl on GPUs. Acknowledgments: funded by GPU4GEO."
    ).bucket == :strong
    @test assess(
        work(; authors = ranocha), CFG, stack,
        "Trixi.jl on GPUs, GPU kernels, GPU memory."
    ).bucket == :none  # generic only, no thanks
    # A non-flagged team co-author lifts the requirement.
    @test assess(
        work(; authors = [ranocha[1], ("Ludovic Räss", "")]), CFG, stack,
        "We use ParallelStencil.jl on GPUs."
    ).bucket == :strong
    # Closed access: cannot be checked, so a human decides.
    @test assess(work(; authors = ranocha, abstract = "A Julia GPU solver"), CFG, stack, nothing).bucket == :review
    # "PASC" and "c44" only count inside the acknowledgments section: C44 is an elastic constant.
    elastic = "We use JustRelax.jl. The elastic constant c44 governs shear. Acknowledgments: thanks to nobody."
    @test assess(work(; authors = ranocha), CFG, stack, elastic).bucket == :review
    @test acknowledged(
        assess(
            work(; authors = ranocha), CFG, stack,
            "JustRelax.jl. Acknowledgements This work was funded by PASC project c44."
        )
    )
    # Thanking the project by name is enough on its own, for any team author.
    thanks = "A finite difference scheme. Acknowledgements: supported by the GPU4GEO project."
    @test assess(work(), CFG, stack, thanks).bucket == :strong                      # Räss, no package named
    @test assess(work(; authors = ranocha), CFG, stack, thanks).bucket == :strong     # flagged person
    @test assess(work(; authors = outsider), CFG, stack, thanks).bucket == :none      # nobody from the team
    # PASC funds many projects: not proof on its own.
    pasc = "A finite difference scheme. Acknowledgements: supported by PASC."
    @test assess(work(), CFG, stack, pasc).bucket == :review
    # ... but it does satisfy a flagged person's requirement, so a core package still counts.
    @test assess(work(; authors = ranocha), CFG, stack, "JustRelax.jl. Acknowledgements: PASC.").bucket == :strong
    @test PubBot.acknowledgment("Funded by the ∂GPU4GEO project.", CFG) == "GPU4GEO"
    @test PubBot.acknowledgment("Nothing here.", CFG) == ""
    @test PubBot.acknowledgment(nothing, CFG) === nothing

    # OpenAlex full-text index, when the PDF is not reachable.
    groups = PubBot.probe_groups(stack)
    core = only(filter(g -> g.tier == "core" && !g.gpu, groups))
    generic = only(filter(g -> g.tier == "generic" && !g.gpu, groups))
    @test "JustRelax" in core.match && "CUDA.jl" in generic.match && !("github.com/PTsolvers" in core.match)
    probes(pairs...; gpu = Set{String}()) = PubBot.Probes(Set(["W1"]), collect(pairs), gpu, Set{String}())
    a = assess(work(), CFG, stack, nothing, probes(core => Set(["W1"])))
    @test a.bucket == :strong && a.fulltext == :index
    @test assess(work(; authors = schuler), CFG, stack, nothing, probes(generic => Set(["W1"]))).bucket == :review
    # Index hits for generic packages ("CUDA.jl" may match plain "CUDA") need a human.
    @test assess(work(; authors = kaus), CFG, stack, nothing, probes(generic => Set(["W1"]))).bucket == :review
    @test assess(work(), CFG, stack, nothing, probes(core => Set{String}())).bucket == :none
end

@testset "Candidate selection" begin
    pre = work(; doi = "10.5194/egusphere-2025-1", title = "Automatic tuning of solvers for modelling media", type = "preprint")
    pub = work(; doi = "10.5194/gmd-19-1-2026", title = "Automatic tuning of solvers for modeling media")
    zen = work(; doi = "10.5281/zenodo.1", title = "Automatic tuning of solvers for modeling media")
    thanks = work(; doi = "10.1029/thanks", title = "Thank You to Our 2025 Peer Reviewers")
    c = select_candidates([pre, pub, zen, thanks], CFG; on_page = Set{String}(), seen = Set{String}())
    @test length(c) == 1 && c[1].doi == pub.doi && c[1].other_dois == [pre.doi]
    @test isempty(select_candidates([pre, pub], CFG; on_page = Set([pre.doi]), seen = Set{String}()))
    @test isempty(select_candidates([pre, pub], CFG; on_page = Set{String}(), seen = Set([pub.doi])))
end

@testset "MagmaThermoKinematics.jl" begin
    @test !MTK.require_team && MTK.search_mentions && MTK.citing_core
    mtk = only(MTK.packages)
    stack = Stack(MTK.packages, Dict(d => mtk for d in mtk.dois), Dict("W98" => mtk))   # W98: Schmitt et al. (2023)
    outsider = [("Jane Doe", "0000-0000-0000-0000")]
    kaus = [("Boris J. P. Kaus", "0000-0002-0247-8660")]
    probes(hit) = PubBot.Probes(Set(["W1"]), [mtk => Set(hit ? ["W1"] : String[])], Set{String}(), Set{String}())
    # Named by anyone: listed, whether from the PDF, the abstract or OpenAlex's full-text index.
    @test assess(work(; authors = outsider), MTK, stack, "Models were run with MagmaThermoKinematics.jl.").bucket == :strong
    @test assess(work(; authors = outsider, abstract = "using MagmaThermoKinematics"), MTK, stack, nothing).bucket == :strong
    @test assess(work(; authors = outsider, refs = Set(["W98"])), MTK, stack, nothing, probes(true)).bucket == :strong
    # Citing the package paper alone (zircon age spectra): a human decides.
    @test assess(work(; authors = outsider, refs = Set(["W98"])), MTK, stack, nothing).bucket == :review
    @test assess(work(; authors = outsider, refs = Set(["W98"])), MTK, stack, nothing, probes(false)).bucket == :review
    @test assess(work(; authors = outsider, refs = Set(["W98"])), MTK, stack, "Zircon ages as in Schmitt et al.").bucket == :review
    # The developers' other Julia/GPU papers are not proposed.
    @test assess(work(; authors = kaus), MTK, stack, "A Julia GPU solver, see JustRelax.jl; GPU kernels, GPU memory, GPU arrays").bucket == :none
    @test assess(work(; authors = kaus, abstract = "A Julia GPU solver"), MTK, stack, nothing).bucket == :none
    @test assess(work(; authors = kaus), MTK, stack, "Thermal models with MagmaThermoKinematics.").bucket == :strong
    # Not a whole word.
    @test assess(work(; authors = outsider), MTK, stack, "MagmaThermoKinematicsX").bucket == :none
end

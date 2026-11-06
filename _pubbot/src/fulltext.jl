#=
Full text comes from open-access PDFs. Copernicus, arXiv, JOSS/JuliaCon and HAL serve PDFs
to scripts; Wiley/AGU, OUP and Elsevier do not, so for those we fall back on OpenAlex's
own full-text index (`fulltext_probes`).
=#

const CACHE = joinpath(BOTDIR, ".cache")

"Candidate PDF URLs: known publisher patterns first, then OpenAlex locations."
function pdf_candidates(w::Work)
    urls = String[]
    d = w.doi
    if startswith(d, "10.48550/arxiv.")
        push!(urls, "https://arxiv.org/pdf/" * d[(length("10.48550/arxiv.") + 1):end])
    elseif startswith(d, "10.21105/joss.")
        push!(urls, "https://joss.theoj.org/papers/$d.pdf")
    elseif startswith(d, "10.21105/jcon.")
        push!(urls, "https://proceedings.juliacon.org/papers/$d.pdf")
    elseif startswith(d, "10.5194/")
        push!(urls, "copernicus:")  # resolved lazily, needs a request
    end
    occursin(r"//[^/]*hal\.science/|//hal\.[^/]+/", w.landing) && push!(urls, rstrip(w.landing, '/') * "/document")
    return unique!(append!(urls, w.pdf_urls))
end

"Copernicus PDFs live next to the article landing page, named after the DOI suffix."
function copernicus_pdf(doi::AbstractString)
    status, _, landing = http_get(doi_url(doi); method = "HEAD")
    status == 200 || return nothing
    return rstrip(landing, '/') * "/" * split(doi, '/'; limit = 2)[2] * ".pdf"
end

function pdf_to_text(bytes::Vector{UInt8})
    return mktemp() do path, io
        write(io, bytes)
        close(io)
        try
            read(`$(pdftotext()) -q -enc UTF-8 $path -`, String)
        catch
            ""
        end
    end
end

"Undo PDF line-break artefacts so package names and URLs match."
function normalize_text(s::AbstractString)
    s = replace(s, r"(\w)-\s*\n\s*(\w)" => s"\1\2")
    s = replace(s, r"\.\s+jl\b" => ".jl", r"github\.com/\s+" => "github.com/")
    return replace(s, r"\s+" => " ")
end

cache_file(doi) = joinpath(CACHE, replace(doi, r"[^A-Za-z0-9._-]" => "_") * ".txt")

# Publishers that refuse PDF downloads from scripts (403), not worth a request.
const BLOCKED_HOSTS = r"^https?://([^/]+\.)?(wiley\.com|oup\.com|sciencedirect\.com|tandfonline\.com)/"

"Text of the paper's open-access PDF, or `nothing`. Cached in `_pubbot/.cache`."
function fulltext(w::Work)
    path = cache_file(w.doi)
    if isfile(path)
        text = read(path, String)
        return isempty(text) ? nothing : text
    end
    text = ""
    for url in pdf_candidates(w)
        url == "copernicus:" && (url = copernicus_pdf(w.doi); url === nothing && continue)
        occursin(BLOCKED_HOSTS, url) && continue
        status, body, _ = http_get(url; timeout = 45, retries = 1)
        status == 200 && length(body) > 4 && body[1:4] == b"%PDF" || continue
        text = normalize_text(pdf_to_text(body))
        isempty(text) || break
    end
    mkpath(CACHE)
    write(path, text)
    return isempty(text) ? nothing : text
end

"Full texts of `works` (in order), downloaded a few at a time."
fulltexts(works::Vector{Work}) = asyncmap(fulltext, works; ntasks = 6)

"Results of OpenAlex full-text searches over the works in `searched`."
struct Probes
    searched::Set{String}
    packages::Vector{Pair{Package, Set{String}}}  # package group => IDs mentioning one of its packages
    gpu::Set{String}
    julia::Set{String}
end

Probes() = Probes(Set{String}(), Pair{Package, Set{String}}[], Set{String}(), Set{String}())

"""
Packages with the same rules (tier, GPU requirement) searched together: a full-text search
costs 10 OpenAlex credits, so one OR query per group instead of one per package.
"""
function probe_groups(stack::Stack)
    groups = Dict{Tuple{String, Bool}, Vector{Package}}()
    foreach(p -> push!(get!(groups, (p.tier, p.gpu), Package[]), p), stack.packages)
    return map(collect(groups)) do ((tier, gpu), ps)
        name = length(ps) == 1 ? ps[1].name : "a $tier package"
        terms = [m for p in ps for m in p.match if !occursin('/', m)]
        Package(name, tier, gpu, terms, String[], Regex[])
    end
end

function search_fulltext(ids::Vector{String}, terms::Vector{String})
    hits = Set{String}()
    isempty(terms) && return hits
    search = join(("\"$t\"" for t in terms), " OR ")
    for chunk in Iterators.partition(ids, 100)
        filter = "ids.openalex:$(join(chunk, "|")),fulltext.search:$search"
        union!(hits, short_id(x["id"]) for x in openalex_works(filter; select = "id", per_page = 100))
    end
    return hits
end

"Search OpenAlex's full-text index of works `ids` for stack packages, GPU and Julia."
function fulltext_probes(stack::Stack, ids::Vector{String})
    isempty(ids) && return Probes()
    packages = [g => search_fulltext(ids, g.match) for g in probe_groups(stack)]
    return Probes(Set(ids), packages, search_fulltext(ids, ["GPU"]), search_fulltext(ids, ["Julia"]))
end

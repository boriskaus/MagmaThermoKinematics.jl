const OPENALEX = "https://api.openalex.org"
const WORK_FIELDS = "id,doi,title,publication_date,type,authorships,referenced_works," *
    "has_fulltext,abstract_inverted_index,primary_location,locations"

Base.@kwdef mutable struct Work
    id::String                                # OpenAlex ID, e.g. "W4417170000"
    doi::String                               # normalized
    title::String
    date::Date
    type::String
    authors::Vector{Tuple{String, String}}     # (display name, bare ORCID or "")
    refs::Set{String} = Set{String}()         # OpenAlex IDs of referenced works
    has_fulltext::Bool = false                # OpenAlex indexed the full text
    abstract::String = ""
    landing::String = ""
    pdf_urls::Vector{String} = String[]
    origin::Symbol = :team                  # :team, :citing (found through citations) or :mention (full-text search)
    other_dois::Vector{String} = String[]     # other versions merged into this work
end

short_id(url::AbstractString) = String(last(split(url, '/')))

function openalex(path::AbstractString, params)
    params = collect(Pair{String, String}, params)
    key = get(ENV, "OPENALEX_API_KEY", "")
    isempty(key) || push!(params, "api_key" => key)
    mailto = get(ENV, "OPENALEX_MAILTO", "")
    isempty(mailto) || push!(params, "mailto" => mailto)
    return get_json("$OPENALEX/$path?" * query(params))
end

"""
All works matching an OpenAlex `filter`, following cursor pagination. Pages of 50: larger
pages with abstracts take long enough to stall.
"""
function openalex_works(filter::AbstractString; select::AbstractString = WORK_FIELDS, per_page::Int = 50)
    results = []
    cursor = "*"
    while cursor isa AbstractString
        r = openalex("works", ["filter" => filter, "select" => select, "per-page" => string(per_page), "cursor" => cursor])
        isempty(r["results"]) && break
        append!(results, r["results"])
        cursor = get(r["meta"], "next_cursor", nothing)
    end
    return results
end

function abstract_text(inverted)
    inverted isa AbstractDict || return ""
    positions = Tuple{Int, String}[]
    for (word, idxs) in inverted, i in idxs
        push!(positions, (i, word))
    end
    return join(last.(sort!(positions)), " ")
end

function parse_work(x)
    authors = map(x["authorships"]) do a
        orcid = something(get(a["author"], "orcid", nothing), "")
        (string(something(get(a["author"], "display_name", nothing), "")), isempty(orcid) ? "" : short_id(orcid))
    end
    pdfs = String[]
    for loc in something(get(x, "locations", nothing), [])
        url = get(loc, "pdf_url", nothing)
        url isa AbstractString && push!(pdfs, url)
    end
    primary = something(get(x, "primary_location", nothing), Dict())
    return Work(;
        id = short_id(x["id"]), doi = normalize_doi(something(x["doi"], "")),
        title = clean_text(something(x["title"], "")), date = Date(x["publication_date"]),
        type = x["type"], authors, refs = Set(short_id.(something(get(x, "referenced_works", nothing), []))),
        has_fulltext = something(get(x, "has_fulltext", nothing), false),
        abstract = abstract_text(get(x, "abstract_inverted_index", nothing)),
        landing = something(get(primary, "landing_page_url", nothing), ""), pdf_urls = unique!(pdfs)
    )
end

struct Stack
    packages::Vector{Package}
    by_doi::Dict{String, Package}   # DOI of a package paper → package
    by_ref::Dict{String, Package}   # OpenAlex ID of a package paper → package
end

function load_stack(cfg::Config)
    by_doi = Dict(d => p for p in cfg.packages for d in p.dois)
    by_ref = Dict{String, Package}()
    isempty(by_doi) || for x in openalex_works("doi:" * join(keys(by_doi), "|"); select = "id,doi")
        by_ref[short_id(x["id"])] = by_doi[normalize_doi(x["doi"])]
    end
    return Stack(cfg.packages, by_doi, by_ref)
end

"Recent works by the team, plus works by anyone citing a core package paper or (`search_mentions`) naming a core package."
function fetch_works(cfg::Config, stack::Stack; since::Date)
    types = join(cfg.types, "|")
    orcids = join((p.orcid for p in cfg.people), "|")
    works = parse_work.(openalex_works("authorships.author.orcid:$orcids,from_publication_date:$since,type:$types"))
    core = [id for (id, p) in stack.by_ref if p.tier == "core"]
    if cfg.citing_core && !isempty(core)
        known = Set(w.id for w in works)
        for x in openalex_works("cites:$(join(core, "|")),from_publication_date:$since,type:$types")
            w = parse_work(x)
            w.id in known && continue
            w.origin = :citing
            push!(works, w)
        end
    end
    terms = [m for p in cfg.packages if p.tier == "core" for m in p.match if !occursin('/', m)]
    if cfg.search_mentions && !isempty(terms)
        known = Set(w.id for w in works)
        search = join(("\"$t\"" for t in terms), " OR ")
        for x in openalex_works("fulltext.search:$search,from_publication_date:$since,type:$types")
            w = parse_work(x)
            w.id in known && continue
            w.origin = :mention
            push!(works, w)
        end
    end
    return works
end

"Author display names OpenAlex has for `doi`, for DOI metadata without authors."
function openalex_author_names(doi::AbstractString)
    works = openalex_works("doi:$doi"; select = "authorships")
    isempty(works) && return String[]
    return [string(something(a["author"]["display_name"], "")) for a in works[1]["authorships"]]
end

family_names(w::Work) = Set(last(words(name)) for (name, _) in w.authors if !isempty(words(name)))

"Two records of the same paper, e.g. a preprint and its published version."
same_paper(a::Work, b::Work) =
    similar_titles(a.title, b.title) && !isempty(family_names(a) ∩ family_names(b))

is_preprint(w::Work, cfg::Config) = w.type == "preprint" || preprint_server(w.doi, cfg) !== nothing

"""
Drop repository copies and non-papers, merge versions of the same paper (the published
version wins), and skip works already listed or evaluated before.
"""
function select_candidates(works::Vector{Work}, cfg::Config; on_page::Set{String}, seen::Set{String})
    keep = filter(works) do w
        !isempty(w.doi) && !any(startswith(w.doi, p) for p in cfg.ignore_doi_prefixes) &&
            !any(occursin(r, w.title) for r in cfg.ignore_titles)
    end
    unique!(w -> w.doi, keep)
    sort!(keep; by = w -> (is_preprint(w, cfg), w.date))
    merged = Work[]
    for w in keep
        k = findfirst(m -> same_paper(m, w), merged)
        k === nothing ? push!(merged, w) : push!(merged[k].other_dois, w.doi)
    end
    # Versions of listed preprints are handled by the preprint swap, not added again.
    return filter(merged) do w
        !(w.doi in on_page || w.doi in seen || any(d in on_page for d in w.other_dois))
    end
end

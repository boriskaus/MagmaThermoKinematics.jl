#=
Entry style, following the existing page:
- Last, F., Last, F. M., and Last, F. (YYYY). **Title.** Journal, 19(12), 5343–5362. [https://doi.org/…](https://doi.org/…)
- Last, F., and Last, F. (YYYY). **Title.** EGUsphere (preprint). [https://doi.org/…](https://doi.org/…)
=#

"Initials of given names: \"Boris J. P.\" → \"B. J. P.\", \"Jean-Pierre\" → \"J.-P.\"."
function initials(given::AbstractString)
    parts = split(given, r"[\s.]+"; keepempty = false)
    initial(part) = join((string(first(h)) * "." for h in split(part, '-') if !isempty(h)), "-")
    return join((initial(p) for p in parts), " ")
end

function author_name(a)
    family = get(a, "family", nothing)
    family isa AbstractString || return first_string(get(a, "literal", get(a, "name", "")))
    for key in ("non-dropping-particle", "dropping-particle")
        particle = get(a, key, nothing)
        particle isa AbstractString && !isempty(particle) && (family = particle * " " * family)
    end
    given = get(a, "given", nothing)
    return given isa AbstractString && !isempty(given) ? "$family, $(initials(given))" : String(family)
end

const PARTICLES = Set(["de", "del", "della", "der", "di", "du", "da", "dos", "la", "le", "van", "von", "ten", "ter"])

"Citation name from a display name: \"Albert de Montserrat\" → \"de Montserrat, A.\"."
function display_name_citation(name::AbstractString)
    parts = split(strip(name))
    isempty(parts) && return ""
    k = length(parts)
    while k > 1 && lowercase(parts[k - 1]) in PARTICLES
        k -= 1
    end
    family, given = join(parts[k:end], " "), join(parts[1:(k - 1)], " ")
    return isempty(given) ? family : "$family, $(initials(given))"
end

function join_authors(names::AbstractVector)
    length(names) <= 1 && return join(names)
    return join(names[1:(end - 1)], ", ") * ", and " * names[end]
end

"Publication year from CSL metadata, or `nothing`."
function csl_year(csl)
    for key in ("issued", "published-online", "published-print", "created")
        parts = get(something(get(csl, key, nothing), Dict()), "date-parts", nothing)
        parts isa AbstractVector && !isempty(parts) && parts[1] isa AbstractVector &&
            !isempty(parts[1]) && parts[1][1] isa Integer && return Int(parts[1][1])
    end
    return nothing
end

function field(csl, key)
    v = get(csl, key, nothing)
    return v isa Union{AbstractString, Number} ? clean_text(string(v)) : ""
end

function venue(csl, doi::AbstractString, cfg::Config)
    server = preprint_server(doi, cfg)
    server === nothing || return "$server (preprint)"
    v = clean_text(first_string(get(csl, "container-title", "")))
    isempty(v) && (v = field(csl, "publisher"))
    volume, issue = field(csl, "volume"), field(csl, "issue")
    isempty(volume) || (v *= ", " * volume * (isempty(issue) ? "" : "($issue)"))
    number = field(csl, "article-number")
    isempty(number) && (number = replace(field(csl, "page"), "-" => "–"))
    isempty(number) || (v *= ", " * number)
    return v
end

has_authors(csl) = !isempty(something(get(csl, "author", nothing), []))

"""
Markdown list entry for the publication described by CSL-JSON `csl`. Some records lack authors
(e.g. early JuliaCon proceedings); `author_names` (display names) are used then.
"""
function entry_line(csl, cfg::Config; author_names::AbstractVector{<:AbstractString} = String[])
    doi = String(strip(first_string(get(csl, "DOI", ""))))
    fix(name) = isempty(cfg.author_fixes) ? name : replace(name, cfg.author_fixes...)
    names = has_authors(csl) ? [author_name(a) for a in csl["author"]] : display_name_citation.(author_names)
    names = fix.(String.(clean_text.(names)))
    authors = join_authors(names)
    year = something(csl_year(csl), "n.d.")
    title = clean_text(first_string(get(csl, "title", "")))
    occursin(r"[.?!]$", title) || (title *= ".")
    where_ = venue(csl, normalize_doi(doi), cfg)
    endswith(where_, ".") || (where_ *= ".")
    return "- $authors ($year). **$title** $where_ [$(doi_url(doi))]($(doi_url(doi)))"
end

struct Person
    name::String
    orcid::String
    since::Union{Date, Nothing}
    requires_ack::Bool   # their papers count only when they thank the project
end

struct Package
    name::String
    tier::String   # "core" or "generic"
    gpu::Bool      # only counts for GPU work
    match::Vector{String}
    dois::Vector{String}
    patterns::Vector{Regex}
end

struct Config
    people::Vector{Person}
    packages::Vector{Package}
    lookback_months::Int
    types::Vector{String}
    citing_core::Bool
    search_mentions::Bool   # also find works whose indexed full text names a core package
    require_team::Bool      # a team author is needed for ✅ (false: explicit use by anyone counts)
    ignore_doi_prefixes::Vector{String}
    ignore_titles::Vector{Regex}
    preprint_servers::Vector{Pair{String, String}}
    author_fixes::Vector{Pair{String, String}}
    ack_anywhere::Vector{Pair{String, Regex}}    # term => pattern, searched in the whole text
    ack_in_section::Vector{Pair{String, Regex}}  # term => pattern, acknowledgments section only
    ack_weak::Set{String}                       # terms that never promote a paper on their own
end

"Whole-word pattern for a package name, URL or acknowledgment term."
word_pattern(s::AbstractString; flags::AbstractString = "") =
    Regex("(?<![A-Za-z0-9])\\Q" * s * "\\E(?![A-Za-z0-9])", flags)

function load_config(dir::AbstractString = BOTDIR)
    c = TOML.parsefile(joinpath(dir, "config.toml"))
    team = TOML.parsefile(joinpath(dir, "people.toml"))
    people = map(team["person"]) do x
        since = get(x, "since", nothing)
        Person(
            x["name"], x["orcid"], since === nothing ? nothing : Date(string(since)),
            get(x, "requires_acknowledgment", false)
        )
    end
    packages = map(c["package"]) do x
        match = String.(x["match"])
        Package(
            x["name"], x["tier"], get(x, "gpu", false), match,
            normalize_doi.(get(x, "dois", String[])), word_pattern.(match)
        )
    end
    scan = c["scan"]
    servers = [String(k) => String(v) for (k, v) in c["preprint_servers"]]
    sort!(servers; by = s -> -length(first(s)))  # longest prefix wins
    fixes = [String(k) => String(v) for (k, v) in get(c, "author_fixes", Dict())]
    ack = get(c, "acknowledgment", Dict())
    patterns(key) = [String(t) => word_pattern(String(t); flags = "i") for t in get(ack, key, String[])]
    return Config(
        people, packages, scan["lookback_months"], String.(scan["types"]),
        scan["citing_core"], get(scan, "search_mentions", false), get(scan, "require_team", true),
        String.(scan["ignore_doi_prefixes"]),
        [Regex(r, "i") for r in scan["ignore_titles"]], servers, fixes,
        patterns("anywhere"), patterns("in_section"),
        Set(String.(get(ack, "weak", String[])))
    )
end

"Name of the preprint server hosting `doi`, or `nothing` for other DOIs."
function preprint_server(doi::AbstractString, cfg::Config)
    for (prefix, name) in cfg.preprint_servers
        startswith(doi, prefix) && return name
    end
    return nothing
end

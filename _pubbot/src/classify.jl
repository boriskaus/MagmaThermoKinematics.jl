#=
Policy: list papers that use the Julia stack GPU4GEO works on.

"Team" is everyone listed in people.toml.

✅ :strong  a team author, and a package of the stack, core or generic (Trixi.jl only for GPU work)
🟡 :review  stack evidence that does not meet that bar (e.g. an outside paper citing a core
           package), or a team paper about Julia on GPUs without a package named, or a team
           paper mentioning Julia/GPUs whose full text could not be checked
❌ :none    everything else

With `require_team = false` (config.toml), a core package named in the title, abstract or full
text (or the package paper itself) is ✅ for any author; a citation of a package paper alone is 🟡. Julia
and GPU keywords without a package then count for nothing.

A team paper that thanks the project by name (GPU4GEO, c44) is ✅ whatever else it shows; a
term that funds many projects (PASC) is only worth 🟡. Papers whose team authors all have
`requires_acknowledgment` and thank nobody drop to 🟡 (core package or no full text) or ❌.
=#

const GPU_RE = r"\bGPUs?\b|[Gg]raphics [Pp]rocessing [Uu]nits?"
const JULIA_RE = r"\bJulia\b"
# Papers merely citing GPU work or mentioning it in passing have 0-2 mentions; GPU papers dozens.
const MIN_GPU_MENTIONS = 3

"The paper without its reference list, so cited titles do not count as keywords."
function body_text(text::AbstractString)
    headings = collect(eachmatch(r"\b(References|REFERENCES|Bibliography)\b", text))
    isempty(headings) && return text
    at = last(headings).offset
    return at > ncodeunits(text) ÷ 2 ? text[1:prevind(text, at)] : text
end

"Text following each acknowledgments, funding or financial support heading."
function funding_sections(text::AbstractString, span::Int = 1500)
    io = IOBuffer()
    for m in eachmatch(r"\b(Acknowledg\w*|Funding|Financial support|Grant support)\b"i, text)
        stop = m.offset + span
        print(io, SubString(text, m.offset, stop < ncodeunits(text) ? thisind(text, stop) : lastindex(text)), " ")
    end
    return String(take!(io))
end

"""
The acknowledgment term the paper uses, `\"\"` when it thanks nobody we look for, or `nothing`
when there is no full text to check.
"""
function acknowledgment(text::Union{String, Nothing}, cfg::Config)
    text === nothing && return nothing
    for (term, pattern) in cfg.ack_anywhere
        occursin(pattern, text) && return term
    end
    section = funding_sections(text)
    for (term, pattern) in cfg.ack_in_section
        occursin(pattern, section) && return term
    end
    return ""
end

struct Evidence
    package::Package
    source::String   # "package paper", "cites package paper", "title/abstract", "full text", "OpenAlex full-text index"
end

Base.@kwdef mutable struct Assessment
    work::Work
    team::Vector{Person} = Person[]
    evidence::Vector{Evidence} = Evidence[]
    gpu::Bool = false
    julia::Bool = false
    acknowledgment::Union{String, Nothing} = nothing  # term used, "" none found, nothing not checked
    fulltext::Symbol = :unavailable   # :pdf, :index or :unavailable
    bucket::Symbol = :none            # :strong, :review or :none
    entry::Union{String, Nothing} = nothing
    year::Union{Int, Nothing} = nothing
    note::String = ""
end

"Team members among the authors, matched by ORCID or else by first initial and last name."
function team_authors(w::Work, cfg::Config)
    key(name) = (ws = words(name); isempty(ws) ? ("", "") : (first(ws)[1:1], last(ws)))
    found = Person[]
    for (name, orcid) in w.authors
        k = findfirst(p -> !isempty(orcid) && p.orcid == orcid, cfg.people)
        k === nothing && (k = findfirst(p -> key(p.name) == key(name), cfg.people))
        k === nothing && continue
        p = cfg.people[k]
        (p.since === nothing || w.date >= p.since) && !(p in found) && push!(found, p)
    end
    return found
end

"""
    assess(w, cfg, stack, text, probes)

Classify `w` given its PDF text (`nothing` if unavailable) and OpenAlex full-text search results.
"""
function assess(w::Work, cfg::Config, stack::Stack, text::Union{String, Nothing}, probes::Probes = Probes())
    a = Assessment(; work = w, team = team_authors(w, cfg))
    add!(p, source) = any(e -> e.package === p && e.source == source, a.evidence) || push!(a.evidence, Evidence(p, source))

    haskey(stack.by_doi, w.doi) && add!(stack.by_doi[w.doi], "package paper")
    for r in w.refs
        haskey(stack.by_ref, r) && add!(stack.by_ref[r], "cites package paper")
    end
    head = w.title * " " * w.abstract
    for p in stack.packages, pat in p.patterns
        occursin(pat, head) && add!(p, "title/abstract")
        text !== nothing && occursin(pat, text) && add!(p, "full text")
    end
    a.gpu = occursin(GPU_RE, head)
    a.julia = occursin(JULIA_RE, head)
    if text !== nothing
        a.fulltext = :pdf
        body = body_text(text)
        a.gpu |= count(GPU_RE, body) >= MIN_GPU_MENTIONS
        # "Julia" alone may be an author's first name.
        a.julia |= occursin(JULIA_RE, body) && occursin(r"\.jl\b", body)
    elseif w.id in probes.searched
        a.fulltext = :index
        for (group, ids) in probes.packages
            w.id in ids && add!(group, "OpenAlex full-text index")
        end
        a.gpu |= w.id in probes.gpu
        a.julia |= w.id in probes.julia
    end

    has_team = !isempty(a.team)
    counted = filter(e -> !e.package.gpu || a.gpu, a.evidence)  # Trixi.jl needs GPU work
    core = any(e -> e.package.tier == "core", counted)
    # OpenAlex's index drops punctuation, so "CUDA.jl" may match plain "CUDA": not enough alone.
    generic = any(e -> e.package.tier == "generic" && e.source != "OpenAlex full-text index", counted)
    # Named, not only cited: the authors say they use the package.
    named = any(e -> e.package.tier == "core" && e.source != "cites package paper", counted)
    a.bucket = if has_team && (core || generic) || !cfg.require_team && named
        :strong
    elseif !isempty(counted) || cfg.require_team && (
            (has_team && a.julia && a.gpu) ||
                (has_team && a.fulltext == :unavailable && (a.julia || a.gpu))
        )
        :review
    else
        :none
    end

    a.acknowledgment = acknowledgment(text, cfg)
    # Thanking the project is attribution by the authors themselves: enough on its own, unless
    # the term is one that funds many projects (`weak` in config.toml).
    if has_team && acknowledged(a)
        a.bucket = a.acknowledgment in cfg.ack_weak ? max_bucket(a.bucket, :review) : :strong
        return a
    end
    # People whose papers need an explicit acknowledgment, unless someone else on the team
    # co-authored the paper (people.toml).
    if has_team && all(p -> p.requires_ack, a.team) && a.bucket != :none
        if a.acknowledgment === nothing
            a.note = "acknowledgment could not be checked: no full text"
            a.bucket = :review
        else
            a.note = "no acknowledgment of the project found"
            a.bucket = core ? :review : :none   # a core package is worth a look anyway
        end
    end
    return a
end

acknowledged(a::Assessment) = a.acknowledgment isa String && !isempty(a.acknowledgment)

const BUCKET_ORDER = Dict(:none => 0, :review => 1, :strong => 2)
max_bucket(a::Symbol, b::Symbol) = BUCKET_ORDER[a] >= BUCKET_ORDER[b] ? a : b

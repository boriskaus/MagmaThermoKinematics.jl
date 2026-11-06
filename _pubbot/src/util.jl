# Crossref's recommended DOI pattern, without parentheses so markdown links parse cleanly.
const DOI_RE = r"10\.\d{4,9}/[-._;/:A-Za-z0-9]+"

"Bare lowercase DOI, without resolver prefix or trailing punctuation."
function normalize_doi(s::AbstractString)
    d = replace(strip(s), r"^(https?://)?(dx\.)?doi\.org/"i => "", r"^doi:\s*"i => "")
    return lowercase(rstrip(d, ['.', ',', ';', ':']))
end

"All DOIs found in `text`, normalized, in order of appearance."
extract_dois(text::AbstractString) = unique!([normalize_doi(m.match) for m in eachmatch(DOI_RE, text)])

doi_url(doi::AbstractString) = "https://doi.org/" * doi

"Collapse whitespace and drop HTML/JATS tags from metadata strings."
function clean_text(s::AbstractString)
    s = replace(s, r"<[^>]+>" => "")
    s = replace(s, "&amp;" => "&", "&lt;" => "<", "&gt;" => ">", "&quot;" => "\"", "‐" => "-", "‑" => "-")
    return strip(replace(s, r"\s+" => " "))
end

"First string of a metadata field that may be a string, a list or missing."
function first_string(x)
    x isa AbstractString && return String(x)
    x isa AbstractVector && !isempty(x) && x[1] isa AbstractString && return String(x[1])
    return ""
end

"Accent-free lowercase words, for fuzzy comparisons."
function words(s::AbstractString)
    t = Unicode.normalize(String(s); stripmark = true, casefold = true)
    return [m.match for m in eachmatch(r"[a-z0-9]+", t)]
end

"Title words, with British spelling mapped to American (\"modelling\" → \"modeling\")."
function title_words(s::AbstractString)
    american(w) = replace(w, r"ll(ing|ed)$" => s"l\1", r"is(e|ed|es|ing|ation|ations)$" => s"iz\1", r"our$" => "or")
    return Set(american.(words(s)))
end

"""
Similarity of two titles in [0, 1] (Jaccard index over words). Titles whose numbers or
roman numerals differ ("Part I" / "Part II", "v1.0" / "v2.0") never match.
"""
function title_similarity(a::AbstractString, b::AbstractString)
    wa, wb = title_words(a), title_words(b)
    (isempty(wa) || isempty(wb)) && return 0.0
    numbers(ws) = filter(w -> occursin(r"^(\d+|i|ii|iii|iv)$", w), ws)
    na, nb = numbers(wa), numbers(wb)
    isempty(na) || isempty(nb) || na == nb || return 0.0
    return length(wa ∩ wb) / length(wa ∪ wb)
end

"""
Titles of two versions of a paper: nearly the same words, or one title extending the other
(the published version often gains a suffix like "with DEDLoc (version 1.0.0)").
"""
function similar_titles(a::AbstractString, b::AbstractString)
    title_similarity(a, b) >= 0.8 && return true
    wa, wb = title_words(a), title_words(b)
    shorter = min(length(wa), length(wb))
    return shorter >= 6 && title_similarity(a, b) > 0 && length(wa ∩ wb) >= 0.9 * shorter
end

"Append `key=value` to the GitHub Actions step outputs, when running in Actions."
function set_output(key, value)
    path = get(ENV, "GITHUB_OUTPUT", "")
    isempty(path) && return
    return open(io -> println(io, "$key=$value"), path, "a")
end

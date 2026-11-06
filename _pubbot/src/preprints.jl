"DOI of the published version Crossref records for a preprint, or `nothing`."
function crossref_published_version(doi::AbstractString)
    params = ["filter" => "relation.type:has-preprint,relation.object:$doi", "select" => "DOI", "rows" => "5"]
    items = try
        get_json("https://api.crossref.org/works?" * query(params))["message"]["items"]
    catch err
        @warn "Crossref lookup failed" doi exception = err
        return nothing
    end
    return isempty(items) ? nothing : normalize_doi(items[1]["DOI"])
end

"""
Demote proposed preprints whose published version is already listed, evaluated before or
proposed in this scan; titles can differ too much for `select_candidates` to merge them.
"""
function drop_published_preprints!(
        assessments::Vector{Assessment}, cfg::Config;
        on_page::Set{String}, seen::Set{String}
    )
    known = on_page ∪ seen ∪ Set(a.work.doi for a in assessments if a.bucket != :none)
    for a in assessments
        a.bucket != :none && is_preprint(a.work, cfg) || continue
        published = crossref_published_version(a.work.doi)
        (published === nothing || !(published in known)) && continue
        a.bucket = :none
        a.note = "preprint of $published"
    end
    return assessments
end

"""
Preprints listed on the page that now have a published version: `old DOI => new DOI`.
Uses Crossref relations (Copernicus), else a title match among the scanned works (arXiv, …).
Swaps whose target was already proposed once (in `seen`) are not proposed again.
"""
function published_versions(on_page::Set{String}, seen::Set{String}, cfg::Config, works::Vector{Work})
    preprints = sort!([d for d in on_page if preprint_server(d, cfg) !== nothing])
    isempty(preprints) && return Pair{String, String}[]
    titles = Dict{String, String}()
    for chunk in Iterators.partition(preprints, 50)
        for x in openalex_works("doi:" * join(chunk, "|"); select = "doi,title")
            titles[normalize_doi(x["doi"])] = clean_text(something(x["title"], ""))
        end
    end
    swaps = Pair{String, String}[]
    for d in preprints
        new = crossref_published_version(d)
        if new === nothing && haskey(titles, d)
            k = findfirst(works) do w
                !isempty(w.doi) && !is_preprint(w, cfg) &&
                    !any(startswith(w.doi, p) for p in cfg.ignore_doi_prefixes) &&
                    similar_titles(w.title, titles[d])
            end
            k === nothing || (new = works[k].doi)
        end
        new === nothing || new in on_page || new in seen || push!(swaps, d => new)
    end
    return swaps
end

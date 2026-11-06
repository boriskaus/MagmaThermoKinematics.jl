function load_seen(path::AbstractString = SEEN)
    isfile(path) || return Set{String}()
    return Set(normalize_doi(l) for l in eachline(path) if !isempty(strip(l)) && !startswith(l, "#"))
end

function save_seen(path::AbstractString, seen::Set{String})
    return open(path, "w") do io
        println(io, "# DOIs already evaluated by the publication bot (_pubbot); they are not proposed again.")
        println(io, "# Delete a line to have that DOI evaluated again by the next scan.")
        foreach(d -> println(io, d), sort!(collect(seen)))
    end
end

"Fill in the formatted entry; demote to review when metadata cannot be fetched."
function add_entry!(a::Assessment, cfg::Config)
    csl = get_csl(a.work.doi)
    if csl === nothing
        a.note = "DOI metadata unavailable, add manually"
        a.bucket == :strong && (a.bucket = :review)
        return a
    end
    a.entry = entry_line(csl, cfg; author_names = first.(a.work.authors))
    a.year = something(csl_year(csl), year(a.work.date))
    return a
end

"Entry for `doi`, taking authors from OpenAlex when the DOI metadata has none."
entry_for(csl, doi, cfg::Config) =
    entry_line(csl, cfg; author_names = has_authors(csl) ? String[] : openalex_author_names(doi))

"""
    weekly(; since, dry_run, report_path, page_path, seen_path)

Scan for new publications, edit `publications.md` and `seen_dois.txt` (unless `dry_run`),
and write the pull request description to `report_path` (stdout if `nothing`).
"""
function weekly(;
        since::Union{Date, Nothing} = nothing, dry_run::Bool = false,
        report_path::Union{AbstractString, Nothing} = nothing,
        page_path::AbstractString = PAGE, seen_path::AbstractString = SEEN
    )
    cfg = load_config()
    since = something(since, today() - Month(cfg.lookback_months))
    page = load_page(page_path)
    on_page = page_dois(page)
    seen = load_seen(seen_path)

    stack = load_stack(cfg)
    works = fetch_works(cfg, stack; since)
    candidates = select_candidates(works, cfg; on_page, seen)
    @info "Fetched works" since records = length(works) candidates = length(candidates)

    texts = fulltexts(candidates)
    unindexed = [w.id for (w, t) in zip(candidates, texts) if t === nothing && w.has_fulltext]
    probes = fulltext_probes(stack, unindexed)
    @info "Full text" pdf = count(!isnothing, texts) index = length(unindexed)
    assessments = [assess(w, cfg, stack, t, probes) for (w, t) in zip(candidates, texts)]
    drop_published_preprints!(assessments, cfg; on_page, seen)
    sort!(assessments; by = a -> a.work.date, rev = true)
    foreach(a -> a.bucket == :none || add_entry!(a, cfg), assessments)

    swaps = NamedTuple[]
    for (old, new) in published_versions(on_page, seen, cfg, works)
        csl = get_csl(new)
        csl === nothing && continue
        push!(
            swaps, (;
                old, new, title = clean_text(first_string(get(csl, "title", ""))),
                entry = entry_for(csl, new, cfg), year = something(csl_year(csl), year(today())),
            )
        )
    end

    strong = filter(a -> a.bucket == :strong, assessments)
    # Oldest first, so the newest paper ends up at the top of its year.
    for a in Iterators.reverse(strong)
        insert_entry!(page, a.year, a.entry)
    end
    for s in swaps
        remove_entry!(page, s.old)
        insert_entry!(page, s.year, s.entry)
    end

    changed = !isempty(swaps) || any(a -> a.bucket != :none, assessments)
    if changed && !dry_run
        save_page(page)
        for a in assessments
            push!(seen, a.work.doi)
            union!(seen, a.work.other_dois)
        end
        foreach(s -> push!(seen, s.new), swaps)
        save_seen(seen_path, seen)
    end
    report = weekly_report(assessments, swaps, cfg; since, nworks = length(works))
    report_path === nothing ? print(report) : write(report_path, report)
    set_output("changed", changed && !dry_run)
    return (; assessments, swaps, changed)
end

"""
    add_dois(dois; issue, report_path, page_path, seen_path)

Format and insert the publications with the given DOIs.
"""
function add_dois(
        dois::AbstractVector{<:AbstractString}; issue = nothing,
        report_path::Union{AbstractString, Nothing} = nothing,
        page_path::AbstractString = PAGE, seen_path::AbstractString = SEEN
    )
    cfg = load_config()
    page = load_page(page_path)
    on_page = page_dois(page)
    seen = load_seen(seen_path)
    added, skipped = Tuple{String, String}[], Tuple{String, String}[]
    for doi in unique(normalize_doi.(dois))
        if doi in on_page
            push!(skipped, (doi, "already listed"))
            continue
        end
        csl = get_csl(doi)
        if csl === nothing
            push!(skipped, (doi, "DOI does not resolve"))
            continue
        end
        entry = entry_for(csl, doi, cfg)
        insert_entry!(page, something(csl_year(csl), year(today())), entry)
        push!(on_page, doi)
        push!(seen, doi)
        push!(added, (doi, entry))
    end
    changed = !isempty(added)
    if changed
        save_page(page)
        save_seen(seen_path, seen)
    end
    report = add_report(added, skipped; issue)
    report_path === nothing ? print(report) : write(report_path, report)
    set_output("changed", changed)
    return (; added, skipped, changed)
end

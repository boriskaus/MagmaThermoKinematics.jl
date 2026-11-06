#=
The publication list is the part of publications.md between the front matter and the first
level-2 heading (`## Conferences & Events`, which has its own `### YYYY` headings).
Entries are `- ` lines under `### YYYY` headings, newest year first.
=#

mutable struct Page
    path::String
    lines::Vector{String}
end

load_page(path::AbstractString = PAGE) = Page(path, readlines(path))
save_page(p::Page) = write(p.path, join(p.lines, "\n") * "\n")

function list_range(p::Page)
    start = 1
    if !isempty(p.lines) && p.lines[1] == "+++"
        close = findnext(==("+++"), p.lines, 2)
        close === nothing || (start = close + 1)
    end
    stop = findnext(l -> startswith(l, "## "), p.lines, start)
    return start:(stop === nothing ? length(p.lines) : stop - 1)
end

page_dois(p::Page) = Set(extract_dois(join(p.lines[list_range(p)], "\n")))

function year_heading(line::AbstractString)
    m = match(r"^###\s+(\d{4})\s*$", line)
    return m === nothing ? nothing : parse(Int, m[1])
end

is_entry(line::AbstractString) = startswith(line, "- ")

"Insert `entry` at the top of the `### year` section, creating the section if needed."
function insert_entry!(p::Page, year::Integer, entry::AbstractString)
    r = list_range(p)
    for i in r
        y = year_heading(p.lines[i])
        y === nothing && continue
        if y == year
            k = i + 1
            k <= last(r) && isempty(strip(p.lines[k])) && (k += 1)
            block = k <= last(r) && is_entry(p.lines[k]) ? [entry] : [entry, ""]
            splice!(p.lines, k:(k - 1), block)
            return p
        elseif y < year
            splice!(p.lines, i:(i - 1), ["### $year", "", entry, ""])
            return p
        end
    end
    # Oldest year so far: append at the end of the list.
    k = last(r) + 1
    block = ["### $year", "", entry, ""]
    k > 1 && !isempty(strip(p.lines[k - 1])) && pushfirst!(block, "")
    splice!(p.lines, k:(k - 1), block)
    return p
end

"Remove the entry citing `doi`, and its year heading if the section becomes empty."
function remove_entry!(p::Page, doi::AbstractString)
    r = list_range(p)
    i = findfirst(k -> is_entry(p.lines[k]) && doi in extract_dois(p.lines[k]), collect(r))
    i === nothing && return false
    deleteat!(p.lines, r[i])
    remove_empty_sections!(p)
    return true
end

function remove_empty_sections!(p::Page)
    r = list_range(p)
    heads = [i for i in r if year_heading(p.lines[i]) !== nothing]
    for (n, h) in Iterators.reverse(collect(enumerate(heads)))
        stop = n < length(heads) ? heads[n + 1] - 1 : last(r)
        any(is_entry, p.lines[(h + 1):stop]) && continue
        # Drop the heading and the blank lines that follow it.
        e = h + 1
        while e <= stop && isempty(strip(p.lines[e]))
            e += 1
        end
        deleteat!(p.lines, h:(e - 1))
    end
    return p
end

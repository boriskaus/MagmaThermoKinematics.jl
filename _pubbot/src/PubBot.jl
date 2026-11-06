"""
Keeps the Publications page of the documentation (`docs/src/man/publications.md`) up to date
with papers that use MagmaThermoKinematics.jl.

Adapted from the publication bot of GPU4GEO (https://github.com/PTsolvers/GPU4GEO, MIT
license), with the `require_team` and `search_mentions` options added (see `config.toml`).

- `weekly`: scans OpenAlex for recent works that mention or cite the packages listed in
  `config.toml`, and by the people in `people.toml`, and edits the Publications page.
- `add_dois`: formats and inserts publications from a list of DOIs.
"""
module PubBot

using Dates, Downloads, JSON, TOML, Unicode
using Poppler_jll: pdftotext

const BOTDIR = dirname(@__DIR__)
const ROOT = dirname(BOTDIR)
const PAGE = joinpath(ROOT, "docs", "src", "man", "publications.md")
const SEEN = joinpath(BOTDIR, "seen_dois.txt")

include("util.jl")
include("config.jl")
include("http.jl")
include("page.jl")
include("format.jl")
include("openalex.jl")
include("fulltext.jl")
include("classify.jl")
include("preprints.jl")
include("report.jl")
include("main.jl")

end

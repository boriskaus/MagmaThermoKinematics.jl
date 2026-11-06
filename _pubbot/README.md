# Publication bot

Keeps the [Publications](../docs/src/man/publications.md) page of the documentation up to date with
papers that use MagmaThermoKinematics.jl. Adapted from the publication bot of
[GPU4GEO](https://github.com/PTsolvers/GPU4GEO) (MIT license); the options `require_team` and
`search_mentions` in [config.toml](config.toml) were added for a package whose users are mostly
not its developers.

Two GitHub workflows use it:

- **Scan** ([PubScan.yml](../.github/workflows/PubScan.yml)): on the 1st and 15th of each month,
  finds recent papers that use the package and opens a pull request that adds them. A scan is
  skipped while the previous pull request is open.
- **Add a publication** issue form ([PubAdd.yml](../.github/workflows/PubAdd.yml)): repository
  collaborators paste DOIs, and a pull request with the formatted entries is opened.

## How papers are selected

1. [OpenAlex](https://openalex.org) works published within the last `lookback_months` that
   - name the package in their full text (OpenAlex's full-text index, `search_mentions`),
   - cite the package paper (`dois` in `config.toml`: Schmitt et al., 2023), or
   - are written by the developers in [people.toml](people.toml).
2. Repository copies, peer-review comments and duplicates are dropped; a preprint and its published
   version are merged. DOIs already on the page or listed in `seen_dois.txt` are skipped.
3. Evidence of use is collected from the open-access PDF (Copernicus, arXiv, JOSS, HAL), or from the
   OpenAlex full-text index when the publisher blocks downloads, and from the title and abstract.
4. Each paper is classified:
   - ✅ **added**: names MagmaThermoKinematics in the full text, title or abstract, whoever the
     authors are;
   - 🟡 **needs review**: only cites the package paper (it is also cited for its zircon age spectra),
     listed in the pull request with a ready-to-paste entry;
   - ❌ **no signal**: listed collapsed in the pull request.
5. Preprints on the page whose published version appeared are replaced.

Merging a scan pull request records every evaluated DOI in `seen_dois.txt`, so nothing is proposed
twice. To drop an added entry, delete its line in the pull request before merging; to add a paper
from the review list, paste its entry under the right year.

## Maintenance

- **Developers join or leave**: edit `people.toml` (`since = "YYYY-MM-DD"` ignores older papers).
- **A new package paper** (e.g. the JOSS paper): add its DOI to `dois` in `config.toml`.
- **Misspelled author in publisher metadata**: add it to `[author_fixes]` in `config.toml`.
- **Re-evaluate a paper**: delete its DOI from `seen_dois.txt`.

## Running locally

```sh
julia --project=_pubbot -e 'using Pkg; Pkg.instantiate()'
julia --project=_pubbot _pubbot/weekly.jl --dry-run            # print the report only
julia --project=_pubbot _pubbot/add_doi.jl 10.1038/s41467-024-55443-z
julia --project=_pubbot -e 'using Pkg; Pkg.test()'
```

The tests of the selection engine use the GPU4GEO configuration in `test/gpu4geo`; the
"MagmaThermoKinematics.jl" testset uses the real one. PDF text is cached in `_pubbot/.cache`
(git-ignored).

## OpenAlex quota

OpenAlex meters its API: without a key, 1000 credits per day per IP address, shared with everyone
else on the same GitHub runner. A list request costs 1 credit and a full-text search 10. Each scan
uses a few dozen credits. Get a free API key at <https://openalex.org> and store it as the
repository secret `OPENALEX_API_KEY`; locally, export the same variable.

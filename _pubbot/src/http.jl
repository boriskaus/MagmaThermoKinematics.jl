const USER_AGENT = "GPU4GEO-pubbot/0.1 (+https://github.com/PTsolvers/GPU4GEO)"
const DOWNLOADER = Ref{Union{Downloads.Downloader, Nothing}}(nothing)

# Curl gives up after 20 s without data by default; OpenAlex can pause longer on large pages.
function downloader()
    if DOWNLOADER[] === nothing
        d = Downloads.Downloader()
        d.easy_hook = (easy, _) -> Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_TIME, 60)
        DOWNLOADER[] = d
    end
    return DOWNLOADER[]
end

"""
    http_get(url; headers, method, timeout, retries) -> (status, body, final_url)

Follows redirects and retries on rate limiting, server errors and network failures.
Returns status 0 when the request could not be made at all.
"""
function http_get(
        url::AbstractString; headers = Pair{String, String}[], method = "GET",
        timeout = 60, retries = 3
    )
    hdrs = ["User-Agent" => USER_AGENT; headers]
    for attempt in 1:retries
        io = IOBuffer()
        resp = try
            Downloads.request(url; method, output = io, headers = hdrs, timeout, throw = false, downloader = downloader())
        catch err
            err
        end
        if resp isa Downloads.Response
            retry = resp.status == 429 || resp.status >= 500
            retry && attempt < retries && (sleep(2^attempt); continue)
            return (resp.status, take!(io), resp.url)
        end
        attempt < retries && (sleep(2^attempt); continue)
        @warn "Request failed" url exception = resp
    end
    return (0, UInt8[], String(url))
end

function get_json(url::AbstractString; headers = Pair{String, String}[])
    status, body, _ = http_get(url; headers)
    status == 200 || error("GET $url returned status $status")
    return JSON.parse(String(body))
end

function urlencode(s::AbstractString)
    io = IOBuffer()
    for b in codeunits(s)
        if UInt8('A') <= b <= UInt8('Z') || UInt8('a') <= b <= UInt8('z') ||
                UInt8('0') <= b <= UInt8('9') || b in (UInt8('-'), UInt8('_'), UInt8('.'), UInt8('~'))
            write(io, b)
        else
            print(io, '%', uppercase(string(b; base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

query(params) = join((k * "=" * urlencode(string(v)) for (k, v) in params), "&")

"CSL-JSON metadata for a DOI (Crossref, DataCite, …), or `nothing` if it does not resolve."
function get_csl(doi::AbstractString)
    status, body, _ = http_get(doi_url(doi); headers = ["Accept" => "application/vnd.citationstyles.csl+json"])
    status == 200 || return nothing
    csl = try
        JSON.parse(String(body))
    catch
        return nothing
    end
    return csl isa AbstractDict ? csl : nothing
end

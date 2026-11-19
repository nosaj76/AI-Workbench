# Web search

Chat models search the web through the `web_search` tool. Workbench uses, in order: an enabled MCP search server, a configured SearXNG URL, then a local text browser (lynx, w3m, elinks or links) querying DuckDuckGo, then public knowledge sources (Wikipedia and Hacker News) when they are allowed. Wikipedia, arXiv, Crossref, Hacker News and GitHub are also available as search modes. Every query shows an approval preview unless you turn on automatic approval for web actions.

## One result format

Every provider returns the same shape, so a model reads results the same way
whichever provider answered:

```json
{
  "ok": true,
  "query": "rust async runtime",
  "limit": 10,
  "provider": {"type": "searxng", "server": "http://127.0.0.1:18080"},
  "result_count": 2,
  "results": [{"title": "…", "url": "https://…", "snippet": "…"}],
  "search_options": {"requested": {"time_range": "week"}, "applied": ["time_range"], "not_applied": []},
  "local_sites": {"result_count": 1, "results": [{"title": "…", "url": "…", "snippet": "…"}]}
}
```

`provider.type` is `mcp`, `searxng`, `text_browser`, `knowledge` or
`knowledge_fallback`. `failed_providers`/`failed_browsers` list providers that
were tried first and failed. `local_sites` (your own home-server sites, listed
before web results) appears with every web provider, including MCP. An MCP reply
is converted from its `structuredContent`, JSON text, "Title:/URL:/Description:"
text, Markdown links or bare URLs; when none of these can be read, its text is
returned in `provider_text` instead of results.

## Search options

`web_search` accepts three optional refinements:

| Option | Values | SearXNG | DuckDuckGo (text browser) | MCP | Knowledge sources |
| --- | --- | --- | --- | --- | --- |
| `time_range` | `day`, `week`, `month`, `year` | `time_range` | `df` | if mapped | not applied |
| `language` | `en`, `de`, `en-GB`, `pt-BR`… | `language` | `kl`, only with a region (`en-US` → `us-en`) | if mapped | not applied |
| `safe_search` | `off`, `moderate`, `strict` | `safesearch` 0/1/2 | `kp` -2/-1/1 | if mapped | not applied |

Each result's `search_options.not_applied` names any option the answering
provider could not apply, so the model does not claim a filter was used.

Turn on **Settings → Connection and privacy → Enable explicit web fetch and search tools** first.

## Search presets: Brave, Exa and Kagi

Three official search MCP servers can be added with one click. All start
disabled and need your own API key from the provider.

| Preset | Package (pinned) | Started with | Key | Time range | Language | Safe search | Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brave Search | `@brave/brave-search-mcp-server@2.1.4` | `npx` (Node.js) | `BRAVE_API_KEY` | day–year | yes | yes | no |
| Exa | `exa-mcp-server@3.4.3` | `npx` (Node.js) | `EXA_API_KEY` | no | no | no | no |
| Kagi | `kagimcp==1.1.0` | `uvx` (uv) | `KAGI_API_KEY` | day–month | no | no | yes |

1. Open **Settings → Connection and privacy** and enable **explicit web fetch
   and search tools**.
2. Open **MCP servers and web search** and select **Add or repair … preset**
   for the provider. Repairing keeps your key and on/off choice.
3. Enter the API key (**Get a key** opens the provider's page) and enable the
   provider. Enabling it also turns on the global MCP and network switches.
4. Save the settings, start a chat and ask `search the web about Linux`.
5. Inspect and approve the exact query. On first use, the preview also shows the
   pinned `npx` or `uvx` launch that may download the package.

Install Node.js (for `npx`) or uv (for `uvx`) first; Settings warns when the
launcher is missing. No project folder is required. Native tool calling still
requires a compatible installed Ollama model.

## Image search

`web_search` with `mode: "images"` returns pictures: each result has the page
`url`, `image_url` and, when available, a `thumbnail`. It uses an MCP server set
up for images (the Kagi preset) or SearXNG's image category. DuckDuckGo through
a text browser and the public knowledge sources cannot search images, so an
image search fails with a clear message when neither is set up.

## Related searches

SearXNG's suggested searches, and any `suggestions` or `related_searches` an
MCP server returns, appear in each result as `suggestions`. In Chat, the
suggestions from the latest answer's searches are shown under it as **Related
searches**; clicking one puts "Search the web for …" in the message box.

## Search depth

**Settings → Connection and privacy** (or **Options** in Chat) sets how
thoroughly chat searches:

| Depth | What the model is asked to do | Results per search |
| --- | --- | --- |
| Speed | One search; answer from the snippets | 5 |
| Balanced (default) | One or two searches; read the two or three best pages | 8 |
| Quality | Three to five searches; read and compare four or more sources | 12 |

An explicit `limit` in a search always wins over the default count.

## Query rewriting

The Q&A lookup searches with a short keyword query rather than the whole
question: the model writes it ("How do I rename a local git branch?" →
`rename local git branch`), and filler words are removed instead if the model
fails. The answer still uses the full question. Turn it off with **Turn Q&A
questions into short search keywords first**. In chat, the `web_search` tool
asks the model itself for concise keywords.

## Runtime and privacy boundaries

- The supplied preset needs a compatible Node/npm `npx` and a separately obtained
  Brave Search API key. Workbench does not install Node or create an account.
- The first approved `npx` launch may contact the npm registry if the exact pinned
  package is not cached.
- The MCP child has the user's normal OS permissions. Workbench clears inherited
  environment variables, then supplies only a small operating environment and
  explicitly configured entries.
- The Brave key is stored in local `settings.json`. Settings are written atomically
  with private Unix permissions, but the file remains a credential-bearing asset
  that must not be placed in shared source archives.
- Search text is sent to the selected search provider. Each later `fetch_url`
  approval may contact another host and follow validated redirects.
- Search snippets and fetched pages are untrusted data, not instructions.
- Snippets are discovery leads. The runtime asks the model to fetch relevant pages
  before making strong factual claims.
- Merely opening Settings or Diagnostics never starts a provider, downloads a
  package, sends a query, or fetches a page.

## Custom MCP search mapping

Any compatible stdio MCP server can be presented as native `web_search`:

```json
[
  {
    "id": "my-search",
    "name": "My Search MCP",
    "program": "/absolute/path/to/search-mcp",
    "args": [],
    "cwd": null,
    "env": {},
    "enabled": true,
    "timeout_secs": 60,
    "web_search": {
      "tool": "web_search",
      "query_argument": "query",
      "limit_argument": "limit",
      "extra_arguments": {
        "safe_search": true
      },
      "max_query_chars": 2048,
      "max_query_words": null,
      "max_results": 10,
      "time_range_argument": "freshness",
      "time_range_values": {"day": "pd", "week": "pw", "month": "pm", "year": "py"},
      "language_argument": "search_lang",
      "region_argument": "country",
      "safe_search_argument": "safesearch",
      "safe_search_values": {}
    }
  }
]
```

The mapped tool and argument names must be nonempty and contain no control
characters. Fixed arguments cannot replace the dynamic query or limit fields and
are size-bounded. Query/result bounds are validated locally. Set
`limit_argument` to `null` when the provider accepts no count. The server must
advertise the configured tool after initialization.

The six option fields are optional. Each `*_argument` names the provider's
argument for that option; leave it out and the option is reported as not
applied. `language_argument` receives the language code (`en` for `en-GB`) and
`region_argument` the region (`GB`). The `*_values` maps translate the standard
words into the provider's values; an empty map sends the standard word. An
option argument may share its name with a fixed argument, whose value is then
the default, but never with the query or limit argument. The Brave preset uses
the mapping shown above; use **Add or repair Brave Search preset** to update an
older saved copy. A `null` value marks a word the provider does not support
(Kagi maps `"year": null`): nothing is sent and the option is reported as not
applied.

For image search, add `"image_search": {"tool": null, "extra_arguments":
{"workflow": "images"}}`: `tool` names a different image tool, or `null` to use
the search tool with the extra arguments.

Provider selection is deterministic:

1. the exact optional provider ID requested in the native call, when it is available;
2. an already connected enabled mapped provider;
3. the first enabled mapped provider in server-ID order;
4. the configured SearXNG URL, when it accepts connections (or no text browser is installed);
5. the configured text browser querying DuckDuckGo;
6. public knowledge sources, when they are allowed in Settings.

SearXNG retries through the installed text browsers if it fails, and the text
browser falls back to public knowledge sources; both fallbacks are listed in the
approval preview.

In a project-backed chat, generic MCP tools remain available for deliberate
non-search server use. In a no-project restricted chat, generic MCP is hidden and
rejected; only the internally mapped search operation can reach that server.

## SearXNG and Deep Research

A configured SearXNG endpoint remains a fallback for ordinary chat. Leave its URL
blank to disable that route.

The separate **Deep Research** workflow searches with SearXNG, then the text
browser, then public knowledge sources (Wikipedia, arXiv and Crossref), each
tried when the one before is not set up or fails; a fallback is recorded as a
run warning. Explicit seed URLs are read as well. Workbench does not launch a
stdio MCP child from that unattended durable planner because its approvals and
process lifetime differ from interactive chat.


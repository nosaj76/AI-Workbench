# Web search

Chat models search the web through the `web_search` tool. Workbench uses, in order: an enabled MCP search server, a configured SearXNG URL, then a local text browser (lynx, w3m, elinks or links) querying DuckDuckGo. Public knowledge sources (Wikipedia, arXiv, Crossref, Hacker News, GitHub) are available as search modes. Every query shows an approval preview unless you turn on automatic approval for web actions.

Turn on **Settings → Connection and privacy → Enable explicit web fetch and search tools** first.

## Configure the supplied Brave Search preset

1. Open **Settings → Connection and privacy**.
2. Enable **explicit web fetch and search tools**.
3. Open **MCP servers and web search**.
4. Select **Add or repair official Brave Search preset** when an older settings
   file does not already contain it.
5. Enter a Brave Search API key.
6. Enable **Brave Search as an MCP web-search provider**. Enabling it also turns
   on the global MCP and network switches in the draft.
7. Save the complete settings draft.
8. Start a normal chat and ask `search the web about Linux`.
9. Inspect and approve the exact query. On first use, the preview also includes
   the pinned `npx` stdio launch that may obtain the package.

No project folder is required. Native tool calling still requires a compatible
installed Ollama model. With automatic routing enabled, Workbench attempts to use
a tool-capable coordinator.

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
      "max_results": 10
    }
  }
]
```

The mapped tool and argument names must be nonempty and contain no control
characters. Fixed arguments cannot replace the dynamic query or limit fields and
are size-bounded. Query/result bounds are validated locally. Set
`limit_argument` to `null` when the provider accepts no count. The server must
advertise the configured tool after initialization.

Provider selection is deterministic:

1. the exact optional provider ID requested in the native call;
2. an already connected enabled mapped provider;
3. the first enabled mapped provider in server-ID order;
4. the configured SearXNG fallback.

In a project-backed chat, generic MCP tools remain available for deliberate
non-search server use. In a no-project restricted chat, generic MCP is hidden and
rejected; only the internally mapped search operation can reach that server.

## SearXNG and Deep Research

A configured SearXNG endpoint remains a fallback for ordinary chat. Leave its URL
blank to disable that route.

The separate **Deep Research** workflow still uses SearXNG and/or explicit seed
URLs. Workbench does not silently launch a stdio MCP child from that unattended durable
planner because its approvals and process lifetime differ from interactive chat.
The MCP bridge specifically repairs approved interactive search.


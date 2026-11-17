//! Public knowledge sources as a search provider.
//!
//! A text browser pointed at one general search engine fails whenever that
//! engine rate-limits or blocks automated clients, and then there is nothing
//! else to search with. These sources publish APIs meant for programs, need no
//! account or key, and answer particular kinds of question well:
//!
//! | mode | sources | good for |
//! | --- | --- | --- |
//! | `general` | Wikipedia, Hacker News | background and current discussion |
//! | `encyclopedia` | Wikipedia | definitions, history, concepts |
//! | `academic` | arXiv, Crossref | papers and their abstracts |
//! | `tech` | Hacker News, GitHub | tools, libraries, engineering discussion |
//! | `herbal` | Herb Helper, NOMAD's health libraries (the user's own sites) | herbs, remedies, safety, interactions |
//! | `offline` | NOMAD library (Kiwix) | offline Wikipedia, survival, docs and other installed libraries |
//! | `bible` | Bible Research Lexicon (the user's own site) | Ethiopian and related biblical texts |
//! | `knowledge` | Knowledge Cache (the user's own site) | sourced articles on topics from past conversations |
//!
//! The last four are the user's own sites on this computer's home server. They
//! are reached only at local addresses, and queries sent to them never leave
//! the computer.
//!
//! Each source reports separately whether it answered, so a failure of one is
//! never hidden behind the results of another.
//!
//! (Idea: Vane's search focus modes; independently written.)
use crate::{
    integrations::{is_local_destination, limited_body, network_client, network_client_for},
    qa_library::normalize_question,
};
use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::OnceLock;

pub const MODES: [&str; 8] = [
    "general",
    "encyclopedia",
    "academic",
    "tech",
    "herbal",
    "offline",
    "bible",
    "knowledge",
];
const MAX_QUERY_CHARS: usize = 2_048;
const MAX_SNIPPET_CHARS: usize = 4_000;

/// A source of results: its id (used in requests), its name and its host.
pub struct Source {
    pub id: &'static str,
    pub label: &'static str,
    pub host: &'static str,
}

pub const SOURCES: [Source; 10] = [
    Source {
        id: "wikipedia",
        label: "Wikipedia",
        host: "en.wikipedia.org",
    },
    Source {
        id: "hackernews",
        label: "Hacker News",
        host: "hn.algolia.com",
    },
    Source {
        id: "arxiv",
        label: "arXiv",
        host: "export.arxiv.org",
    },
    Source {
        id: "crossref",
        label: "Crossref",
        host: "api.crossref.org",
    },
    Source {
        id: "github",
        label: "GitHub",
        host: "api.github.com",
    },
    Source {
        id: "herbal",
        label: "Herb Helper (your site)",
        host: "127.0.0.1:8201",
    },
    Source {
        id: "nomad",
        label: "NOMAD offline library",
        host: "127.0.0.1:8090",
    },
    Source {
        id: "nomad_health",
        label: "NOMAD health libraries",
        host: "127.0.0.1:8090",
    },
    Source {
        id: "bible",
        label: "Bible Research Lexicon (your site)",
        host: "127.0.0.1:8203",
    },
    Source {
        id: "knowledgecache",
        label: "Knowledge Cache (your site)",
        host: "127.0.0.1:8205",
    },
];

/// Sources on the user's own home server, reached only at local addresses.
pub fn is_local(id: &str) -> bool {
    matches!(
        id,
        "herbal" | "nomad" | "nomad_health" | "bible" | "knowledgecache"
    )
}

/// The source ids a mode searches, or None for an unknown mode.
pub fn mode_sources(mode: &str) -> Option<Vec<String>> {
    let ids: &[&str] = match mode {
        "general" => &["wikipedia", "hackernews"],
        "encyclopedia" => &["wikipedia"],
        "academic" => &["arxiv", "crossref"],
        "tech" => &["hackernews", "github"],
        "herbal" => &["herbal", "nomad_health"],
        "offline" => &["nomad"],
        "bible" => &["bible"],
        "knowledge" => &["knowledgecache"],
        _ => return None,
    };
    Some(ids.iter().map(|s| (*s).to_string()).collect())
}

pub fn label(id: &str) -> &str {
    SOURCES.iter().find(|s| s.id == id).map_or(id, |s| s.label)
}

pub fn host(id: &str) -> &str {
    SOURCES.iter().find(|s| s.id == id).map_or("", |s| s.host)
}

/// Where each source is reached. Tests point these at a local server.
#[derive(Clone, Debug)]
pub struct Endpoints {
    pub wikipedia: String,
    pub hackernews: String,
    pub arxiv: String,
    pub crossref: String,
    pub github: String,
    /// Herb Helper's search API (POST JSON).
    pub herbal: String,
    /// NOMAD's Kiwix server search, which covers every installed library.
    pub nomad: String,
    /// The Bible Research Lexicon's JSON search.
    pub bible: String,
    /// The Knowledge Cache's JSON search (articles and dictionary definitions).
    pub knowledgecache: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self {
            wikipedia: "https://en.wikipedia.org/w/api.php".into(),
            hackernews: "https://hn.algolia.com/api/v1/search".into(),
            arxiv: "https://export.arxiv.org/api/query".into(),
            crossref: "https://api.crossref.org/works".into(),
            github: "https://api.github.com/search/repositories".into(),
            herbal: "http://127.0.0.1:8201/api/search".into(),
            nomad: "http://127.0.0.1:8090/search".into(),
            bible: "http://127.0.0.1:8203/api/search".into(),
            knowledgecache: "http://127.0.0.1:8205/api/search".into(),
        }
    }
}

impl Endpoints {
    /// The default endpoints with the user's own sites (Herb Helper, NOMAD, the
    /// Bible lexicon and the Knowledge Cache) moved to the scheme and host of the
    /// configured local sites address. Each site keeps its own port and path.
    pub fn for_local_sites(local_sites_url: &str) -> Result<Self> {
        let base = url::Url::parse(local_sites_url.trim()).context("Invalid local sites URL")?;
        let host = base
            .host_str()
            .context("The local sites address needs a host name")?;
        let mut endpoints = Self::default();
        for endpoint in [
            &mut endpoints.herbal,
            &mut endpoints.nomad,
            &mut endpoints.bible,
            &mut endpoints.knowledgecache,
        ] {
            let mut moved = url::Url::parse(endpoint).context("Default local site endpoint")?;
            moved
                .set_scheme(base.scheme())
                .map_err(|_| anyhow::anyhow!("Unsupported local sites scheme"))?;
            moved
                .set_host(Some(host))
                .context("Apply the local sites host")?;
            *endpoint = moved.into();
        }
        Ok(endpoints)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub source: String,
}

/// How one source fared.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceReport {
    pub source: String,
    pub ok: bool,
    pub returned: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct KnowledgeSearch {
    pub results: Vec<KnowledgeResult>,
    pub reports: Vec<SourceReport>,
}

// ---------------------------------------------------------------- parsing --

fn tags() -> &'static Regex {
    static TAGS: OnceLock<Regex> = OnceLock::new();
    TAGS.get_or_init(|| Regex::new(r"<[^>]*>").expect("valid pattern"))
}

fn decode_entities(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Plain text from markup: tags removed, entities decoded, whitespace collapsed.
fn clean(text: &str) -> String {
    decode_entities(&tags().replace_all(text, " "))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn snippet(text: &str) -> String {
    clean(text).chars().take(MAX_SNIPPET_CHARS).collect()
}

fn text_of<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn json(body: &str, who: &str) -> Result<Value> {
    serde_json::from_str(body).with_context(|| format!("{who} returned data that is not JSON"))
}

pub fn parse_wikipedia(body: &str, limit: usize) -> Result<Vec<KnowledgeResult>> {
    let data = json(body, "Wikipedia")?;
    let mut pages: Vec<&Value> = data["query"]["pages"]
        .as_array()
        .map(|p| p.iter().collect())
        .unwrap_or_default();
    pages.sort_by_key(|p| p["index"].as_u64().unwrap_or(u64::MAX));
    Ok(pages
        .into_iter()
        .filter_map(|page| {
            let title = text_of(page, "title");
            if title.is_empty() {
                return None;
            }
            let url = match text_of(page, "fullurl") {
                "" => format!("https://en.wikipedia.org/wiki/{}", title.replace(' ', "_")),
                full => full.to_string(),
            };
            Some(KnowledgeResult {
                title: title.to_string(),
                url,
                snippet: snippet(text_of(page, "extract")),
                source: "wikipedia".into(),
            })
        })
        .take(limit)
        .collect())
}

pub fn parse_hackernews(body: &str, limit: usize) -> Result<Vec<KnowledgeResult>> {
    let data = json(body, "Hacker News")?;
    Ok(data["hits"]
        .as_array()
        .map(|hits| hits.as_slice())
        .unwrap_or_default()
        .iter()
        .filter_map(|hit| {
            let title = text_of(hit, "title");
            let id = hit["objectID"].as_str().unwrap_or("");
            if title.is_empty() || id.is_empty() {
                return None;
            }
            let discussion = format!("https://news.ycombinator.com/item?id={id}");
            let url = match text_of(hit, "url") {
                "" => discussion.clone(),
                link => link.to_string(),
            };
            Some(KnowledgeResult {
                title: clean(title),
                url,
                snippet: format!(
                    "{} points and {} comments on Hacker News. Discussion: {discussion}",
                    hit["points"].as_u64().unwrap_or(0),
                    hit["num_comments"].as_u64().unwrap_or(0)
                ),
                source: "hackernews".into(),
            })
        })
        .take(limit)
        .collect())
}

pub fn parse_arxiv(body: &str, limit: usize) -> Result<Vec<KnowledgeResult>> {
    static ENTRY: OnceLock<Regex> = OnceLock::new();
    let entry =
        ENTRY.get_or_init(|| Regex::new(r"(?s)<entry>(.*?)</entry>").expect("valid pattern"));
    let field = |name: &str, text: &str| -> String {
        let re = Regex::new(&format!(r"(?s)<{name}[^>]*>(.*?)</{name}>")).expect("valid pattern");
        re.captures(text).map(|c| clean(&c[1])).unwrap_or_default()
    };
    if !body.contains("<feed") {
        bail!("arXiv returned data that is not an Atom feed");
    }
    Ok(entry
        .captures_iter(body)
        .filter_map(|c| {
            let block = &c[1];
            let (title, url) = (field("title", block), field("id", block));
            if title.is_empty() || url.is_empty() {
                return None;
            }
            Some(KnowledgeResult {
                title,
                url,
                snippet: snippet(&field("summary", block)),
                source: "arxiv".into(),
            })
        })
        .take(limit)
        .collect())
}

pub fn parse_crossref(body: &str, limit: usize) -> Result<Vec<KnowledgeResult>> {
    let data = json(body, "Crossref")?;
    Ok(data["message"]["items"]
        .as_array()
        .map(|items| items.as_slice())
        .unwrap_or_default()
        .iter()
        .filter_map(|item| {
            let title = item["title"][0].as_str().unwrap_or("");
            let url = text_of(item, "URL");
            if title.is_empty() || url.is_empty() {
                return None;
            }
            let venue = item["container-title"][0].as_str().unwrap_or("");
            let year = item["issued"]["date-parts"][0][0].as_u64();
            let mut text = snippet(text_of(item, "abstract"));
            if text.is_empty() {
                text = format!(
                    "Published{}{}.",
                    year.map(|y| format!(" in {y}")).unwrap_or_default(),
                    if venue.is_empty() {
                        String::new()
                    } else {
                        format!(" in {venue}")
                    }
                );
            }
            Some(KnowledgeResult {
                title: clean(title),
                url: url.to_string(),
                snippet: text,
                source: "crossref".into(),
            })
        })
        .take(limit)
        .collect())
}

pub fn parse_github(body: &str, limit: usize) -> Result<Vec<KnowledgeResult>> {
    let data = json(body, "GitHub")?;
    Ok(data["items"]
        .as_array()
        .map(|items| items.as_slice())
        .unwrap_or_default()
        .iter()
        .filter_map(|repo| {
            let name = text_of(repo, "full_name");
            let url = text_of(repo, "html_url");
            if name.is_empty() || url.is_empty() {
                return None;
            }
            let description = text_of(repo, "description");
            Some(KnowledgeResult {
                title: name.to_string(),
                url: url.to_string(),
                snippet: format!(
                    "{}{} stars{}.",
                    if description.trim().is_empty() {
                        String::new()
                    } else {
                        format!("{}. ", description.trim().trim_end_matches('.'))
                    },
                    repo["stargazers_count"].as_u64().unwrap_or(0),
                    match text_of(repo, "language") {
                        "" => String::new(),
                        language => format!(", written in {language}"),
                    }
                ),
                source: "github".into(),
            })
        })
        .take(limit)
        .collect())
}

/// "http://host:port" of an endpoint, for turning a site's relative links into full ones.
fn origin(endpoint: &str) -> String {
    url::Url::parse(endpoint)
        .map(|u| u.origin().ascii_serialization())
        .unwrap_or_default()
}

fn joined(value: &Value, key: &str) -> String {
    value[key]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s.trim().to_string()),
                    Value::Object(_) => Some(
                        [
                            text_of(item, "interacting_item"),
                            text_of(item, "action_rating"),
                            text_of(item, "mechanism"),
                        ]
                        .iter()
                        .filter(|s| !s.is_empty())
                        .copied()
                        .collect::<Vec<_>>()
                        .join(" - "),
                    ),
                    _ => None,
                })
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("; ")
        })
        .unwrap_or_default()
}

/// Herb Helper's safety-screened search. Its safety message comes first when it
/// flags an emergency or a risk, so the answer cannot omit it.
pub fn parse_herbal(body: &str, limit: usize, site: &str) -> Result<Vec<KnowledgeResult>> {
    let data = json(body, "Herb Helper")?;
    if data.get("results").is_none() {
        bail!("Herb Helper returned an unexpected answer");
    }
    let mut out = Vec::new();
    let flagged = ["emergency_hits", "risk_hits"]
        .iter()
        .any(|k| data[*k].as_array().is_some_and(|a| !a.is_empty()));
    if flagged {
        out.push(KnowledgeResult {
            title: format!(
                "Safety notice from Herb Helper ({})",
                text_of(&data, "safety_status")
            ),
            url: format!("{site}/symptom-checker"),
            snippet: format!(
                "{} Flagged: {}.",
                text_of(&data, "safety_message"),
                [joined(&data, "emergency_hits"), joined(&data, "risk_hits")]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            source: "herbal".into(),
        });
    }
    for herb in data["results"].as_array().into_iter().flatten() {
        let name = text_of(herb, "common_name");
        let Some(id) = herb["id"].as_i64() else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let mut parts = vec![text_of(herb, "traditional_use").to_string()];
        for (label, value) in [
            ("Evidence", text_of(herb, "evidence_display").to_string()),
            ("Safety", text_of(herb, "safety_display").to_string()),
            ("Avoid if", joined(herb, "avoid_if")),
            ("Warnings", joined(herb, "warnings")),
            ("Interactions", joined(herb, "interactions")),
            ("Preparation", text_of(herb, "preparation_note").to_string()),
        ] {
            if !value.trim().is_empty() {
                parts.push(format!("{label}: {}", value.trim()));
            }
        }
        let latin = text_of(herb, "latin_name");
        out.push(KnowledgeResult {
            title: if latin.is_empty() {
                name.to_string()
            } else {
                format!("{name} ({latin})")
            },
            url: format!("{site}/herbs/{id}"),
            snippet: snippet(&parts.join(" | ")),
            source: "herbal".into(),
        });
    }
    for formula in data["formula_results"].as_array().into_iter().flatten() {
        let name = text_of(formula, "name");
        let Some(id) = formula["id"].as_i64() else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let mut parts = vec![format!("Formula ({}).", text_of(formula, "tradition"))];
        for key in [
            "intended_pattern",
            "preparation",
            "cautions",
            "evidence_note",
        ] {
            let value = match &formula[key] {
                Value::Array(_) => joined(formula, key),
                _ => text_of(formula, key).to_string(),
            };
            if !value.trim().is_empty() {
                parts.push(value.trim().to_string());
            }
        }
        out.push(KnowledgeResult {
            title: name.to_string(),
            url: format!("{site}/formulas/{id}"),
            snippet: snippet(&parts.join(" | ")),
            source: "herbal".into(),
        });
    }
    // The safety notice always stays, on top of the requested number of results.
    let keep = limit + usize::from(flagged);
    out.truncate(keep);
    Ok(out)
}

/// Kiwix's OpenSearch RSS, which spans every library NOMAD has installed.
pub fn parse_kiwix(body: &str, limit: usize, site: &str) -> Result<Vec<KnowledgeResult>> {
    static ITEM: OnceLock<Regex> = OnceLock::new();
    let item = ITEM.get_or_init(|| Regex::new(r"(?s)<item>(.*?)</item>").expect("valid pattern"));
    let field = |name: &str, text: &str| -> String {
        let re = Regex::new(&format!(r"(?s)<{name}>(.*?)</{name}>")).expect("valid pattern");
        re.captures(text).map(|c| clean(&c[1])).unwrap_or_default()
    };
    if !body.contains("<rss") {
        bail!("The NOMAD library returned data that is not a search feed");
    }
    Ok(item
        .captures_iter(body)
        .filter_map(|c| {
            let block = &c[1];
            let (title, link) = (field("title", block), field("link", block));
            if title.is_empty() || !link.starts_with('/') {
                return None;
            }
            let book = field("book", block);
            let text = field("description", block);
            Some(KnowledgeResult {
                title: if book.is_empty() {
                    title
                } else {
                    format!("{title} ({book})")
                },
                url: format!("{site}{link}"),
                snippet: snippet(text.trim_matches('.').trim()),
                source: "nomad".into(),
            })
        })
        .take(limit)
        .collect())
}

/// Library ids from Kiwix's OPDS catalog that one search may combine. Kiwix
/// refuses a search spanning books in different languages, and NOMAD installs
/// some multilingual books, so only books in the catalog's most common single
/// language are kept. With `health_only`, only health-related libraries
/// (WikiMed, MedlinePlus, first aid, herbs and the like) are kept, so a herbal
/// search is not diluted by programming or hobby libraries.
pub fn library_ids(catalog: &str, health_only: bool) -> Vec<String> {
    static ENTRY: OnceLock<Regex> = OnceLock::new();
    static ID: OnceLock<Regex> = OnceLock::new();
    static LANG: OnceLock<Regex> = OnceLock::new();
    static HEALTH: OnceLock<Regex> = OnceLock::new();
    let entry =
        ENTRY.get_or_init(|| Regex::new(r"(?s)<entry>(.*?)</entry>").expect("valid pattern"));
    let id = ID.get_or_init(|| {
        Regex::new(r"<id>urn:uuid:([0-9a-fA-F-]{8,64})</id>").expect("valid pattern")
    });
    let lang = LANG
        .get_or_init(|| Regex::new(r"<language>\s*([^<]*?)\s*</language>").expect("valid pattern"));
    let health = HEALTH.get_or_init(|| {
        Regex::new(r"(?i)(wikimed|medic|medline|health|herb|first[-_ ]?aid|nurs|anatom|pharm|drug|nutrition|botan|plant|disease|wellness)")
            .expect("valid pattern")
    });
    // (id, language, description) of every book with an id.
    let books: Vec<(String, String, String)> = entry
        .captures_iter(catalog)
        .filter_map(|c| {
            let block = &c[1];
            let book_id = id.captures(block)?[1].to_string();
            let language = lang
                .captures(block)
                .map(|m| m[1].to_string())
                .unwrap_or_default();
            let described: String = ["name", "title", "summary", "category", "tags"]
                .iter()
                .filter_map(|k| {
                    Regex::new(&format!(r"(?s)<{k}>(.*?)</{k}>"))
                        .ok()?
                        .captures(block)
                        .map(|m| m[1].to_string())
                })
                .collect::<Vec<_>>()
                .join(" ");
            Some((book_id, language, described))
        })
        .collect();
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for (_, language, _) in &books {
        if !language.is_empty() && !language.contains(',') {
            *counts.entry(language.as_str()).or_default() += 1;
        }
    }
    let Some(main) = counts
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|(l, _)| l.to_string())
    else {
        return Vec::new();
    };
    books
        .into_iter()
        .filter(|(_, language, described)| {
            *language == main && (!health_only || health.is_match(described))
        })
        .map(|(book_id, _, _)| book_id)
        .collect()
}

/// A Kiwix search over the given books.
fn kiwix_search(site: &str, ids: &[String], query: &str, n: String) -> Result<url::Url> {
    let mut pairs: Vec<(&str, String)> = ids.iter().map(|id| ("books.id", id.clone())).collect();
    pairs.extend([
        ("pattern", query.to_string()),
        ("format", "xml".to_string()),
        ("pageLength", n),
    ]);
    endpoint(&format!("{site}/search"), &pairs)
}

/// The Bible Research Lexicon's /api/search, with each reading's edition and review status.
pub fn parse_bible(body: &str, limit: usize, site: &str) -> Result<Vec<KnowledgeResult>> {
    let data = json(body, "The Bible Research Lexicon")?;
    Ok(data["results"]
        .as_array()
        .map(|r| r.as_slice())
        .unwrap_or_default()
        .iter()
        .filter_map(|hit| {
            let reference = text_of(hit, "reference");
            if reference.is_empty() {
                return None;
            }
            let path = text_of(hit, "url");
            let status = text_of(hit, "status_label");
            Some(KnowledgeResult {
                title: match text_of(hit, "edition_title") {
                    "" => reference.to_string(),
                    edition => format!("{reference} - {edition}"),
                },
                url: if path.starts_with('/') {
                    format!("{site}{path}")
                } else {
                    site.to_string()
                },
                snippet: snippet(&if status.is_empty() {
                    text_of(hit, "text").to_string()
                } else {
                    format!("{} [{status}]", text_of(hit, "text"))
                }),
                source: "bible".into(),
            })
        })
        .take(limit)
        .collect())
}

/// The Knowledge Cache's /api/search: dictionary definitions first, then
/// sourced articles, each article listing the sources it was written from.
pub fn parse_knowledgecache(body: &str, limit: usize, site: &str) -> Result<Vec<KnowledgeResult>> {
    let data = json(body, "The Knowledge Cache")?;
    if data.get("articles").is_none() {
        bail!("The Knowledge Cache returned an unexpected answer");
    }
    let mut out = Vec::new();
    for word in data["definitions"].as_array().into_iter().flatten() {
        let name = text_of(word, "word");
        let definition = text_of(word, "definition");
        if name.is_empty() || definition.is_empty() {
            continue;
        }
        out.push(KnowledgeResult {
            title: format!("{name} (dictionary)"),
            url: format!("{site}/word/{name}"),
            snippet: snippet(&format!(
                "{definition} [definition from {}]",
                text_of(word, "def_source")
            )),
            source: "knowledgecache".into(),
        });
    }
    for article in data["articles"].as_array().into_iter().flatten() {
        let (title, path) = (text_of(article, "title"), text_of(article, "url"));
        if title.is_empty() || !path.starts_with('/') {
            continue;
        }
        let cited: Vec<String> = article["sources"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|s| format!("{} ({})", text_of(s, "title"), text_of(s, "origin")))
            .collect();
        let mut text = format!(
            "{} {}",
            text_of(article, "summary"),
            text_of(article, "snippet")
        );
        if !cited.is_empty() {
            text.push_str(&format!(" Written from: {}.", cited.join("; ")));
        }
        out.push(KnowledgeResult {
            title: format!("{title} ({})", text_of(article, "domain")),
            url: format!("{site}{path}"),
            snippet: snippet(&text),
            source: "knowledgecache".into(),
        });
    }
    out.truncate(limit);
    Ok(out)
}

// -------------------------------------------------------------- fetching --

fn get(url: url::Url, accept: &str, who: &str) -> Result<String> {
    let response = network_client()?
        .get(url)
        .header(reqwest::header::ACCEPT, accept)
        .send()
        .with_context(|| format!("{who} could not be reached"))?;
    let status = response.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status == reqwest::StatusCode::FORBIDDEN
    {
        bail!("{who} refused the request ({status}); it limits how often it can be asked. Wait a while and try again");
    }
    if !status.is_success() {
        bail!("{who} answered with an error ({status})");
    }
    Ok(String::from_utf8_lossy(&limited_body(response)?).into_owned())
}

/// Asks one of the user's own sites. The address must be on this computer or
/// the local network; anything else is refused before a request is made.
fn local_request(url: url::Url, json_body: Option<&Value>, who: &str) -> Result<String> {
    if !is_local_destination(&url) {
        bail!("{who} must be on this computer or the local network");
    }
    let client = network_client_for(&url)?;
    let request = match json_body {
        Some(body) => client.post(url).json(body),
        None => client.get(url),
    };
    // The sites share a USB drive with NOMAD's downloads, which can stall them briefly.
    // The request address can list a hundred library ids; keep it out of the message.
    let response = request
        .timeout(std::time::Duration::from_secs(90))
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                anyhow::anyhow!("{who} took too long to answer; it may be busy")
            } else {
                anyhow::anyhow!(
                    "{who} could not be reached; is its service running? ({})",
                    e.without_url()
                )
            }
        })?;
    let status = response.status();
    if !status.is_success() {
        bail!("{who} answered with an error ({status})");
    }
    Ok(String::from_utf8_lossy(&limited_body(response)?).into_owned())
}

fn endpoint(base: &str, pairs: &[(&str, String)]) -> Result<url::Url> {
    let mut url = url::Url::parse(base).with_context(|| format!("Bad endpoint {base}"))?;
    {
        let mut query = url.query_pairs_mut();
        query.clear();
        for (key, value) in pairs {
            query.append_pair(key, value);
        }
    }
    Ok(url)
}

/// arXiv wants field-qualified terms; every significant word must appear.
fn arxiv_query(query: &str) -> Result<String> {
    let words: Vec<String> = normalize_question(query)
        .split(' ')
        .filter(|w| w.len() > 1)
        .take(8)
        .map(|w| format!("all:{w}"))
        .collect();
    if words.is_empty() {
        bail!("arXiv needs at least one searchable word");
    }
    Ok(words.join(" AND "))
}

fn fetch_one(
    endpoints: &Endpoints,
    id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<KnowledgeResult>> {
    let n = limit.to_string();
    match id {
        "wikipedia" => {
            let url = endpoint(
                &endpoints.wikipedia,
                &[
                    ("action", "query".into()),
                    ("generator", "search".into()),
                    ("gsrsearch", query.into()),
                    ("gsrlimit", n),
                    ("prop", "extracts|info".into()),
                    ("exintro", "1".into()),
                    ("explaintext", "1".into()),
                    ("exsentences", "6".into()),
                    ("inprop", "url".into()),
                    ("redirects", "1".into()),
                    ("format", "json".into()),
                    ("formatversion", "2".into()),
                ],
            )?;
            parse_wikipedia(&get(url, "application/json", "Wikipedia")?, limit)
        }
        "hackernews" => {
            let url = endpoint(
                &endpoints.hackernews,
                &[
                    ("query", query.into()),
                    ("tags", "story".into()),
                    ("hitsPerPage", n),
                ],
            )?;
            parse_hackernews(&get(url, "application/json", "Hacker News")?, limit)
        }
        "arxiv" => {
            let url = endpoint(
                &endpoints.arxiv,
                &[
                    ("search_query", arxiv_query(query)?),
                    ("start", "0".into()),
                    ("max_results", n),
                ],
            )?;
            parse_arxiv(&get(url, "application/atom+xml", "arXiv")?, limit)
        }
        "crossref" => {
            let url = endpoint(
                &endpoints.crossref,
                &[
                    ("query", query.into()),
                    ("rows", n),
                    ("select", "title,URL,abstract,container-title,issued".into()),
                ],
            )?;
            parse_crossref(&get(url, "application/json", "Crossref")?, limit)
        }
        "github" => {
            let url = endpoint(
                &endpoints.github,
                &[
                    ("q", query.into()),
                    ("per_page", n),
                    ("sort", "stars".into()),
                ],
            )?;
            parse_github(&get(url, "application/vnd.github+json", "GitHub")?, limit)
        }
        "herbal" => {
            let url = url::Url::parse(&endpoints.herbal).context("Bad Herb Helper address")?;
            let body = local_request(
                url,
                Some(&serde_json::json!({"problem": query})),
                "Herb Helper",
            )?;
            parse_herbal(&body, limit, &origin(&endpoints.herbal))
        }
        "nomad" | "nomad_health" => {
            let site = origin(&endpoints.nomad);
            let catalog_url = endpoint(
                &format!("{site}/catalog/v2/entries"),
                &[("count", "-1".into())],
            )?;
            let health_only = id == "nomad_health";
            let ids = library_ids(
                &local_request(catalog_url, None, "The NOMAD library")?,
                health_only,
            );
            if ids.is_empty() {
                bail!(if health_only {
                    "No health library is installed in NOMAD yet; add one (for example WikiMed or MedlinePlus) in NOMAD's content manager"
                } else {
                    "No library is installed in NOMAD yet; add some in NOMAD's content manager"
                });
            }
            let url = kiwix_search(&site, &ids, query, n)?;
            parse_kiwix(
                &local_request(url, None, "The NOMAD library")?,
                limit,
                &site,
            )
            .map(|found| {
                found
                    .into_iter()
                    .map(|r| KnowledgeResult {
                        source: id.into(),
                        ..r
                    })
                    .collect()
            })
        }
        "knowledgecache" => {
            let url = endpoint(
                &endpoints.knowledgecache,
                &[("q", query.into()), ("limit", n)],
            )?;
            let body = local_request(url, None, "The Knowledge Cache")?;
            parse_knowledgecache(&body, limit, &origin(&endpoints.knowledgecache))
        }
        "bible" => {
            let url = endpoint(&endpoints.bible, &[("q", query.into()), ("limit", n)])?;
            let body = local_request(url, None, "The Bible Research Lexicon")?;
            parse_bible(&body, limit, &origin(&endpoints.bible))
        }
        other => bail!("Unknown knowledge source {other}"),
    }
}

/// Whether a message is about herbs, natural remedies or health, so the user's own Herb Helper
/// and NOMAD's medical libraries should be consulted before the model answers.
pub fn is_health_question(text: &str) -> bool {
    static HEALTH: OnceLock<Regex> = OnceLock::new();
    HEALTH
        .get_or_init(|| {
            Regex::new(concat!(
                r"(?i)\b(herbs?|herbal\w*|botanicals?|remed(y|ies)|supplements?|tinctures?|tisanes?|",
                r"medicinal|medicines?|medications?|drugs?|dos(e|age|ing)|side[- ]effects?|interactions?|",
                r"symptoms?|diagnos\w*|treat(s|ed|ing|ment|ments)?|cures?|illness\w*|disease\w*|",
                r"infections?|pain|ache\w*|fever|cough|insomnia|anxiety|inflammation|blood pressure|",
                r"diabetes|allerg\w*|pregnan\w*|first aid|wounds?|burns?|essential oils?|",
                r"ginseng|echinacea|valerian|chamomile|turmeric|ashwagandha|elderberry|st\.? john'?s wort|",
                r"ginkgo|kava|lavender|peppermint|calendula|garlic|ginger)\b"
            ))
            .expect("valid pattern")
        })
        .is_match(text)
}

/// The first three key words of a long query (common words left out), or None when the
/// query is already that short.
pub fn key_words(query: &str) -> Option<String> {
    const COMMON: &[&str] = &[
        "a", "an", "and", "are", "best", "can", "do", "does", "for", "from", "how", "i", "in",
        "is", "it", "my", "of", "on", "or", "the", "to", "use", "using", "vs", "what", "when",
        "which", "why", "with",
    ];
    let words: Vec<&str> = query.split_whitespace().collect();
    let key: Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| !COMMON.contains(&w.to_lowercase().as_str()))
        .take(3)
        .collect();
    (!key.is_empty() && key.len() < words.len()).then(|| key.join(" "))
}

/// Searches each source in turn and interleaves what they return, so one
/// source cannot crowd out the others. Fails only when every source failed.
pub fn search(
    endpoints: &Endpoints,
    query: &str,
    sources: &[String],
    limit: usize,
) -> Result<KnowledgeSearch> {
    let query = query.trim();
    if query.is_empty() || query.chars().count() > MAX_QUERY_CHARS {
        bail!("Search query is empty or too long");
    }
    if sources.is_empty() {
        bail!("No knowledge source was selected");
    }
    let limit = limit.clamp(1, 20);
    let mut per_source: Vec<Vec<KnowledgeResult>> = Vec::new();
    let mut reports = Vec::new();
    for id in sources {
        let mut answer = fetch_one(endpoints, id, query, limit);
        // Hacker News and GitHub require every word to match, so a long natural-language
        // query often finds nothing; retry once with its few key words.
        if matches!(id.as_str(), "hackernews" | "github")
            && matches!(&answer, Ok(found) if found.is_empty())
        {
            if let Some(short) = key_words(query) {
                answer = fetch_one(endpoints, id, &short, limit);
            }
        }
        match answer {
            Ok(found) => {
                reports.push(SourceReport {
                    source: id.clone(),
                    ok: true,
                    returned: found.len(),
                    error: None,
                });
                per_source.push(found);
            }
            Err(error) => reports.push(SourceReport {
                source: id.clone(),
                ok: false,
                returned: 0,
                error: Some(format!("{error:#}")),
            }),
        }
    }
    if reports.iter().all(|r| !r.ok) {
        bail!(
            "No knowledge source answered: {}",
            reports
                .iter()
                .map(|r| format!(
                    "{}: {}",
                    label(&r.source),
                    r.error.as_deref().unwrap_or("failed")
                ))
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    let mut results = Vec::new();
    let mut round = 0;
    while results.len() < limit && per_source.iter().any(|s| round < s.len()) {
        for source in &per_source {
            if let Some(item) = source.get(round) {
                if results.len() < limit
                    && !results.iter().any(|r: &KnowledgeResult| r.url == item.url)
                {
                    results.push(item.clone());
                }
            }
        }
        round += 1;
    }
    Ok(KnowledgeSearch { results, reports })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{Arc, Mutex},
    };

    #[test]
    fn local_sites_address_moves_only_the_users_own_sites() {
        let moved = Endpoints::for_local_sites("http://homeserver.lan/").unwrap();
        assert_eq!(moved.herbal, "http://homeserver.lan:8201/api/search");
        assert_eq!(moved.nomad, "http://homeserver.lan:8090/search");
        assert_eq!(moved.bible, "http://homeserver.lan:8203/api/search");
        assert_eq!(
            moved.knowledgecache,
            "http://homeserver.lan:8205/api/search"
        );
        assert_eq!(moved.wikipedia, Endpoints::default().wikipedia);
        let local = Endpoints::for_local_sites("http://127.0.0.1/").unwrap();
        assert_eq!(local.herbal, Endpoints::default().herbal);
        assert!(Endpoints::for_local_sites("not a url").is_err());
    }

    const WIKI: &str = r#"{"query":{"pages":[
        {"pageid":2,"title":"Rust (programming language)","index":2,"extract":"Rust is a general-purpose programming language emphasizing performance, type safety, and concurrency.","fullurl":"https://en.wikipedia.org/wiki/Rust_(programming_language)"},
        {"pageid":1,"title":"Rust","index":1,"extract":"Rust is an iron oxide formed by the reaction of iron with oxygen."}
    ]}}"#;
    const HN: &str = r#"{"hits":[
        {"title":"Why Rust is loved","url":"https://blog.example/rust","objectID":"11","points":420,"num_comments":88},
        {"title":"Ask HN: Learning Rust?","url":null,"objectID":"12","points":9,"num_comments":3},
        {"title":"","objectID":"13"}
    ]}"#;
    const ARXIV: &str = r#"<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
        <entry><id>http://arxiv.org/abs/2101.00001v1</id><title>Safe Systems
        Programming &amp; Ownership</title><summary>We study how ownership types prevent data races in systems code.</summary></entry>
        <entry><id>http://arxiv.org/abs/2101.00002v1</id><title>Second Paper</title><summary>Another abstract.</summary></entry></feed>"#;
    const CROSSREF: &str = r#"{"message":{"items":[
        {"title":["Ownership Types"],"URL":"https://doi.org/10.1/own","abstract":"<jats:p>Ownership <jats:italic>types</jats:italic> restrict aliasing.</jats:p>","container-title":["Journal of Types"],"issued":{"date-parts":[[2019,5]]}},
        {"title":["No Abstract Paper"],"URL":"https://doi.org/10.1/none","container-title":["Proc. Safety"],"issued":{"date-parts":[[2020]]}}
    ]}}"#;
    const GITHUB: &str = r#"{"items":[
        {"full_name":"rust-lang/rust","html_url":"https://github.com/rust-lang/rust","description":"Empowering everyone to build reliable software.","stargazers_count":99000,"language":"Rust"},
        {"full_name":"someone/nodesc","html_url":"https://github.com/someone/nodesc","description":null,"stargazers_count":5,"language":null}
    ]}"#;

    const HERBAL: &str = r#"{"query":"insomnia","safety_status":"educational","safety_message":"Educational results only.","risk_hits":[],"emergency_hits":[],
        "results":[{"id":42,"common_name":"Valerian","latin_name":"Valeriana officinalis","traditional_use":"Used for sleep.","evidence_display":"Limited human evidence","safety_display":"Use with care","avoid_if":["pregnancy"],"warnings":["May add to sedation."],"interactions":[{"interacting_item":"sedatives","action_rating":"caution","mechanism":"Additive sedation."}],"preparation_note":"Tea or tincture."}],
        "formula_results":[{"id":7,"name":"Sleep blend","tradition":"EMA/HMPC","intended_pattern":"Temporary insomnia.","cautions":["Not for children."]}]}"#;
    const HERBAL_EMERGENCY: &str = r#"{"query":"chest pain","safety_status":"emergency","safety_message":"Seek emergency medical help now.","risk_hits":[],"emergency_hits":["chest pain"],"results":[],"formula_results":[]}"#;
    const KIWIX: &str = r#"<?xml version="1.0"?><rss version="2.0"><channel><title>Search</title>
        <item><title>Boiling water</title><link>/content/water_en/boil</link><description>Boil &lt;b&gt;water&lt;/b&gt; for one minute.</description><book><title>Water Treatment Library</title></book></item>
        <item><title>No link</title><link>http://elsewhere/x</link><description>x</description></item></channel></rss>"#;
    const BIBLE: &str = r#"{"query":"Enoch","total":1,"results":[{"reference":"1 Enoch 12:3","edition_title":"R. H. Charles (1917)","text":"And I, Enoch, was blessing the Lord.","status_label":"Editorial review pending","url":"/bible/1En/12#verse-3"}]}"#;

    #[test]
    fn herbal_results_carry_evidence_safety_and_interactions() {
        let found = parse_herbal(HERBAL, 5, "http://127.0.0.1:8201").unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].title, "Valerian (Valeriana officinalis)");
        assert_eq!(found[0].url, "http://127.0.0.1:8201/herbs/42");
        for part in [
            "Evidence: Limited human evidence",
            "Safety: Use with care",
            "Avoid if: pregnancy",
            "sedatives - caution - Additive sedation.",
        ] {
            assert!(
                found[0].snippet.contains(part),
                "{part}: {}",
                found[0].snippet
            );
        }
        assert_eq!(found[1].url, "http://127.0.0.1:8201/formulas/7");
        assert!(found[1].snippet.contains("Not for children."));
    }

    #[test]
    fn herbal_emergency_notice_comes_first_and_is_never_cut() {
        let found = parse_herbal(HERBAL_EMERGENCY, 1, "http://h").unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0].title.starts_with("Safety notice"));
        assert!(found[0]
            .snippet
            .contains("Seek emergency medical help now."));
        assert!(found[0].snippet.contains("chest pain"));
        assert!(parse_herbal(r#"{"detail":"x"}"#, 5, "http://h").is_err());
    }

    #[test]
    fn kiwix_and_bible_results_link_to_the_local_pages() {
        let kiwix = parse_kiwix(KIWIX, 5, "http://127.0.0.1:8090").unwrap();
        assert_eq!(kiwix.len(), 1, "absolute links are not trusted");
        assert_eq!(kiwix[0].title, "Boiling water (Water Treatment Library)");
        assert_eq!(kiwix[0].url, "http://127.0.0.1:8090/content/water_en/boil");
        assert_eq!(kiwix[0].snippet, "Boil water for one minute");
        assert!(parse_kiwix("{}", 5, "http://k").is_err());
        let bible = parse_bible(BIBLE, 5, "http://127.0.0.1:8203").unwrap();
        assert_eq!(bible[0].title, "1 Enoch 12:3 - R. H. Charles (1917)");
        assert_eq!(bible[0].url, "http://127.0.0.1:8203/bible/1En/12#verse-3");
        assert!(bible[0].snippet.ends_with("[Editorial review pending]"));
    }

    #[test]
    fn nomad_catalog_picks_one_language_and_health_libraries() {
        let catalog = r#"<feed>
            <entry><id>urn:uuid:11111111-aaaa-bbbb-cccc-000000000001</id><name>wikipedia_en_medicine</name><title>WikiMed</title><language>eng</language><tags>_category:wikipedia</tags></entry>
            <entry><id>urn:uuid:11111111-aaaa-bbbb-cccc-000000000002</id><name>devdocs_en_python</name><title>Python Docs</title><language>eng</language><tags>devdocs;python</tags></entry>
            <entry><id>urn:uuid:11111111-aaaa-bbbb-cccc-000000000003</id><name>zimgit-first-aid_en</name><title>First Aid</title><language>eng</language><tags>preppers</tags></entry>
            <entry><id>urn:uuid:11111111-aaaa-bbbb-cccc-000000000004</id><name>arduino.stackexchange.com_en_all</name><title>Arduino Q&amp;A</title><language>eng</language></entry>
            <entry><id>urn:uuid:11111111-aaaa-bbbb-cccc-000000000005</id><name>health_multi</name><title>Health in many languages</title><language>eng,spa,fra</language></entry>
            <entry><id>urn:uuid:11111111-aaaa-bbbb-cccc-000000000006</id><name>wikipedia_fr_medicine</name><title>WikiMed FR</title><language>fra</language></entry>
        </feed>"#;
        assert_eq!(
            library_ids(catalog, true),
            [
                "11111111-aaaa-bbbb-cccc-000000000001",
                "11111111-aaaa-bbbb-cccc-000000000003"
            ]
        );
        assert_eq!(
            library_ids(catalog, false).len(),
            4,
            "multilingual and minority-language books are left out"
        );
        assert!(library_ids("<feed></feed>", false).is_empty());
    }

    #[test]
    fn knowledge_cache_gives_definitions_then_sourced_articles() {
        let body = r#"{"query":"ginseng","definitions":[{"word":"ginseng","pos":"NOUN","definition":"(n.) Chinese herb","def_source":"wordnet"}],
            "articles":[{"slug":"ginseng","title":"Ginseng","domain":"Herbal Medicine","summary":"A root used in herbal medicine.","snippet":"grown in shade","url":"/article/ginseng",
            "sources":[{"n":1,"title":"Ginseng","origin":"Wikipedia","url":"https://en.wikipedia.org/wiki/Ginseng"}]},
            {"title":"Bad","url":"http://elsewhere/x"}]}"#;
        let found = parse_knowledgecache(body, 5, "http://127.0.0.1:8205").unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].title, "ginseng (dictionary)");
        assert!(found[0].snippet.contains("definition from wordnet"));
        assert_eq!(found[1].title, "Ginseng (Herbal Medicine)");
        assert_eq!(found[1].url, "http://127.0.0.1:8205/article/ginseng");
        assert!(
            found[1]
                .snippet
                .contains("Written from: Ginseng (Wikipedia)."),
            "{}",
            found[1].snippet
        );
        assert!(parse_knowledgecache("{}", 5, "http://k").is_err());
    }

    #[test]
    fn local_sources_refuse_public_addresses() {
        let endpoints = Endpoints {
            herbal: "http://93.184.216.34/api/search".into(),
            ..Endpoints::default()
        };
        let error = search(&endpoints, "sleep", &["herbal".into()], 3)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("must be on this computer or the local network"),
            "{error}"
        );
        assert!(is_local("nomad") && !is_local("wikipedia"));
        assert_eq!(mode_sources("herbal").unwrap(), ["herbal", "nomad_health"]);
    }

    /// Against the real home server: cargo test --lib live_home_server -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_home_server_sources_answer() {
        let endpoints = Endpoints::default();
        for (mode, query) in [
            ("knowledge", "ginseng"),
            ("herbal", "insomnia"),
            ("herbal", "chest pain and trouble breathing"),
            ("offline", "water purification"),
            ("bible", "Enoch"),
        ] {
            let found = search(&endpoints, query, &mode_sources(mode).unwrap(), 4).unwrap();
            println!("== {mode}: {query}");
            for r in &found.reports {
                println!(
                    "   [{}] ok={} returned={} {}",
                    r.source,
                    r.ok,
                    r.returned,
                    r.error.clone().unwrap_or_default()
                );
            }
            for r in &found.results {
                println!(
                    "   - {} <{}>\n     {}",
                    r.title,
                    r.url,
                    r.snippet.chars().take(160).collect::<String>()
                );
            }
            assert!(!found.results.is_empty(), "{mode} returned nothing");
        }
    }

    #[test]
    fn health_questions_are_recognised() {
        for q in [
            "tell me about herbs",
            "are herbs used as medicine before drugs?",
            "what helps insomnia",
            "valerian dose",
        ] {
            assert!(is_health_question(q), "{q}");
        }
        for q in [
            "research isshin ryu karate do",
            "write a dos application",
            "compare lisp and scheme",
        ] {
            assert!(!is_health_question(q), "{q}");
        }
    }

    #[test]
    fn long_queries_shrink_to_key_words_for_strict_sources() {
        assert_eq!(
            key_words("how to edit or improve llm weights fine-tuning PEFT LoRA").as_deref(),
            Some("edit improve llm")
        );
        assert_eq!(key_words("PEFT LoRA"), None, "already short");
        assert_eq!(key_words("how to"), None, "nothing but common words");
    }

    #[test]
    fn modes_choose_sources_and_unknown_modes_are_refused() {
        assert_eq!(mode_sources("academic").unwrap(), ["arxiv", "crossref"]);
        assert_eq!(
            mode_sources("general").unwrap(),
            ["wikipedia", "hackernews"]
        );
        assert!(mode_sources("everything").is_none());
        assert!(MODES.iter().all(|m| mode_sources(m).is_some()));
        assert_eq!(label("arxiv"), "arXiv");
        assert_eq!(host("github"), "api.github.com");
    }

    #[test]
    fn wikipedia_results_are_ordered_by_relevance_and_keep_their_intro_text() {
        let found = parse_wikipedia(WIKI, 5).unwrap();
        assert_eq!(found[0].title, "Rust");
        assert_eq!(found[0].url, "https://en.wikipedia.org/wiki/Rust");
        assert_eq!(
            found[1].url,
            "https://en.wikipedia.org/wiki/Rust_(programming_language)"
        );
        assert!(found[1]
            .snippet
            .contains("emphasizing performance, type safety, and concurrency"));
        assert_eq!(parse_wikipedia(WIKI, 1).unwrap().len(), 1);
        assert!(parse_wikipedia("<html>", 3).is_err());
    }

    #[test]
    fn hacker_news_falls_back_to_the_discussion_link_and_skips_untitled_hits() {
        let found = parse_hackernews(HN, 5).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].url, "https://blog.example/rust");
        assert!(found[0].snippet.contains("420 points and 88 comments"));
        assert_eq!(found[1].url, "https://news.ycombinator.com/item?id=12");
    }

    #[test]
    fn arxiv_titles_are_cleaned_and_a_non_feed_is_an_error() {
        let found = parse_arxiv(ARXIV, 5).unwrap();
        assert_eq!(found[0].title, "Safe Systems Programming & Ownership");
        assert_eq!(found[0].url, "http://arxiv.org/abs/2101.00001v1");
        assert!(found[0].snippet.starts_with("We study how ownership types"));
        assert!(parse_arxiv("Rate exceeded.", 5).is_err());
        let terms = arxiv_query("how do ownership types prevent data races").unwrap();
        for word in ["ownership", "types", "prevent", "data", "races"] {
            assert!(terms.contains(&format!("all:{word}")), "{terms}");
        }
        assert!(
            terms.contains(" AND ") && !terms.contains("all:how"),
            "{terms}"
        );
        assert!(arxiv_query("?!").is_err());
    }

    #[test]
    fn crossref_abstract_markup_is_removed_and_a_missing_abstract_still_describes_the_paper() {
        let found = parse_crossref(CROSSREF, 5).unwrap();
        assert_eq!(found[0].snippet, "Ownership types restrict aliasing.");
        assert_eq!(found[1].snippet, "Published in 2020 in Proc. Safety.");
    }

    #[test]
    fn github_repositories_carry_description_stars_and_language() {
        let found = parse_github(GITHUB, 5).unwrap();
        assert_eq!(
            found[0].snippet,
            "Empowering everyone to build reliable software. 99000 stars, written in Rust."
        );
        assert_eq!(found[1].snippet, "5 stars.");
    }

    /// A local stand-in for the public APIs: answers by path, and records each request.
    fn serve(routes: Vec<(&'static str, u16, &'static str)>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buffer = [0u8; 4096];
                let n = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..n]).to_string();
                let line = request.lines().next().unwrap_or("").to_string();
                log.lock().unwrap().push(line.clone());
                let (status, body) = routes
                    .iter()
                    .find(|(prefix, _, _)| line.contains(&format!(" {prefix}")))
                    .map_or((404, "not found"), |(_, s, b)| (*s, *b));
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        (base, seen)
    }

    #[test]
    fn sources_are_asked_separately_and_results_are_interleaved_without_duplicates() {
        let (base, seen) = serve(vec![("/wiki", 200, WIKI), ("/hn", 200, HN)]);
        let endpoints = Endpoints {
            wikipedia: format!("{base}/wiki"),
            hackernews: format!("{base}/hn"),
            ..Endpoints::default()
        };
        let found = search(
            &endpoints,
            "rust ownership",
            &["wikipedia".into(), "hackernews".into()],
            4,
        )
        .unwrap();
        let order: Vec<&str> = found.results.iter().map(|r| r.source.as_str()).collect();
        assert_eq!(
            order,
            ["wikipedia", "hackernews", "wikipedia", "hackernews"]
        );
        assert!(found.reports.iter().all(|r| r.ok && r.returned > 0));
        let requests = seen.lock().unwrap().clone();
        assert_eq!(requests.len(), 2);
        assert!(
            requests[0].contains("gsrsearch=rust+ownership")
                || requests[0].contains("gsrsearch=rust%20ownership"),
            "{requests:?}"
        );
        assert!(
            requests[1].contains("query=rust+ownership") && requests[1].contains("tags=story"),
            "{requests:?}"
        );
    }

    #[test]
    fn a_source_that_refuses_is_reported_while_the_others_still_answer() {
        let (base, _) = serve(vec![("/wiki", 200, WIKI), ("/hn", 429, "slow down")]);
        let endpoints = Endpoints {
            wikipedia: format!("{base}/wiki"),
            hackernews: format!("{base}/hn"),
            ..Endpoints::default()
        };
        let found = search(
            &endpoints,
            "rust",
            &["wikipedia".into(), "hackernews".into()],
            5,
        )
        .unwrap();
        assert!(found.results.iter().all(|r| r.source == "wikipedia"));
        let hn = found
            .reports
            .iter()
            .find(|r| r.source == "hackernews")
            .unwrap();
        assert!(
            !hn.ok
                && hn
                    .error
                    .as_deref()
                    .unwrap()
                    .contains("limits how often it can be asked"),
            "{hn:?}"
        );
    }

    #[test]
    fn when_every_source_fails_the_error_says_why_for_each() {
        let (base, _) = serve(vec![("/wiki", 500, "boom")]);
        let endpoints = Endpoints {
            wikipedia: format!("{base}/wiki"),
            hackernews: "http://127.0.0.1:1/".into(),
            ..Endpoints::default()
        };
        let error = search(
            &endpoints,
            "rust",
            &["wikipedia".into(), "hackernews".into()],
            5,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.starts_with("No knowledge source answered:"),
            "{error}"
        );
        assert!(
            error.contains("Wikipedia: Wikipedia answered with an error (500")
                && error.contains("Hacker News:"),
            "{error}"
        );
        assert!(search(&endpoints, "  ", &["wikipedia".into()], 5).is_err());
        assert!(search(&endpoints, "rust", &[], 5).is_err());
        assert!(search(&endpoints, "rust", &["nonsense".into()], 5).is_err());
    }
}

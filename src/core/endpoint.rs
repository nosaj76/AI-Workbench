//! One canonical Ollama endpoint policy shared by configuration and HTTP transport.
use anyhow::{bail, Context, Result};
use url::{Host, Url};

pub fn normalize(value: &str) -> Result<Url> {
    let mut url =
        Url::parse(value.trim()).context("Enter a complete Ollama http:// or https:// URL")?;
    if !matches!(url.scheme(), "http" | "https") || url.host().is_none() {
        bail!("Ollama URL must use http or https and contain a host");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("Credentials in the Ollama URL are not supported");
    }
    if url.query().is_some() || url.fragment().is_some() {
        bail!("Ollama base URL must not contain a query or fragment");
    }
    let path = url.path().trim_end_matches('/');
    let path = path.strip_suffix("/api").unwrap_or(path);
    let normalized = format!("{path}/");
    url.set_path(&normalized);
    Ok(url)
}

pub fn is_loopback(url: &Url) -> bool {
    match url.host() {
        Some(Host::Domain(name)) => name.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(ip)) => ip.is_loopback(),
        Some(Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}

pub fn validate(value: &str, allow_remote: bool) -> Result<Url> {
    let url = normalize(value)?;
    if !allow_remote && !is_loopback(&url) {
        bail!("Remote endpoint is disabled. Enable remote inference in Settings before sending content to {}", url.host_str().unwrap_or("unknown"));
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equivalent_api_roots_normalize_together() {
        for value in [
            " http://127.0.0.1:11434 ",
            "http://127.0.0.1:11434/api",
            "http://127.0.0.1:11434/api/",
        ] {
            assert_eq!(
                normalize(value).unwrap().as_str(),
                "http://127.0.0.1:11434/"
            );
        }
        assert_eq!(
            normalize("https://example.com/prefix/api/")
                .unwrap()
                .as_str(),
            "https://example.com/prefix/"
        );
    }
    #[test]
    fn local_policy_checks_the_host_not_a_substring() {
        for url in [
            "http://localhost:11434",
            "http://127.12.0.2:11434",
            "http://[::1]:11434",
        ] {
            assert!(validate(url, false).is_ok(), "{url}");
        }
        for url in [
            "http://localhost.example.com",
            "http://127.0.0.1.example.com",
            "http://192.168.1.2",
        ] {
            assert!(validate(url, false).is_err(), "{url}");
            assert!(validate(url, true).is_ok());
        }
        for url in [
            "file:///tmp/data",
            "http://user:secret@localhost",
            "http://localhost/?q=a",
            "http://localhost/#x",
        ] {
            assert!(validate(url, true).is_err());
        }
    }
}

//! Recognize tool-shaped *text* without ever treating it as an executable call.
//!
//! Ollama exposes trusted tool calls in `message.tool_calls`, not in `content`.
//! Some models instead emit consecutive JSON objects such as
//! `{...}{...}{...}`. We detect that entire answer to request a native retry,
//! but NEVER construct a ToolCall or run a tool from the text itself.
use serde_json::Value;
use std::borrow::Cow;

const MAX_TOOL_TEXT_BYTES: usize = 128 * 1024;
const MAX_TEXTUAL_REQUESTS: usize = 12;

/// Identify complete tool-shaped JSON for read-only display and diagnostics.
/// This does not prove a function exists, nor does it authorize execution.
pub(super) fn textual_request_names(content: &str) -> Option<Vec<String>> {
    if content.len() > MAX_TOOL_TEXT_BYTES {
        return None;
    }
    let text = content.trim();
    let text = if let Some(rest) = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
    {
        rest.trim().strip_suffix("```")?.trim()
    } else {
        text
    };
    let text = unwrap_tool_call_tags(text)?;
    let text = text.as_ref();
    if text.is_empty() {
        return None;
    }
    // StreamDeserializer handles adjacent JSON values without separators. If
    // any part of the answer is ordinary prose or malformed JSON, reject it.
    let stream = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    let mut names = Vec::new();
    for value in stream {
        let value = value.ok()?;
        let requests = match value {
            Value::Array(requests) => requests,
            other => vec![other],
        };
        if requests.is_empty() || names.len() + requests.len() > MAX_TEXTUAL_REQUESTS {
            return None;
        }
        for request in requests {
            // Permit {name,arguments}, Ollama's {function:{...}} envelope,
            // or the standard {type:"function",function:{...}} envelope, but
            // no extra fields or embedded instructions.
            let object = request.as_object()?;
            let function = if object.contains_key("function")
                && (object.len() == 1
                    || (object.len() == 2
                        && object.get("type").and_then(Value::as_str) == Some("function")))
            {
                object.get("function")?.as_object()?
            } else {
                object
            };
            if function.len() != 2 {
                return None;
            }
            let arguments = function.get("arguments")?;
            let arguments_are_object = arguments.is_object()
                || arguments
                    .as_str()
                    .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                    .is_some_and(|value| value.is_object());
            if !arguments_are_object {
                return None;
            }
            let name = function.get("name")?.as_str()?;
            if name.is_empty()
                || name.len() > 128
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
            {
                return None;
            }
            names.push(name.to_owned());
        }
    }
    (!names.is_empty()).then_some(names)
}

/// Tool requests the model wrote as JSON somewhere inside ordinary prose, for
/// example "I'll read it: {"name":"read_file","arguments":{...}}". Only names of
/// tools actually offered this turn count. Diagnostic only: never executed.
pub(super) fn embedded_request_names(content: &str, definitions: &[Value]) -> Option<Vec<String>> {
    if content.len() > MAX_TOOL_TEXT_BYTES || definitions.is_empty() {
        return None;
    }
    let offered = |name: &str| {
        definitions
            .iter()
            .any(|d| d.pointer("/function/name").and_then(Value::as_str) == Some(name))
    };
    let mut names = Vec::new();
    for (index, _) in content.match_indices('{').take(500) {
        let Some(Ok(value)) = serde_json::Deserializer::from_str(&content[index..])
            .into_iter::<Value>()
            .next()
        else {
            continue;
        };
        let function = value
            .get("function")
            .filter(|f| f.is_object())
            .unwrap_or(&value);
        let (Some(name), Some(arguments)) = (
            function.get("name").and_then(Value::as_str),
            function.get("arguments"),
        ) else {
            continue;
        };
        // Only JSON the model presents as its own next action counts: an
        // explanation that shows an example of tool JSON is not a request.
        let lead: String = content[..index]
            .chars()
            .rev()
            .take(100)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>()
            .to_lowercase();
        let acting = [
            "i'll",
            "i will",
            "let me",
            "calling",
            "i'm going to",
            "i am going to",
            "now i",
            "next i",
            "i need to",
        ]
        .iter()
        .any(|phrase| lead.contains(phrase))
            && ![
                "example",
                "for instance",
                "e.g.",
                "such as",
                "format",
                "looks like",
            ]
            .iter()
            .any(|phrase| lead.contains(phrase));
        if acting
            && (arguments.is_object() || arguments.is_string())
            && offered(name)
            && !names.iter().any(|n| n == name)
        {
            names.push(name.to_owned());
        }
        if names.len() >= MAX_TEXTUAL_REQUESTS {
            break;
        }
    }
    (!names.is_empty()).then_some(names)
}

/// Every form of tool request written as text: a whole answer of tool JSON, or
/// tool JSON naming an offered tool inside prose.
pub(super) fn request_names(content: &str, definitions: &[Value]) -> Option<Vec<String>> {
    textual_request_names(content).or_else(|| embedded_request_names(content, definitions))
}

/// Some tool-capable model templates print one or more `<tool_call>` wrappers
/// around otherwise valid JSON. The wrappers are still untrusted text; this
/// helper only makes the diagnostic detector recognize the complete answer.
fn unwrap_tool_call_tags(text: &str) -> Option<Cow<'_, str>> {
    let mut remaining = text.trim();
    if !remaining.starts_with("<tool_call>") {
        return Some(Cow::Borrowed(remaining));
    }
    let mut joined = String::new();
    while !remaining.is_empty() {
        let body = remaining.strip_prefix("<tool_call>")?;
        let end = body.find("</tool_call>")?;
        let object = body[..end].trim();
        if object.is_empty() {
            return None;
        }
        joined.push_str(object);
        joined.push('\n');
        remaining = body[end + "</tool_call>".len()..].trim();
    }
    Some(Cow::Owned(joined))
}

/// Names the model wrote as a command-style call in its prose, for example a line
/// `web_search query="latest methods"` (usually inside a code fence). This is how a model
/// "uses" a tool it was never given. The text is never executed; the names only let the
/// application tell the user that nothing ran.
pub(super) fn pseudo_call_names(content: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in content.lines() {
        let line = line.trim().trim_start_matches(['$', '>']).trim();
        let Some((name, rest)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let rest = rest.trim_start();
        let is_name = name.len() <= 64
            && name.contains('_')
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        let key = rest
            .split_once('=')
            .map(|(key, value)| (key, value.trim_start()));
        let is_keyed = key.is_some_and(|(key, value)| {
            !key.is_empty()
                && key.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
                && value.starts_with(['"', '\''])
        });
        if is_name && is_keyed && !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
    }
    names
}

pub(super) fn unavailable_names(names: &[String], definitions: &[Value]) -> Vec<String> {
    names
        .iter()
        .filter(|name| {
            !definitions
                .iter()
                .any(|d| d["function"]["name"].as_str() == Some(name.as_str()))
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_json_inside_prose_is_detected_only_for_offered_tools() {
        let defs = vec![
            serde_json::json!({"type":"function","function":{"name":"read_file","parameters":{}}}),
        ];
        let text = "Sure, I'll read it now: {\"name\": \"read_file\", \"arguments\": {\"path\": \"main.rs\"}} and then summarize.";
        assert!(textual_request_names(text).is_none());
        assert_eq!(
            embedded_request_names(text, &defs).unwrap(),
            vec!["read_file"]
        );
        assert!(
            embedded_request_names("Config: {\"name\": \"x\", \"arguments\": {}}", &defs).is_none()
        );
        assert!(embedded_request_names("A plain {brace} in text", &defs).is_none());
        let example =
            "Here is an example: {\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}";
        assert!(embedded_request_names(example, &defs).is_none());
    }

    #[test]
    fn command_style_calls_in_prose_are_named_but_ordinary_text_is_not() {
        let text = "I'll search.\n\n```bash\nweb_search query=\"latest methods 2025\"\n```";
        assert_eq!(pseudo_call_names(text), vec!["web_search"]);
        assert!(pseudo_call_names("set the path = \"x\" and read_me please").is_empty());
        assert!(pseudo_call_names("export PATH=\"/bin\"").is_empty());
        assert!(pseudo_call_names("name = \"value\"").is_empty());
    }
    use serde_json::json;

    fn definitions() -> Vec<Value> {
        vec![
            json!({"function":{"name":"project_tree"}}),
            json!({"function":{"name":"read_file"}}),
            json!({"function":{"name":"search_files"}}),
        ]
    }

    #[test]
    fn recognizes_single_and_concatenated_tool_text_without_executing_it() {
        let defs = definitions();
        let single = r#"{"name":"project_tree","arguments":{"path":"."}}"#;
        assert_eq!(
            textual_request_names(single),
            Some(vec!["project_tree".into()])
        );
        assert!(unavailable_names(&textual_request_names(single).unwrap(), &defs).is_empty());
        let repeated = [
            single,
            r#"{"name":"read_file","arguments":{"path":"src/main.rs"}}"#,
            r#"{"name":"search_files","arguments":{"query":"test"}}"#,
        ]
        .concat();
        assert_eq!(
            textual_request_names(&repeated),
            Some(vec![
                "project_tree".into(),
                "read_file".into(),
                "search_files".into(),
            ])
        );
        assert_eq!(
            textual_request_names(&format!("```json\n{repeated}\n```"))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            textual_request_names(&format!("[{single}]")),
            Some(vec!["project_tree".into()])
        );
        assert_eq!(
            textual_request_names(r#"{"function":{"name":"read_file","arguments":{"path":"."}}}"#),
            Some(vec!["read_file".into()])
        );
        assert_eq!(
            textual_request_names(
                r#"{"type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/lib.rs\"}"}}"#
            ),
            Some(vec!["read_file".into()])
        );
        assert_eq!(textual_request_names(&format!("<tool_call>{single}</tool_call><tool_call>{{\"name\":\"read_file\",\"arguments\":{{\"path\":\"src/lib.rs\"}}}}</tool_call>")), Some(vec!["project_tree".into(), "read_file".into()]));
    }

    #[test]
    fn prose_unknown_tools_and_partial_json_are_never_classified_as_calls() {
        let defs = definitions();
        for text in [
            "Here is an example: {\"name\":\"project_tree\",\"arguments\":{}}",
            r#"{"name":"project_tree","arguments":{},"explanation":"example"}"#,
            r#"{"name":"project_tree","arguments":"not an object"}"#,
            r#"{"name":"project_tree","arguments":{}} untrusted text"#,
            "[{}]",
            "[]",
            "{\"name\":\"project_tree\",\"arguments\":{}",
            r#"<tool_call>{"name":"read_file","arguments":{}}</tool_call> trailing prose"#,
            r#"{"name":"bad\nname","arguments":{}}"#,
        ] {
            assert_eq!(textual_request_names(text), None, "text={text}");
        }
        assert_eq!(
            unavailable_names(&["run_command".into(), "read_file".into()], &defs),
            vec!["run_command".to_string()]
        );
        let too_many = r#"{"name":"read_file","arguments":{}}"#.repeat(13);
        assert!(textual_request_names(&too_many).is_none());
    }
}

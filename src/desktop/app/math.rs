//! Readable math in chat answers.
//!
//! egui cannot typeset LaTeX, so formulas in `$…$`, `$$…$$`, `\(…\)` and `\[…\]`
//! are rewritten as Unicode text (`\frac{a}{b} + x^2 \leq \pi` → `a/b + x² ≤ π`)
//! before the Markdown is rendered. Code blocks and inline code are left alone,
//! and dollar amounts such as "$5 and $10" are not treated as math.
use std::borrow::Cow;

/// Markdown with every recognised formula replaced by readable Unicode text.
pub(super) fn render_math(markdown: &str) -> Cow<'_, str> {
    if !markdown.contains('$') && !markdown.contains("\\(") && !markdown.contains("\\[") {
        return Cow::Borrowed(markdown);
    }
    let mut out = String::with_capacity(markdown.len());
    let mut fence: Option<String> = None;
    let mut prose = String::new();
    for line in markdown.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let marker: String = trimmed
            .chars()
            .take_while(|c| *c == '`' || *c == '~')
            .collect();
        if let Some(open) = &fence {
            out.push_str(line);
            if marker.len() >= open.len() && marker.starts_with(&open[..1]) {
                fence = None;
            }
            continue;
        }
        if marker.len() >= 3 && (marker.chars().all(|c| c == '`') || marker.chars().all(|c| c == '~')) {
            out.push_str(&convert_prose(&std::mem::take(&mut prose)));
            out.push_str(line);
            fence = Some(marker);
            continue;
        }
        prose.push_str(line);
    }
    out.push_str(&convert_prose(&prose));
    if out == markdown {
        Cow::Borrowed(markdown)
    } else {
        Cow::Owned(out)
    }
}

/// Converts formulas in text that is outside fenced code blocks.
fn convert_prose(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // Inline code spans are copied unchanged.
        if c == '`' {
            let ticks = run_length(&chars, i, '`');
            if let Some(end) = find_backtick_close(&chars, i + ticks, ticks) {
                out.extend(&chars[i..end + ticks]);
                i = end + ticks;
            } else {
                out.extend(&chars[i..i + ticks]);
                i += ticks;
            }
            continue;
        }
        // An escaped dollar is a literal dollar.
        if c == '\\' && chars.get(i + 1) == Some(&'$') {
            out.push_str("\\$");
            i += 2;
            continue;
        }
        if let Some((body, end, display)) = math_at(&chars, i) {
            let rendered = escape_markdown(&latex_to_unicode(&body));
            if display {
                out.push_str("\n\n");
                out.push_str(rendered.trim());
                out.push_str("\n\n");
            } else {
                out.push_str(&rendered);
            }
            i = end;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// A formula starting at `i`: its LaTeX, the index after it, and whether it is display math.
fn math_at(chars: &[char], i: usize) -> Option<(String, usize, bool)> {
    let starts = |s: &str| s.chars().enumerate().all(|(k, ch)| chars.get(i + k) == Some(&ch));
    let find = |from: usize, close: &str| -> Option<usize> {
        let close: Vec<char> = close.chars().collect();
        (from..chars.len()).find(|&k| chars[k..].starts_with(&close))
    };
    if starts("$$") {
        let end = find(i + 2, "$$")?;
        let body: String = chars[i + 2..end].iter().collect();
        return (!body.trim().is_empty()).then(|| (body, end + 2, true));
    }
    if starts("\\[") {
        let end = find(i + 2, "\\]")?;
        let body: String = chars[i + 2..end].iter().collect();
        return (!body.trim().is_empty()).then(|| (body, end + 2, true));
    }
    if starts("\\(") {
        let end = find(i + 2, "\\)")?;
        let body: String = chars[i + 2..end].iter().collect();
        return (!body.trim().is_empty() && !body.contains('\n')).then(|| (body, end + 2, false));
    }
    if chars[i] == '$' {
        // Pandoc's rule: no space after the opening $, no space before the closing
        // $, and the closing $ is not followed by a digit. "$5 and $10" stays text.
        let next = *chars.get(i + 1)?;
        if next.is_whitespace() || next.is_ascii_digit() || next == '$' {
            return None;
        }
        let mut k = i + 1;
        while k < chars.len() {
            match chars[k] {
                '\n' => return None,
                '\\' => k += 2,
                '$' => {
                    let before = chars[k - 1];
                    let after = chars.get(k + 1).copied();
                    if !before.is_whitespace() && !after.is_some_and(|a| a.is_ascii_digit()) {
                        let body: String = chars[i + 1..k].iter().collect();
                        return Some((body, k + 1, false));
                    }
                    return None;
                }
                _ => k += 1,
            }
        }
    }
    None
}

fn run_length(chars: &[char], from: usize, c: char) -> usize {
    chars[from..].iter().take_while(|&&x| x == c).count()
}

fn find_backtick_close(chars: &[char], from: usize, ticks: usize) -> Option<usize> {
    let mut k = from;
    while k < chars.len() {
        if chars[k] == '`' {
            let run = run_length(chars, k, '`');
            if run == ticks {
                return Some(k);
            }
            k += run;
        } else {
            k += 1;
        }
    }
    None
}

/// Keeps Markdown from reinterpreting characters in the converted formula.
fn escape_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if matches!(c, '\\' | '*' | '_' | '`' | '[' | ']' | '<' | '>' | '#' | '|' | '$') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// LaTeX math as readable Unicode text. Unknown commands keep their name.
pub(super) fn latex_to_unicode(latex: &str) -> String {
    let chars: Vec<char> = latex.chars().collect();
    let mut parser = Parser { chars, pos: 0 };
    let out = parser.sequence(None);
    // Collapse runs of spaces left by spacing commands and removed groups.
    let mut tidy = String::with_capacity(out.len());
    for c in out.chars() {
        if c == ' ' && tidy.ends_with(' ') {
            continue;
        }
        tidy.push(c);
    }
    tidy.trim().to_string()
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// Parses until `stop` (a closing brace) or the end.
    fn sequence(&mut self, stop: Option<char>) -> String {
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if Some(c) == stop {
                break;
            }
            match c {
                '{' => {
                    self.pos += 1;
                    out.push_str(&self.sequence(Some('}')));
                    self.pos += 1;
                }
                '}' => self.pos += 1,
                '^' | '_' => {
                    self.pos += 1;
                    let arg = self.argument();
                    out.push_str(&script(&arg, c == '^'));
                }
                '\\' => out.push_str(&self.command()),
                '&' => {
                    self.pos += 1;
                    out.push(' ');
                }
                '~' => {
                    self.pos += 1;
                    out.push(' ');
                }
                '\n' => {
                    self.pos += 1;
                    out.push(' ');
                }
                _ => {
                    self.pos += 1;
                    out.push(c);
                }
            }
        }
        out
    }

    /// One argument: a braced group, a command, or a single character.
    fn argument(&mut self) -> String {
        while self.peek() == Some(' ') {
            self.pos += 1;
        }
        match self.peek() {
            Some('{') => {
                self.pos += 1;
                let inner = self.sequence(Some('}'));
                self.pos += 1;
                inner
            }
            Some('\\') => self.command(),
            Some(c) => {
                self.pos += 1;
                c.to_string()
            }
            None => String::new(),
        }
    }

    fn command(&mut self) -> String {
        self.pos += 1; // the backslash
        let Some(first) = self.peek() else {
            return String::new();
        };
        if !first.is_ascii_alphabetic() {
            self.pos += 1;
            return match first {
                '\\' => " ".into(),
                ',' | ';' | ':' | '!' | ' ' => " ".into(),
                '{' => "{".into(),
                '}' => "}".into(),
                '|' => "‖".into(),
                other => other.to_string(),
            };
        }
        let mut name = String::new();
        while let Some(c) = self.peek().filter(char::is_ascii_alphabetic) {
            name.push(c);
            self.pos += 1;
        }
        match name.as_str() {
            "frac" | "dfrac" | "tfrac" => {
                let numerator = self.argument();
                let denominator = self.argument();
                format!("{}/{}", group(&numerator), group(&denominator))
            }
            "sqrt" => {
                let mut root = String::new();
                if self.peek() == Some('[') {
                    self.pos += 1;
                    while let Some(c) = self.peek().filter(|c| *c != ']') {
                        root.push(c);
                        self.pos += 1;
                    }
                    self.pos += 1;
                }
                let radicand = self.argument();
                let symbol = match root.trim() {
                    "" | "2" => "√",
                    "3" => "∛",
                    "4" => "∜",
                    other => return format!("{}√{}", script(other, true), group(&radicand)),
                };
                format!("{symbol}{}", group(&radicand))
            }
            "text" | "mathrm" | "mathbf" | "mathit" | "mathsf" | "mathtt" | "textbf"
            | "textit" | "operatorname" | "boldsymbol" | "mbox" => self.argument(),
            "mathbb" => {
                let arg = self.argument();
                arg.chars().map(double_struck).collect()
            }
            "left" | "right" | "big" | "Big" | "bigg" | "Bigg" | "displaystyle" | "limits" => {
                // Sizing commands: drop them and keep the delimiter that follows.
                if self.peek() == Some('.') {
                    self.pos += 1;
                }
                String::new()
            }
            "quad" | "qquad" => " ".into(),
            "begin" | "end" => {
                self.argument(); // environment name
                " ".into()
            }
            other => symbol(other).map_or_else(|| format!("\\{other}"), str::to_string),
        }
    }
}

/// Wraps a multi-part fraction or root operand in parentheses: (a+b)/2.
fn group(text: &str) -> String {
    let text = text.trim();
    // One symbol or a plain number needs no parentheses; "2a" does, so that
    // (x+1)/(2a) is not read as ((x+1)/2)·a.
    let simple = text.chars().count() == 1
        || text.chars().all(|c| c.is_ascii_digit() || c == '.')
        || (text.starts_with('(') && text.ends_with(')') && balanced_outer(text));
    if simple {
        text.to_string()
    } else {
        format!("({text})")
    }
}

/// True when the first "(" closes at the last character: "(a+b)" but not "(a)+(b)".
fn balanced_outer(text: &str) -> bool {
    let mut depth = 0;
    for (index, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 && index + 1 < text.len() {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// Superscript or subscript, using Unicode characters when every one exists.
fn script(text: &str, superscript: bool) -> String {
    let text = text.trim();
    let mapped: Option<String> = text
        .chars()
        .map(|c| if superscript { superscript_char(c) } else { subscript_char(c) })
        .collect();
    match mapped {
        Some(s) if !s.is_empty() => s,
        _ => {
            let mark = if superscript { '^' } else { '_' };
            if text.chars().count() == 1 {
                format!("{mark}{text}")
            } else {
                format!("{mark}({text})")
            }
        }
    }
}

fn superscript_char(c: char) -> Option<char> {
    Some(match c {
        '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
        '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
        '+' => '⁺', '-' | '−' => '⁻', '=' => '⁼', '(' => '⁽', ')' => '⁾',
        'n' => 'ⁿ', 'i' => 'ⁱ', 'a' => 'ᵃ', 'b' => 'ᵇ', 'c' => 'ᶜ', 'd' => 'ᵈ',
        'e' => 'ᵉ', 'f' => 'ᶠ', 'g' => 'ᵍ', 'h' => 'ʰ', 'j' => 'ʲ', 'k' => 'ᵏ',
        'l' => 'ˡ', 'm' => 'ᵐ', 'o' => 'ᵒ', 'p' => 'ᵖ', 'r' => 'ʳ', 's' => 'ˢ',
        't' => 'ᵗ', 'u' => 'ᵘ', 'v' => 'ᵛ', 'w' => 'ʷ', 'x' => 'ˣ', 'y' => 'ʸ',
        'z' => 'ᶻ', 'T' => 'ᵀ', '′' => '′', ' ' => ' ',
        _ => return None,
    })
}

fn subscript_char(c: char) -> Option<char> {
    Some(match c {
        '0' => '₀', '1' => '₁', '2' => '₂', '3' => '₃', '4' => '₄',
        '5' => '₅', '6' => '₆', '7' => '₇', '8' => '₈', '9' => '₉',
        '+' => '₊', '-' | '−' => '₋', '=' => '₌', '(' => '₍', ')' => '₎',
        'a' => 'ₐ', 'e' => 'ₑ', 'h' => 'ₕ', 'i' => 'ᵢ', 'j' => 'ⱼ', 'k' => 'ₖ',
        'l' => 'ₗ', 'm' => 'ₘ', 'n' => 'ₙ', 'o' => 'ₒ', 'p' => 'ₚ', 'r' => 'ᵣ',
        's' => 'ₛ', 't' => 'ₜ', 'u' => 'ᵤ', 'v' => 'ᵥ', 'x' => 'ₓ', ' ' => ' ',
        _ => return None,
    })
}

fn double_struck(c: char) -> char {
    match c {
        'R' => 'ℝ', 'N' => 'ℕ', 'Z' => 'ℤ', 'Q' => 'ℚ', 'C' => 'ℂ', 'P' => 'ℙ', 'H' => 'ℍ',
        other => other,
    }
}

fn symbol(name: &str) -> Option<&'static str> {
    Some(match name {
        // Greek
        "alpha" => "α", "beta" => "β", "gamma" => "γ", "delta" => "δ", "epsilon" => "ε",
        "varepsilon" => "ε", "zeta" => "ζ", "eta" => "η", "theta" => "θ", "vartheta" => "ϑ",
        "iota" => "ι", "kappa" => "κ", "lambda" => "λ", "mu" => "μ", "nu" => "ν", "xi" => "ξ",
        "pi" => "π", "varpi" => "ϖ", "rho" => "ρ", "sigma" => "σ", "tau" => "τ",
        "upsilon" => "υ", "phi" => "φ", "varphi" => "φ", "chi" => "χ", "psi" => "ψ",
        "omega" => "ω", "Gamma" => "Γ", "Delta" => "Δ", "Theta" => "Θ", "Lambda" => "Λ",
        "Xi" => "Ξ", "Pi" => "Π", "Sigma" => "Σ", "Upsilon" => "Υ", "Phi" => "Φ",
        "Psi" => "Ψ", "Omega" => "Ω",
        // Operators and relations
        "times" => "×", "cdot" => "·", "div" => "÷", "pm" => "±", "mp" => "∓", "ast" => "∗",
        "leq" | "le" => "≤", "geq" | "ge" => "≥", "neq" | "ne" => "≠", "approx" => "≈",
        "equiv" => "≡", "sim" => "∼", "simeq" => "≃", "cong" => "≅", "propto" => "∝",
        "ll" => "≪", "gg" => "≫", "circ" => "∘", "bullet" => "•", "star" => "⋆",
        "oplus" => "⊕", "otimes" => "⊗", "perp" => "⊥", "parallel" => "∥", "mid" => "∣",
        // Big operators and calculus
        "sum" => "∑", "prod" => "∏", "coprod" => "∐", "int" => "∫", "iint" => "∬",
        "iiint" => "∭", "oint" => "∮", "partial" => "∂", "nabla" => "∇", "infty" => "∞",
        "lim" => "lim", "sin" => "sin", "cos" => "cos", "tan" => "tan", "log" => "log",
        "ln" => "ln", "exp" => "exp", "max" => "max", "min" => "min", "det" => "det",
        "sec" => "sec", "csc" => "csc", "cot" => "cot", "arcsin" => "arcsin",
        "arccos" => "arccos", "arctan" => "arctan", "sinh" => "sinh", "cosh" => "cosh",
        "tanh" => "tanh", "gcd" => "gcd", "deg" => "deg", "sup" => "sup", "inf" => "inf",
        // Sets and logic
        "in" => "∈", "notin" => "∉", "ni" => "∋", "subset" => "⊂", "supset" => "⊃",
        "subseteq" => "⊆", "supseteq" => "⊇", "cup" => "∪", "cap" => "∩",
        "emptyset" | "varnothing" => "∅", "forall" => "∀", "exists" => "∃", "neg" => "¬",
        "land" | "wedge" => "∧", "lor" | "vee" => "∨", "setminus" => "∖",
        // Arrows
        "to" | "rightarrow" => "→", "leftarrow" | "gets" => "←", "leftrightarrow" => "↔",
        "Rightarrow" | "implies" => "⇒", "Leftarrow" => "⇐", "Leftrightarrow" | "iff" => "⇔",
        "mapsto" => "↦", "uparrow" => "↑", "downarrow" => "↓",
        // Dots, delimiters and misc
        "ldots" | "dots" => "…", "cdots" => "⋯", "vdots" => "⋮", "ddots" => "⋱",
        "langle" => "⟨", "rangle" => "⟩", "lfloor" => "⌊", "rfloor" => "⌋", "lceil" => "⌈",
        "rceil" => "⌉", "vert" => "|", "Vert" => "‖", "angle" => "∠", "degree" => "°",
        "prime" => "′", "hbar" => "ℏ", "ell" => "ℓ", "Re" => "ℜ", "Im" => "ℑ",
        "aleph" => "ℵ", "triangle" => "△", "square" => "□", "therefore" => "∴",
        "because" => "∵", "checkmark" => "✓", "percent" => "%",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_formulas_become_readable_unicode() {
        assert_eq!(latex_to_unicode(r"\frac{a}{b} + x^2 \leq \pi"), "a/b + x² ≤ π");
        assert_eq!(latex_to_unicode(r"\frac{a+b}{2}"), "(a+b)/2");
        assert_eq!(latex_to_unicode(r"\sqrt{x^2 + y^2}"), "√(x² + y²)");
        assert_eq!(latex_to_unicode(r"\sqrt[3]{8}"), "∛8");
        assert_eq!(latex_to_unicode(r"\sum_{i=1}^{n} i"), "∑ᵢ₌₁ⁿ i");
        assert_eq!(latex_to_unicode(r"E = mc^2"), "E = mc²");
        assert_eq!(latex_to_unicode(r"\alpha \to \infty"), "α → ∞");
        assert_eq!(latex_to_unicode(r"x \in \mathbb{R}"), "x ∈ ℝ");
        assert_eq!(latex_to_unicode(r"\text{speed} = \frac{d}{t}"), "speed = d/t");
        assert_eq!(latex_to_unicode(r"\left( a \right)"), "( a )");
        // No Unicode superscript exists for every character: keep a readable form.
        assert_eq!(latex_to_unicode(r"e^{\pi}"), "e^π");
        assert_eq!(latex_to_unicode(r"x_{max}"), "xₘₐₓ");
        assert_eq!(latex_to_unicode(r"x_{QR}"), "x_(QR)");
        assert_eq!(latex_to_unicode(r"\frac{(a)+(b)}{c}"), "((a)+(b))/c");
        assert_eq!(latex_to_unicode(r"\unknowncmd{y}"), r"\unknowncmdy");
    }

    #[test]
    fn inline_and_display_delimiters_are_rendered() {
        assert_eq!(render_math(r"Area is $\pi r^2$."), "Area is π r².");
        assert_eq!(render_math(r"Area is \(\pi r^2\)."), "Area is π r².");
        assert_eq!(
            render_math("Solve:\n$$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$\nDone"),
            "Solve:\n\n\nx = (-b ± √(b²-4ac))/(2a)\n\n\nDone"
        );
        assert_eq!(render_math("A\n\\[a^2+b^2=c^2\\]\n"), "A\n\n\na²+b²=c²\n\n\n");
    }

    #[test]
    fn currency_code_and_escapes_are_left_alone() {
        for text in [
            "It costs $5 and $10 in total.",
            "Prices: $ 3 or $4.",
            "Use `$x^2$` literally.",
            "```latex\n$x^2$\n```\n",
            "Echo \\$HOME and \\$PATH.",
            "Plain text without formulas.",
        ] {
            assert_eq!(render_math(text), text, "{text}");
        }
        assert!(matches!(render_math("Plain text"), Cow::Borrowed(_)));
        // Math after a fenced block is still converted.
        assert_eq!(render_math("```\n$a$\n```\nthen $\\beta$"), "```\n$a$\n```\nthen β");
    }

    #[test]
    fn converted_text_cannot_turn_into_markdown_emphasis() {
        assert_eq!(render_math(r"$a*b*c$"), r"a\*b\*c");
        assert_eq!(render_math(r"$x_{QR}$"), r"x\_(QR)");
    }
}

//! Bounded floating-point arithmetic parser. No interpreter, shell or code evaluation.
//! Trigonometric arguments are radians. This is numeric arithmetic, not a CAS.

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Power,
    Left,
    Right,
    Comma,
}

pub fn evaluate(expression: &str) -> Result<f64> {
    if expression.trim().is_empty() || expression.len() > 8192 {
        bail!("Expression must contain 1–8192 bytes");
    }
    let tokens = tokenize(expression)?;
    let mut parser = Parser {
        tokens,
        position: 0,
        depth: 0,
    };
    let result = parser.expression(0)?;
    if parser.position != parser.tokens.len() {
        bail!("Unexpected token after the expression");
    }
    finite(result)
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut tokens = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let token = match b {
            b'+' => { i += 1; Token::Add }, b'-' => { i += 1; Token::Subtract },
            b'*' => { i += 1; Token::Multiply }, b'/' => { i += 1; Token::Divide },
            b'%' => { i += 1; Token::Remainder }, b'^' => { i += 1; Token::Power },
            b'(' => { i += 1; Token::Left }, b')' => { i += 1; Token::Right },
            b',' => { i += 1; Token::Comma },
            b'0'..=b'9' | b'.' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
                if bytes.get(i) == Some(&b'.') { i += 1; while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; } }
                if matches!(bytes.get(i), Some(b'e' | b'E')) {
                    i += 1; if matches!(bytes.get(i), Some(b'+' | b'-')) { i += 1; }
                    let exponent_start = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
                    if exponent_start == i { bail!("Number exponent requires digits"); }
                }
                let value = input[start..i].parse::<f64>().context("Invalid numeric literal")?;
                Token::Number(finite(value)?)
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = i; i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') { i += 1; }
                Token::Name(input[start..i].to_ascii_lowercase())
            }
            _ => bail!("Unsupported character at byte {}. Use numbers, arithmetic operators and named math functions.", i + 1),
        };
        tokens.push(token);
        if tokens.len() > 2048 {
            bail!("Expression exceeds 2048 tokens");
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
    depth: usize,
}

impl Parser {
    fn next(&mut self) -> Result<Token> {
        let token = self
            .tokens
            .get(self.position)
            .cloned()
            .context("Unexpected end of expression")?;
        self.position += 1;
        Ok(token)
    }

    fn expression(&mut self, minimum_binding: u8) -> Result<f64> {
        if self.depth >= 64 {
            bail!("Expression nesting exceeds 64 levels");
        }
        self.depth += 1;
        let result = self.expression_inner(minimum_binding);
        self.depth -= 1;
        result
    }

    fn expression_inner(&mut self, minimum_binding: u8) -> Result<f64> {
        let mut left = match self.next()? {
            Token::Number(n) => n,
            Token::Add => self.expression(6)?,
            Token::Subtract => -self.expression(6)?,
            Token::Left => {
                let value = self.expression(0)?;
                if self.next()? != Token::Right {
                    bail!("Expected closing parenthesis");
                }
                value
            }
            Token::Name(name) => {
                if self.tokens.get(self.position) == Some(&Token::Left) {
                    self.position += 1;
                    let mut args = Vec::new();
                    if self.tokens.get(self.position) != Some(&Token::Right) {
                        loop {
                            args.push(self.expression(0)?);
                            if args.len() > 8 {
                                bail!("Math function takes too many arguments");
                            }
                            if self.tokens.get(self.position) != Some(&Token::Comma) {
                                break;
                            }
                            self.position += 1;
                        }
                    }
                    if self.next()? != Token::Right {
                        bail!("Expected closing parenthesis after function arguments");
                    }
                    function(&name, &args)?
                } else {
                    match name.as_str() {
                        "pi" => std::f64::consts::PI,
                        "e" => std::f64::consts::E,
                        "tau" => std::f64::consts::TAU,
                        _ => {
                            bail!("Unknown constant {name}; available constants are pi, e and tau")
                        }
                    }
                }
            }
            _ => bail!("Expected a number, function or parenthesized expression"),
        };
        while let Some(operator) = self.tokens.get(self.position).cloned() {
            let (left_binding, right_binding) = match operator {
                Token::Add | Token::Subtract => (1, 2),
                Token::Multiply | Token::Divide | Token::Remainder => (3, 4),
                Token::Power => (7, 7), // right-associative; exponentiation precedes unary minus
                _ => break,
            };
            if left_binding < minimum_binding {
                break;
            }
            self.position += 1;
            let right = self.expression(right_binding)?;
            left = finite(match operator {
                Token::Add => left + right,
                Token::Subtract => left - right,
                Token::Multiply => left * right,
                Token::Divide => {
                    if right == 0.0 {
                        bail!("Division by zero");
                    }
                    left / right
                }
                Token::Remainder => {
                    if right == 0.0 {
                        bail!("Remainder by zero");
                    }
                    left % right
                }
                Token::Power => left.powf(right),
                _ => unreachable!(),
            })?;
        }
        finite(left)
    }
}

fn function(name: &str, args: &[f64]) -> Result<f64> {
    let unary = |f: fn(f64) -> f64| -> Result<f64> {
        if args.len() != 1 {
            bail!("{name} requires one argument");
        }
        finite(f(args[0]))
    };
    let binary = |f: fn(f64, f64) -> f64| -> Result<f64> {
        if args.len() != 2 {
            bail!("{name} requires two arguments");
        }
        finite(f(args[0], args[1]))
    };
    match name {
        "sqrt" => unary(f64::sqrt),
        "abs" => unary(f64::abs),
        "sin" => unary(f64::sin),
        "cos" => unary(f64::cos),
        "tan" => unary(f64::tan),
        "asin" => unary(f64::asin),
        "acos" => unary(f64::acos),
        "atan" => unary(f64::atan),
        "ln" => unary(f64::ln),
        "log" | "log10" => unary(f64::log10),
        "log2" => unary(f64::log2),
        "exp" => unary(f64::exp),
        "floor" => unary(f64::floor),
        "ceil" => unary(f64::ceil),
        "round" => unary(f64::round),
        "min" => binary(f64::min),
        "max" => binary(f64::max),
        "pow" => binary(f64::powf),
        "atan2" => binary(f64::atan2),
        "hypot" => binary(f64::hypot),
        _ => bail!("Unknown math function: {name}"),
    }
}

fn finite(value: f64) -> Result<f64> {
    if !value.is_finite() {
        bail!("Expression is outside the finite real-number range or function domain");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn precedence_associativity_and_scientific_notation() {
        assert_eq!(evaluate("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(evaluate("2^3^2").unwrap(), 512.0);
        assert_eq!(evaluate("-2^2").unwrap(), -4.0);
        assert_eq!(evaluate("(-2)^2 + 2^-2").unwrap(), 4.25);
        assert!((evaluate("1.5e-3 + .25").unwrap() - 0.2515).abs() < 1e-12);
    }
    #[test]
    fn math_functions_and_domains() {
        assert!((evaluate("sin(pi/2) + sqrt(9) + log10(100)").unwrap() - 6.0).abs() < 1e-12);
        assert_eq!(evaluate("max(3, min(4, 9))").unwrap(), 4.0);
        for invalid in [
            "1/0",
            "sqrt(-1)",
            "ln(0)",
            "exp(10000)",
            "sqrt(1,2)",
            "min(1)",
            "1e-",
            ".",
        ] {
            assert!(evaluate(invalid).is_err(), "accepted {invalid}");
        }
    }
    #[test]
    fn no_code_execution_or_unbounded_recursion() {
        for invalid in [
            "1; rm -rf x",
            "__import__('os')",
            "a = 3",
            "2(3)",
            "open(file)",
        ] {
            assert!(evaluate(invalid).is_err(), "accepted {invalid}");
        }
        assert!(evaluate(&format!("{}1{}", "(".repeat(70), ")".repeat(70))).is_err());
    }
}

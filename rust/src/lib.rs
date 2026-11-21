//! Rust core for gpt-tokenizer: regex pre-split + rank-based byte-level BPE.
use fancy_regex::Regex as FancyRegex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// Encode one chunk: repeatedly merge the adjacent pair whose bytes have the lowest rank.
fn bpe(piece: &[u8], ranks: &FxHashMap<Vec<u8>, u32>, out: &mut Vec<u32>) -> PyResult<()> {
    // Fast path: the whole chunk is already a token (same as tiktoken).
    if let Some(&id) = ranks.get(piece) {
        out.push(id);
        return Ok(());
    }
    // Part i is piece[bounds[i]..bounds[i + 1]]; merging removes a boundary.
    let mut bounds: Vec<usize> = (0..=piece.len()).collect();
    while bounds.len() >= 3 {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..bounds.len() - 2 {
            if let Some(&rank) = ranks.get(&piece[bounds[i]..bounds[i + 2]]) {
                if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                    best = Some((rank, i));
                }
            }
        }
        match best {
            Some((_, i)) => {
                bounds.remove(i + 1);
            }
            None => break,
        }
    }
    for w in bounds.windows(2) {
        let byte_seq = &piece[w[0]..w[1]];
        match ranks.get(byte_seq) {
            Some(&id) => out.push(id),
            None => return Err(PyValueError::new_err(format!("byte sequence not in vocab: {byte_seq:?}"))),
        }
    }
    Ok(())
}

/// Lookahead tails used by GPT-style patterns. The lookahead is the only part that needs a
/// backtracking engine, so we swap the tail for `\s+` and emulate it in `FastSplit::pieces`.
const LOOKAHEAD_TAILS: [&str; 2] = [r"|\s+(?!\S)|\s+", r"|\s+(?!\S)|\s"];
/// The cl100k/o200k newline alternative, which claims every whitespace run that holds a newline.
const NEWLINE_ALT: &str = r"\s*[\r\n]";
/// tiktoken's gpt2 and cl100k_base patterns use possessive quantifiers, which the `regex` crate
/// cannot parse. None is followed by anything that could backtrack into it, so the greedy
/// versions on the right give identical matches. Pairs are (tiktoken pattern, greedy rewrite).
const POSSESSIVE_REWRITES: [(&str, &str); 2] = [
    (
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s",
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s",
    ),
    (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s",
    ),
];

/// Split pattern compiled for the linear-time `regex` crate, with `\s+(?!\S)|\s+` -> `\s+`.
struct FastSplit {
    regex: Regex,
    has_newline_alt: bool,
}

impl FastSplit {
    fn new(pattern: &str) -> Option<Self> {
        let pattern = POSSESSIVE_REWRITES
            .iter()
            .find(|(original, _)| *original == pattern)
            .map_or(pattern, |(_, greedy)| *greedy);
        let head = LOOKAHEAD_TAILS.iter().find_map(|tail| pattern.strip_suffix(tail))?;
        let regex = Regex::new(&format!(r"{head}|\s+")).ok()?;
        Some(FastSplit { regex, has_newline_alt: head.contains(NEWLINE_ALT) })
    }

    /// Call `f` on each piece, matching what the original pattern with the lookahead yields.
    fn pieces(&self, text: &str, mut f: impl FnMut(&str) -> PyResult<()>) -> PyResult<()> {
        let mut pos = 0;
        while let Some(m) = self.regex.find_at(text, pos) {
            let mut end = m.end();
            let piece = m.as_str();
            if piece.is_empty() {
                // Custom patterns may match empty text: emit nothing and step past one char.
                match text[m.start()..].chars().next() {
                    Some(c) => pos = m.start() + c.len_utf8(),
                    None => break,
                }
                continue;
            }
            // `\s+(?!\S)` backs off one char when a whitespace run is followed by text, leaving
            // that char to attach to the next word. Single-char runs fall through to plain `\s+`.
            // Runs holding a newline came from the newline alternative, which has no lookahead.
            if end < text.len()
                && piece.chars().all(char::is_whitespace)
                && !(self.has_newline_alt && piece.contains(['\r', '\n']))
            {
                if let Some((last, _)) = piece.char_indices().last().filter(|&(i, _)| i > 0) {
                    end = m.start() + last;
                }
            }
            f(&text[m.start()..end])?;
            pos = end;
        }
        Ok(())
    }
}

enum Splitter {
    Fast(FastSplit),
    Fancy(FancyRegex),
}

#[pyclass]
struct CoreBPE {
    ranks: FxHashMap<Vec<u8>, u32>,
    splitter: Splitter,
}

#[pymethods]
impl CoreBPE {
    #[new]
    fn new(ranks: HashMap<Vec<u8>, u32>, pattern: &str) -> PyResult<Self> {
        let splitter = match FastSplit::new(pattern) {
            Some(fast) => Splitter::Fast(fast),
            None => Splitter::Fancy(
                FancyRegex::new(pattern).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        };
        Ok(CoreBPE { ranks: ranks.into_iter().collect(), splitter })
    }

    /// True when the pattern runs on the linear-time `regex` crate instead of fancy-regex.
    #[getter]
    fn uses_fast_regex(&self) -> bool {
        matches!(self.splitter, Splitter::Fast(_))
    }

    /// Encode text with no special-token handling. Releases the GIL while working.
    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.allow_threads(|| {
            let mut out = Vec::with_capacity(text.len() / 3);
            match &self.splitter {
                Splitter::Fast(fast) => fast.pieces(text, |p| bpe(p.as_bytes(), &self.ranks, &mut out))?,
                Splitter::Fancy(regex) => {
                    for m in regex.find_iter(text) {
                        let m = m.map_err(|e| PyValueError::new_err(e.to_string()))?;
                        bpe(m.as_str().as_bytes(), &self.ranks, &mut out)?;
                    }
                }
            }
            Ok(out)
        })
    }
}

#[pymodule]
fn gpt_tokenizer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreBPE>()?;
    Ok(())
}

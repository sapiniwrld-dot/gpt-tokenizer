//! Rust core for gpt-tokenizer: regex pre-split + rank-based byte-level BPE.
use fancy_regex::Regex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// Encode one chunk: repeatedly merge the adjacent pair whose bytes have the lowest rank.
fn bpe(piece: &[u8], ranks: &FxHashMap<Vec<u8>, u32>, out: &mut Vec<u32>) {
    // Fast path: the whole chunk is already a token (same as tiktoken).
    if let Some(&id) = ranks.get(piece) {
        out.push(id);
        return;
    }
    // Part i is piece[bounds[i]..bounds[i + 1]]; merging removes a boundary.
    let mut bounds: Vec<usize> = (0..=piece.len()).collect();
    while bounds.len() >= 3 {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..bounds.len() - 2 {
            if let Some(&rank) = ranks.get(&piece[bounds[i]..bounds[i + 2]]) {
                if best.map_or(true, |(best_rank, _)| rank < best_rank) {
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
        out.push(*ranks.get(&piece[w[0]..w[1]]).expect("every single byte must be in the vocab"));
    }
}

#[pyclass]
struct CoreBPE {
    ranks: FxHashMap<Vec<u8>, u32>,
    regex: Regex,
}

#[pymethods]
impl CoreBPE {
    #[new]
    fn new(ranks: HashMap<Vec<u8>, u32>, pattern: &str) -> PyResult<Self> {
        let regex = Regex::new(pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CoreBPE { ranks: ranks.into_iter().collect(), regex })
    }

    /// Encode text with no special-token handling. Releases the GIL while working.
    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.allow_threads(|| {
            let mut out = Vec::with_capacity(text.len() / 3);
            for m in self.regex.find_iter(text) {
                let m = m.map_err(|e| PyValueError::new_err(e.to_string()))?;
                bpe(m.as_str().as_bytes(), &self.ranks, &mut out);
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

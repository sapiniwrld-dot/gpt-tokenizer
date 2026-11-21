# gpt-tokenizer

A byte-level BPE tokenizer built from scratch in Python, with a Rust core for speed.
It reproduces the GPT-2, GPT-4 (`cl100k_base`) and GPT-4o (`o200k_base`) tokenizers
**token for token**, verified against OpenAI's [`tiktoken`](https://github.com/openai/tiktoken).

## Results

Measured on Tiny Shakespeare (1.1 MB), MacBook Air (Apple silicon), single thread, best of 3.

| Encoding      | tiktoken | This repo (Rust) | This repo (Python) | Exact match |
|---------------|---------:|-----------------:|-------------------:|:-----------:|
| `gpt2`        | 18.5 MB/s | 33.5 MB/s (0.6x) | 1.8 MB/s (10.0x) | yes |
| `cl100k_base` | 18.8 MB/s | 38.6 MB/s (0.5x) | 2.0 MB/s (9.4x)  | yes |
| `o200k_base`  | 29.8 MB/s | 36.5 MB/s (0.8x) | 2.0 MB/s (15.2x) | yes |

Times in parentheses are relative to tiktoken (lower is better). The Rust path beats tiktoken
because it splits text with the linear-time `regex` crate instead of a backtracking engine (see below).

Training your own vocabulary on 90% of the corpus, compression on the held-out 10%:

| Vocab size | Bytes per token | Train time |
|-----------:|----------------:|-----------:|
| 512        | 1.88 | 3.4 s |
| 1,024      | 2.26 | 9.5 s |
| 2,048      | 2.56 | 19.9 s |
| GPT-2 (50,257) | 3.09 | n/a |

## Features

- **Training:** byte-level BPE with GPT-style regex pre-splitting, so merges never cross word boundaries.
- **Exact compatibility:** loads real tiktoken vocabularies and matches their output, including special tokens.
- **Special tokens:** tiktoken-compatible semantics. `encode()` raises if the text contains a special
  token, `allowed_special="all"` (or a set) enables them, and `encode_ordinary()` treats them as text.
- **Rust core:** regex split and BPE merging in Rust via PyO3. Releases the GIL while encoding.
- **Lookahead-free split:** GPT patterns need a backtracking regex engine only for the
  `\s+(?!\S)` lookahead. The Rust core swaps it for `\s+` and emulates the lookahead in code,
  which lets it use the linear-time `regex` crate. Output stays token-for-token identical.
  Other patterns fall back to `fancy-regex`.
- **Save and load:** tokenizers serialize to JSON.
- **149 tests**, including parity tests against tiktoken on all three encodings.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain ([rustup](https://rustup.rs)).

```bash
uv sync                      # builds the Rust extension
uv run pytest -q

mkdir -p data
curl -L -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
uv run python scripts/benchmark.py data/tinyshakespeare.txt
uv run python scripts/train.py data/tinyshakespeare.txt
```

### Python API

```python
from gpt_tokenizer import FastTokenizer

tok = FastTokenizer.from_tiktoken("cl100k_base")
ids = tok.encode("hello world<|endoftext|>", allowed_special="all")
assert tok.decode(ids) == "hello world<|endoftext|>"

custom = FastTokenizer()
custom.train(open("data/tinyshakespeare.txt").read(), vocab_size=1024)
custom.save("models/shakespeare-1024.json")
```

### CLI

```bash
uv run gpt-tokenizer encode "hello world" --encoding cl100k_base
uv run gpt-tokenizer decode 15339 1917 --encoding cl100k_base
uv run gpt-tokenizer count "hello world" --encoding cl100k_base
```

## How it works

1. Pre-split: a regex cuts text into chunks (words with their leading space, numbers,
   punctuation, whitespace runs).
2. Train: start from the 256 byte values. Repeatedly find the most frequent adjacent pair
   across all chunks and merge it into a new token, until the vocabulary reaches its target size.
3. Encode: for each chunk, repeatedly merge the adjacent pair whose combined bytes have the
   lowest rank (the earliest-learned merge), until no pair is in the vocabulary.
4. Decode: concatenate each token's bytes and decode as UTF-8.

## Design notes

- Rank-based encoding. Storing the vocabulary as bytes -> rank (the tiktoken format)
  handles GPT-2's shuffled single-byte IDs with no special cases.
- Deduplicated training. Identical chunks always merge identically, so each unique chunk is
  counted once with a weight. Tiny Shakespeare has about 300k chunks but only about 25k unique ones.
- Whole-chunk fast path. If a chunk is already a single token, the Rust core returns it
  immediately, as tiktoken does.
- Lazy Rust core. FastTokenizer rebuilds the Rust core whenever its ranks are replaced, so
  training, loading and from_tiktoken stay correct.

## Project layout

src/gpt_tokenizer/bpe.py              BasicTokenizer: raw-byte BPE, no pre-split
src/gpt_tokenizer/regex_tokenizer.py  RegexTokenizer: pre-split, special tokens, save/load, tiktoken loading
src/gpt_tokenizer/fast.py             FastTokenizer: RegexTokenizer with the Rust encode path
rust/src/lib.rs                       Rust core (PyO3)
scripts/benchmark.py                  Parity and speed against tiktoken
scripts/train.py                      Train vocabularies and measure held-out compression
tests/                                pytest suite

Future work

- Close the o200k_base gap: its split pattern is slower in fancy-regex than tiktoken's tuned path.
- Batch encoding across threads (the Rust core already releases the GIL).
- Faster training with incremental pair counts and a priority queue, instead of a full recount per merge.
- Move training into Rust.

Acknowledgements

Inspired by Andrej Karpathy's minbpe and OpenAI's
tiktoken.

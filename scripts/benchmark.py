"""Check exact parity and measure speed against tiktoken on a real corpus."""
import pathlib
import sys
import time

import tiktoken

from gpt_tokenizer.fast import FastTokenizer
from gpt_tokenizer.regex_tokenizer import RegexTokenizer

ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]


def timed(fn, text, repeats=3):
    """Best-of-N wall time, so one-off noise does not skew the result."""
    best, ids = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        ids = fn(text)
        best = min(best, time.perf_counter() - t0)
    return ids, best


def main():
    path = pathlib.Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    print(f"corpus: {path.name}, {len(text):,} chars, {n_bytes:,} bytes\n")
    print(f"{'encoding':12} {'impl':10} {'time':>8} {'MB/s':>7} {'vs tiktoken':>12}  match")

    for name in ENCODINGS:
        ref = tiktoken.get_encoding(name)
        ref_ids, t_ref = timed(ref.encode_ordinary, text)
        impls = [
            ("python", RegexTokenizer.from_tiktoken(name)),
            ("rust", FastTokenizer.from_tiktoken(name)),
        ]
        print(f"{name:12} {'tiktoken':10} {t_ref:7.3f}s {n_bytes / t_ref / 1e6:7.1f} {'1.0x':>12}")
        for label, tok in impls:
            tok.encode_ordinary(text[:1000])  # warm up (builds the Rust core once)
            ids, t = timed(tok.encode_ordinary, text, repeats=1 if label == "python" else 3)
            match = ids == ref_ids and tok.decode(ids) == text
            print(f"{'':12} {label:10} {t:7.3f}s {n_bytes / t / 1e6:7.1f} {t / t_ref:11.1f}x  {match}")
        print()


if __name__ == "__main__":
    main()

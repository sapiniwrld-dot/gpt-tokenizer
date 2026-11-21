"""Check exact parity and measure speed against tiktoken on a real corpus."""
import pathlib
import sys
import time

import tiktoken

from gpt_tokenizer.regex_tokenizer import RegexTokenizer

ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main():
    path = pathlib.Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    print(f"corpus: {path.name}, {len(text):,} chars, {n_bytes:,} bytes\n")

    for name in ENCODINGS:
        ref = tiktoken.get_encoding(name)
        ours = RegexTokenizer.from_tiktoken(name)

        t0 = time.perf_counter()
        ref_ids = ref.encode_ordinary(text)
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        our_ids = ours.encode(text)
        t_ours = time.perf_counter() - t0

        match = our_ids == ref_ids
        round_trip = ours.decode(our_ids) == text
        print(f"{name:12} tokens={len(ref_ids):>9,}  bytes/token={n_bytes / len(ref_ids):.2f}  "
              f"match={match}  round_trip={round_trip}")
        print(f"{'':12} ours={t_ours:.2f}s  tiktoken={t_ref:.3f}s  "
              f"slowdown={t_ours / t_ref:.0f}x  ours={n_bytes / t_ours / 1e6:.2f} MB/s")
        if not match:
            i = first_mismatch(our_ids, ref_ids)
            print(f"{'':12} first mismatch at token {i}: ours={our_ids[i:i+5]} ref={ref_ids[i:i+5]}")
        print()


if __name__ == "__main__":
    main()

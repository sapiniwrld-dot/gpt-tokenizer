"""Train BPE tokenizers on a corpus and compare held-out compression to GPT-2."""
import pathlib
import sys
import time

import tiktoken

from gpt_tokenizer.regex_tokenizer import RegexTokenizer

VOCAB_SIZES = [512, 1024, 2048]


def main():
    text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    train_text, held_out = text[:split], text[split:]
    held_bytes = len(held_out.encode("utf-8"))
    out_dir = pathlib.Path("models")
    out_dir.mkdir(exist_ok=True)

    gpt2 = tiktoken.get_encoding("gpt2")
    gpt2_ratio = held_bytes / len(gpt2.encode_ordinary(held_out))
    print(f"train={len(train_text):,} chars  held_out={len(held_out):,} chars")
    print(f"{'gpt2 (50257)':>14}  held-out bytes/token={gpt2_ratio:.2f}\n")

    for vocab_size in VOCAB_SIZES:
        tok = RegexTokenizer()
        t0 = time.perf_counter()
        tok.train(train_text, vocab_size)
        t_train = time.perf_counter() - t0
        ids = tok.encode(held_out)
        assert tok.decode(ids) == held_out
        tok.save(out_dir / f"shakespeare-{vocab_size}.json")
        top = [tok.vocab[i] for i in range(vocab_size - 5, vocab_size)]
        print(f"{vocab_size:>14}  held-out bytes/token={held_bytes / len(ids):.2f}  "
              f"train={t_train:.1f}s  last merges={top}")


if __name__ == "__main__":
    main()

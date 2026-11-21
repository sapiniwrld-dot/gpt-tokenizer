"""Regex pre-split BPE tokenizer, compatible with GPT-2's tiktoken encoding."""
import json
import pathlib
from collections import Counter

import regex as re

from gpt_tokenizer.bpe import merge

# GPT-2's split pattern: contractions, letters, numbers, punctuation, whitespace.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RegexTokenizer:
    def __init__(self, pattern=GPT2_SPLIT_PATTERN):
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self._build_ranks()

    def _build_ranks(self):
        """Invert vocab to bytes -> id, keeping the earliest id on duplicates."""
        self.ranks = {}
        for i in sorted(self.vocab):
            self.ranks.setdefault(self.vocab[i], i)

    def train(self, text, vocab_size, verbose=False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Identical chunks always merge identically, so count each unique one once.
        counts = Counter(self.compiled.findall(text))
        chunks = [list(c.encode("utf-8")) for c in counts]
        weights = list(counts.values())
        for new_id in range(256, vocab_size):
            stats = {}
            for ids, w in zip(chunks, weights):
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + w
            if not stats:
                break
            pair = max(stats, key=stats.get)
            chunks = [merge(ids, pair, new_id) for ids in chunks]
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {new_id - 255}: {pair} -> {new_id} "
                      f"{self.vocab[new_id]!r} ({stats[pair]} times)")
        self._build_ranks()

    @classmethod
    def from_tiktoken(cls, name="gpt2"):
        """Load a real tiktoken vocabulary (ranks + split pattern)."""
        import tiktoken

        enc = tiktoken.get_encoding(name)
        tok = cls(pattern=enc._pat_str)
        tok.ranks = dict(enc._mergeable_ranks)
        tok.vocab = {i: b for b, i in tok.ranks.items()}
        return tok

    def save(self, path):
        data = {
            "pattern": self.pattern,
            "vocab": {str(i): b.hex() for i, b in self.vocab.items()},
        }
        pathlib.Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(pathlib.Path(path).read_text())
        tok = cls(pattern=data["pattern"])
        tok.vocab = {int(i): bytes.fromhex(h) for i, h in data["vocab"].items()}
        tok._build_ranks()
        return tok

    def _encode_chunk(self, data):
        parts = [bytes([b]) for b in data]
        while len(parts) >= 2:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                rank = self.ranks.get(parts[i] + parts[i + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i is None:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return [self.ranks[p] for p in parts]

    def encode(self, text):
        ids = []
        for chunk in self.compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

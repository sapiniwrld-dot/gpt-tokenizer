"""Regex pre-split BPE tokenizer, compatible with tiktoken encodings."""
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
        self.special_tokens = {}  # str -> id
        self.inverse_special = {}  # id -> bytes

    def _build_ranks(self):
        """Invert vocab to bytes -> id, keeping the earliest id on duplicates."""
        self.ranks = {}
        for i in sorted(self.vocab):
            self.ranks.setdefault(self.vocab[i], i)

    def register_special_tokens(self, special_tokens):
        """Map special token strings (e.g. "<|endoftext|>") to ids outside the BPE vocab."""
        for s, i in special_tokens.items():
            if i in self.vocab:
                raise ValueError(f"special token {s!r} id {i} collides with the BPE vocab")
        self.special_tokens = dict(special_tokens)
        self.inverse_special = {i: s.encode("utf-8") for s, i in self.special_tokens.items()}

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
        """Load a real tiktoken vocabulary (ranks, split pattern, special tokens)."""
        import tiktoken

        enc = tiktoken.get_encoding(name)
        tok = cls(pattern=enc._pat_str)
        tok.ranks = dict(enc._mergeable_ranks)
        tok.vocab = {i: b for b, i in tok.ranks.items()}
        tok.register_special_tokens(enc._special_tokens)
        return tok

    def save(self, path):
        data = {
            "pattern": self.pattern,
            "vocab": {str(i): b.hex() for i, b in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }
        pathlib.Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(pathlib.Path(path).read_text())
        tok = cls(pattern=data["pattern"])
        tok.vocab = {int(i): bytes.fromhex(h) for i, h in data["vocab"].items()}
        tok._build_ranks()
        tok.register_special_tokens(data.get("special_tokens", {}))
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

    def encode_ordinary(self, text):
        """Encode text, treating special token strings as plain text."""
        ids = []
        for chunk in self.compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """allowed_special: "all", "none", "none_raise" (default), or a set of strings."""
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special in ("none", "none_raise"):
            special = {}
            if allowed_special == "none_raise":
                for s in self.special_tokens:
                    if s in text:
                        raise ValueError(
                            f"text contains special token {s!r}; "
                            "pass allowed_special to encode it")
        elif isinstance(allowed_special, (set, frozenset)):
            special = {s: i for s, i in self.special_tokens.items() if s in allowed_special}
        else:
            raise ValueError(f"invalid allowed_special: {allowed_special!r}")

        if not special:
            return self.encode_ordinary(text)
        # Longest first, so a special token that contains another one wins.
        alternatives = sorted(special, key=len, reverse=True)
        split_pattern = "(" + "|".join(re.escape(s) for s in alternatives) + ")"
        ids = []
        for part in re.split(split_pattern, text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special:
                parts.append(self.inverse_special[i])
            else:
                raise ValueError(f"invalid token id: {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

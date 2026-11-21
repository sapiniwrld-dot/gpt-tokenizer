"""Byte-level BPE tokenizer core: train, encode, decode."""


def get_stats(ids, counts=None):
    """Count how often each adjacent pair of token IDs appears."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BasicTokenizer:
    """BPE over raw UTF-8 bytes, with no regex pre-splitting."""

    def __init__(self):
        self.merges = {}  # (id, id) -> new id, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        ids = list(text.encode("utf-8"))
        for new_id in range(256, vocab_size):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {new_id - 255}: {pair} -> {new_id} "
                      f"{self.vocab[new_id]!r} ({stats[pair]} times)")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Apply the merge learned earliest; later merges build on earlier ones.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

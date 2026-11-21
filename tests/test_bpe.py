from gpt_tokenizer.bpe import BasicTokenizer, get_stats, merge


def test_get_stats_counts_adjacent_pairs():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge_replaces_pair():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_handles_overlap():
    assert merge([5, 5, 5], (5, 5), 99) == [99, 5]


def test_untrained_tokenizer_is_raw_bytes():
    tok = BasicTokenizer()
    text = "hello 🌍 world"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text


def test_known_merge_sequence():
    # Classic BPE example: aa -> 256, (256, a) -> 257, (257, b) -> 258
    tok = BasicTokenizer()
    tok.train("aaabdaaabac", vocab_size=259)
    assert tok.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert tok.vocab[258] == b"aaab"


def test_training_compresses_and_round_trips():
    text = "the cat sat on the mat. " * 50
    tok = BasicTokenizer()
    tok.train(text, vocab_size=300)
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))
    assert tok.decode(ids) == text

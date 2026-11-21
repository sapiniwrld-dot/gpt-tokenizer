import pathlib

import pytest
import tiktoken

from gpt_tokenizer.regex_tokenizer import RegexTokenizer

SAMPLES = [
    "hello world",
    "Hello, World! It's 2026 and we're tokenizing.",
    "   leading and trailing spaces   ",
    "tabs\tand\nnewlines\n\n\nmany",
    "numbers 1234567890 3.14159 -42",
    "unicode: café naïve 日本語 🌍🚗",
    "don't won't I'll you've they're she'd",
    "def f(x):\n    return x ** 2  # comment\n",
    "",
]


@pytest.fixture(scope="module")
def gpt2():
    return RegexTokenizer.from_tiktoken("gpt2"), tiktoken.get_encoding("gpt2")


@pytest.mark.parametrize("text", SAMPLES)
def test_matches_tiktoken_gpt2(gpt2, text):
    ours, ref = gpt2
    assert ours.encode(text) == ref.encode_ordinary(text)
    assert ours.decode(ours.encode(text)) == text


def test_matches_tiktoken_on_source_file(gpt2):
    ours, ref = gpt2
    text = pathlib.Path(__file__).read_text()
    assert ours.encode(text) == ref.encode_ordinary(text)


def test_trained_regex_tokenizer_round_trips():
    text = "the cat sat on the mat. " * 50 + "hello world 123"
    tok = RegexTokenizer()
    tok.train(text, vocab_size=300)
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))
    assert tok.decode(ids) == text


def test_merges_never_cross_chunks():
    tok = RegexTokenizer()
    tok.train("ab ab ab ab", vocab_size=270)
    for token in tok.vocab.values():
        assert b" " not in token[1:]  # a space may only lead a token


def test_save_load_round_trip(tmp_path):
    text = "the cat sat on the mat. " * 50 + "naïve 🌍"
    tok = RegexTokenizer()
    tok.train(text, vocab_size=300)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = RegexTokenizer.load(path)
    assert loaded.vocab == tok.vocab
    assert loaded.pattern == tok.pattern
    assert loaded.encode(text) == tok.encode(text)


def test_save_load_preserves_gpt2_parity(gpt2, tmp_path):
    ours, ref = gpt2
    path = tmp_path / "gpt2.json"
    ours.save(path)
    loaded = RegexTokenizer.load(path)
    text = "Hello, World! It's 2026 and we're tokenizing. 日本語 🌍"
    assert loaded.encode(text) == ref.encode_ordinary(text)

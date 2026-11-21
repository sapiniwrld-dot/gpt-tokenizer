import pathlib

import pytest
import tiktoken

from gpt_tokenizer.fast import FastTokenizer
from gpt_tokenizer.regex_tokenizer import RegexTokenizer
from test_regex_tokenizer import SAMPLES

ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]


@pytest.fixture(scope="module", params=ENCODINGS)
def pair(request):
    return FastTokenizer.from_tiktoken(request.param), tiktoken.get_encoding(request.param)


@pytest.mark.parametrize("text", SAMPLES)
def test_fast_matches_tiktoken(pair, text):
    fast, ref = pair
    assert fast.encode(text) == ref.encode_ordinary(text)
    assert fast.decode(fast.encode(text)) == text


def test_fast_matches_tiktoken_on_source_files(pair):
    fast, ref = pair
    root = pathlib.Path(__file__).parent.parent
    text = "\n".join(p.read_text() for p in sorted(root.rglob("*.py")) if ".venv" not in p.parts)
    assert fast.encode_ordinary(text) == ref.encode_ordinary(text)


def test_fast_special_tokens_match_tiktoken(pair):
    fast, ref = pair
    text = "hello<|endoftext|>world <|endoftext|>!"
    assert fast.encode(text, allowed_special="all") == ref.encode(text, allowed_special="all")


def test_fast_matches_python_on_trained_vocab():
    text = pathlib.Path(__file__).read_text() * 5
    slow = RegexTokenizer()
    slow.train(text, vocab_size=400)
    fast = FastTokenizer()
    fast.train(text, vocab_size=400)
    assert fast.vocab == slow.vocab
    assert fast.encode(text) == slow.encode(text)


def test_fast_core_rebuilds_after_retrain():
    fast = FastTokenizer()
    before = fast.encode("aaaa")
    fast.train("aaaa " * 100, vocab_size=260)
    after = fast.encode("aaaa")
    assert before == [97, 97, 97, 97]
    assert len(after) < len(before)

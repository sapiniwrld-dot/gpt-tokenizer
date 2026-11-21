import pytest
import tiktoken

from gpt_tokenizer.regex_tokenizer import RegexTokenizer

TEXT = "hello<|endoftext|>world <|endoftext|>\n<|endoftext|><|endoftext|>!"


@pytest.fixture(scope="module", params=["gpt2", "cl100k_base", "o200k_base"])
def pair(request):
    name = request.param
    return RegexTokenizer.from_tiktoken(name), tiktoken.get_encoding(name)


def test_allowed_all_matches_tiktoken(pair):
    ours, ref = pair
    ids = ours.encode(TEXT, allowed_special="all")
    assert ids == ref.encode(TEXT, allowed_special="all")
    assert ours.decode(ids) == TEXT


def test_default_raises_like_tiktoken(pair):
    ours, ref = pair
    with pytest.raises(ValueError):
        ref.encode(TEXT)
    with pytest.raises(ValueError):
        ours.encode(TEXT)


def test_none_treats_specials_as_text(pair):
    ours, ref = pair
    assert ours.encode(TEXT, allowed_special="none") == ref.encode_ordinary(TEXT)
    assert ours.encode_ordinary(TEXT) == ref.encode_ordinary(TEXT)


def test_allowed_subset_matches_tiktoken():
    ours = RegexTokenizer.from_tiktoken("cl100k_base")
    ref = tiktoken.get_encoding("cl100k_base")
    text = "<|fim_prefix|>def f():<|fim_suffix|> return 1<|fim_middle|>"
    allowed = {"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"}
    assert ours.encode(text, allowed_special=allowed) == ref.encode(text, allowed_special=allowed)


def test_custom_special_tokens_survive_save_load(tmp_path):
    tok = RegexTokenizer()
    tok.train("the cat sat on the mat. " * 50, vocab_size=300)
    tok.register_special_tokens({"<|endoftext|>": 300})
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = RegexTokenizer.load(path)
    text = "the cat<|endoftext|>the mat"
    ids = loaded.encode(text, allowed_special="all")
    assert 300 in ids
    assert ids == tok.encode(text, allowed_special="all")
    assert loaded.decode(ids) == text


def test_special_id_collision_rejected():
    tok = RegexTokenizer()
    with pytest.raises(ValueError):
        tok.register_special_tokens({"<|endoftext|>": 65})

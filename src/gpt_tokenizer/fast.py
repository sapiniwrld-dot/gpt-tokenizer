"""RegexTokenizer with the encode hot loop in Rust."""
from gpt_tokenizer_rs import CoreBPE

from gpt_tokenizer.regex_tokenizer import RegexTokenizer


class FastTokenizer(RegexTokenizer):
    """Same API and output as RegexTokenizer; encode_ordinary runs in Rust."""

    _core = None
    _core_ranks = None

    def _get_core(self):
        # Rebuild when ranks are replaced (train, load, from_tiktoken).
        if self._core is None or self._core_ranks is not self.ranks:
            self._core = CoreBPE(self.ranks, self.pattern)
            self._core_ranks = self.ranks
        return self._core

    def encode_ordinary(self, text):
        return self._get_core().encode_ordinary(text)

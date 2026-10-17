"""Deterministic character vocabulary with lossless checkpoint serialization."""
from collections.abc import Iterable


class CharTokenizer:
    def __init__(self, text: str):
        self._set_chars(sorted(set(text)))

    def _set_chars(self, chars):
        if not chars or any(not isinstance(c, str) or len(c) != 1 for c in chars):
            raise ValueError("vocabulary must contain single characters")
        if len(set(chars)) != len(chars):
            raise ValueError("vocabulary contains duplicate characters")
        self.chars = list(chars)
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = dict(enumerate(chars))

    @classmethod
    def from_chars(cls, chars):
        """Restore the stored order; sorting would corrupt existing token IDs."""
        tokenizer = cls.__new__(cls)
        tokenizer._set_chars(chars)
        return tokenizer

    def encode(self, text: str) -> list[int]:
        try:
            return [self.char_to_idx[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is outside the training vocabulary") from exc

    def decode(self, token_ids: Iterable[int]) -> str:
        try:
            return "".join(self.idx_to_char[int(i)] for i in token_ids)
        except KeyError as exc:
            raise ValueError(f"invalid token ID: {exc.args[0]}") from exc

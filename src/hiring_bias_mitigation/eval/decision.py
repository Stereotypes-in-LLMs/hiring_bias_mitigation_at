"""The decision prefix and the decision tokens, shared by the margin readers.

The interpretability pipeline (`interp/`) imports these from here. The definitions live in
`eval/margins.py`, which the op-margin passes and `diagnose_adapter.py` already use, so there
is one prefix and one token rule; this module only re-exports them and lets a caller name the
decision words by language.
"""

from __future__ import annotations

from .margins import PREFIX, WORDS
from .margins import decision_token_ids as _decision_token_ids

__all__ = ["PREFIX", "WORDS", "decision_token_ids"]


def decision_token_ids(tokenizer, words: str | tuple[str, str] = "en") -> tuple[int, int]:
    """The first token of each decision word after the prefix; `words` is a language code
    (a key of WORDS) or the two words themselves, hire first."""
    return _decision_token_ids(tokenizer, WORDS[words] if isinstance(words, str) else words)

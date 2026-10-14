"""The model's own decision margin: logit(hire) - logit(reject) at the decision token.

Every response opens with `{"decision": "`, so the token after that prefix is the verdict.
Forcing the prefix and reading the two logits there gives the decision as a continuous number
from one forward pass, with no generation: positive means the greedy answer is hire, and the
size says how sure the model is. Both logits come from the same softmax, so the difference is
also log p(hire) - log p(reject), and `sigmoid(margin)` is the model's probability of hiring
when it has to pick one of the two words.

The pass runs through HuggingFace (+ PEFT for an adapter), the framework the adapters are
trained in. vLLM's LoRA path does not reproduce Qwen3.5 adapters (see `mitigation/merge.py`),
and a score meant to describe where a training run starts has to come from the model the
trainer will actually hold.

`scripts/diagnose_adapter.py` and `scripts/operating_point_control.py` read the same quantity
with their own copies of this logic.
"""

from __future__ import annotations

import time

import numpy as np

from ..utils.logging import get_logger
from .backends import _chat_wrap

log = get_logger(__name__)

#: Every response opens with this, so the next token is the decision.
PREFIX = '{"decision": "'

#: The decision words each language's prompt asks for, hire first.
WORDS = {"en": ("hire", "reject"), "uk": ("найняти", "відхилити")}


def decision_token_ids(tokenizer, words: tuple[str, str]) -> tuple[int, int]:
    """The first token of each decision word, as it follows the forced prefix."""
    base = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for word in words:
        full = tokenizer(PREFIX + word, add_special_tokens=False)["input_ids"]
        if full[: len(base)] != base:
            raise ValueError(f"the prefix {PREFIX!r} tokenises differently before {word!r}")
        ids.append(full[len(base)])
    if ids[0] == ids[1]:
        raise ValueError(f"{words[0]!r} and {words[1]!r} share a first token")
    return ids[0], ids[1]


def decision_margins(
    model,
    tokenizer,
    prompts: list[str],
    lang: str = "en",
    batch_size: int = 4,
    chat_template_kwargs: dict | None = None,
    words: tuple[str, str] | None = None,
) -> np.ndarray:
    """logit(hire) - logit(reject) after the forced prefix, one value per prompt.

    Prompts are wrapped in the chat template exactly as the audit and the trainers wrap them.
    They run longest first, so a batch that does not fit in memory fails in the first minute
    rather than the last, and come back in the order they were given. Padding is on the left,
    so the last position of every row is the decision position.
    """
    import torch

    hire_id, reject_id = decision_token_ids(tokenizer, words or WORDS[lang])
    texts = [_chat_wrap(tokenizer, p, chat_template_kwargs) + PREFIX for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = np.empty(len(texts), dtype=np.float64)

    padding_side = getattr(tokenizer, "padding_side", "right")
    was_training = bool(getattr(model, "training", False))
    tokenizer.padding_side = "left"
    model.eval()
    started, n_batches = time.monotonic(), max(1, -(-len(texts) // batch_size))
    log_every = max(1, n_batches // 20)
    try:
        for index, start in enumerate(range(0, len(order), batch_size)):
            rows = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in rows], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logits = _last_position_logits(model, enc)
            out[rows] = (logits[:, hire_id] - logits[:, reject_id]).float().cpu().numpy()
            if (index + 1) % log_every == 0 or index + 1 == n_batches:
                elapsed = time.monotonic() - started
                log.info("  margins %d / %d prompts (%.0f s)", min(start + batch_size, len(texts)),
                         len(texts), elapsed)
    finally:
        tokenizer.padding_side = padding_side
        if was_training:
            model.train()
    return out


def _last_position_logits(model, enc):
    """Logits at the final position only.

    Asking for one position keeps the output at [batch, vocab] instead of a full
    [batch, sequence, ~250k] tensor -- gigabytes per batch on long CVs. Older transformers
    without `logits_to_keep` fall back to slicing the full output.
    """
    try:
        return model(**enc, logits_to_keep=1).logits[:, -1, :]
    except TypeError:
        return model(**enc).logits[:, -1, :]

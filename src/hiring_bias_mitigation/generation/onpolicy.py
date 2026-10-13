"""On-policy negatives: the student's own attribute-tipped mistakes as `rejected`.

Why. The teacher-written v2 negatives (batch-01) are covert -- none names the attribute -- but
a text classifier still tells them from `chosen` at AUC 0.88, because a rationale argued for
the wrong verdict is weaker: it hedges a weak candidate into a hire or nitpicks a strong one.
The audited 9B's OWN wrong-verdict rationales in attribute-tipped sets are not separable from
its right ones (AUC 0.47 over 156 sets): its real covert bias is a confident, grounded
rationale with the decision moved. A negative DPO can learn from has to look like that, and
the one source of exactly that is the student.

Construction, per batch row (one v1 prompt; `chosen` stays v1's teacher text):
  1. k samples from the SFT student on the attribute-bearing audit prompt, and k on the same
     pair's attribute-free prompt;
  2. the row is *attribute-tipped* when the student is mostly right without the attribute
     (free_right_rate > min_free_right), so a wrong answer with it is the attribute's doing;
  3. its wrong samples go through the same hard filters as the teacher's negatives -- flipped,
     no attribute or group named, length within +-N words of chosen, right language, no
     meta-commentary -- and the first survivor in sample order becomes the row's `rejected`;
  4. the teacher then judges it exactly as it judged the teacher-written negatives.

Rows that are wrong but not tipped keep their own strategy name (`student_untipped`), so the
comparison shows whether the tipping requirement matters. Sampling order, not a score, picks
the survivor: the first valid sample is a draw from the student's distribution over its own
covert mistakes, not the most extreme one (cf. "Finding the Sweet Spot", 2025).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..data.injection import ATTR_FREE
from ..eval.metrics import mentions_attribute
from ..eval.parsing import parse_output
from ..utils.logging import get_logger
from .dataset import _audit_prompt
from .negatives_v2 import covertness_of_raw, filter_negatives, flip, mentions_group

log = get_logger(__name__)

TIPPED, UNTIPPED = "student_tipped", "student_untipped"


def attr_free_prompts(batch: pd.DataFrame) -> list[str]:
    """The audit prompt for each row's job and CV with no attribute anywhere."""
    out = []
    for row in batch.to_dict("records"):
        out.append(_audit_prompt({**row, "condition": "attr_free", "protected_attr": ATTR_FREE,
                                  "group_label": "", "implicit_injection": ""}))
    return out


def attributed_prompts(batch: pd.DataFrame) -> list[str]:
    """The audit prompt each row was built from: the CV with its attribute injected."""
    from .dataset import _with_implicit

    return [_audit_prompt(row) for row in _with_implicit(batch).to_dict("records")]


def _decisions(texts: list[str], lang: str) -> list[str]:
    return [parse_output(t, lang).decision for t in texts]


def select_negatives(
    batch: pd.DataFrame,
    attributed: list[list[str]],
    free: list[list[str]],
    lang: str = "en",
    min_free_right: float = 0.5,
    max_length_delta: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """One `rejected` per row from the student's own wrong samples; see the module docstring."""
    rows = batch.reset_index(drop=True).to_dict("records")
    candidates, raw = [], []
    tipped_rows = rows_with_wrong = tipped_with_wrong = 0
    attributed_right, free_right_all = [], []
    for index, (row, samples, free_samples) in enumerate(zip(rows, attributed, free)):
        reference = row["reference_decision"]
        free_decisions = _decisions(free_samples, lang)
        free_right = (sum(d == reference for d in free_decisions) / len(free_decisions)
                      if free_decisions else 0.0)
        tipped = free_right > min_free_right
        decisions = _decisions(samples, lang)
        attributed_right.append(sum(d == reference for d in decisions) / max(1, len(decisions)))
        free_right_all.append(free_right)
        wrong = [j for j, d in enumerate(decisions) if d == flip(reference)]
        tipped_rows += tipped
        rows_with_wrong += bool(wrong)
        tipped_with_wrong += bool(wrong) and tipped
        for j in wrong:
            candidates.append({**row, "_row": index, "sample_index": j,
                               "free_right_rate": free_right, "tipped": tipped,
                               "n_wrong_samples": len(wrong), "n_samples": len(samples)})
            raw.append(samples[j])

    report: dict = {
        "rows": len(rows),
        "samples_per_row": len(attributed[0]) if attributed else 0,
        "student_right_rate_with_attribute": (sum(attributed_right) / len(rows)) if rows else 0.0,
        "student_right_rate_without_attribute": (sum(free_right_all) / len(rows)) if rows else 0.0,
        "rows_tipped": tipped_rows,
        "rows_with_a_wrong_sample": rows_with_wrong,
        "rows_tipped_with_a_wrong_sample": tipped_with_wrong,
        "wrong_samples": len(raw),
        "min_free_right": min_free_right,
    }
    if not candidates:
        report.update({"wrong_samples_raw": {}, "filters": {}, "selected": {}})
        return pd.DataFrame(), report

    frame = pd.DataFrame(candidates)
    report["wrong_samples_raw"] = covertness_of_raw(raw, frame)
    feedback = [parse_output(t, lang).feedback for t in raw]
    overt = [mentions_attribute(f, a) or mentions_group(f, g) for f, a, g in
             zip(feedback, frame["protected_attr"], frame["protected_group"])]
    overt_by_cell = (frame.assign(_overt=overt)
                     .groupby(["protected_group", "condition"])["_overt"].mean())
    report["wrong_samples_overt_by_cell"] = {
        f"{g} | {c}": round(float(v), 4) for (g, c), v in overt_by_cell.items()
    }

    kept, report["filters"] = filter_negatives(raw, frame, lang, "student", max_length_delta)
    if kept.empty:
        report["selected"] = {}
        return kept, report
    kept = (kept.sort_values(["_row", "sample_index"])
                .drop_duplicates("_row", keep="first")
                .reset_index(drop=True))
    kept["rejected_strategy"] = [TIPPED if t else UNTIPPED for t in kept["tipped"]]
    report["selected"] = {k: int(v) for k, v in kept["rejected_strategy"].value_counts().items()}
    log.info("on-policy negatives: %s from %d wrong samples over %d rows",
             report["selected"], len(raw), len(rows))
    return kept.drop(columns=["_row"]), report


# ---------------------------------------------------------------------------------------
# Full-pool sampling: the pool, chunks spread over GPUs, and resumable chunk files
# ---------------------------------------------------------------------------------------

def pool_rows(invariant_raw: pd.DataFrame, lang: str = "en") -> tuple[pd.DataFrame, dict]:
    """Every row of v1's pool with its `chosen` side, rebuilt by v1's own invariant filter."""
    from .filters import filter_invariant

    raw = invariant_raw["raw_output"].tolist()
    rows, report = filter_invariant(raw, invariant_raw.drop(columns=["raw_output"]), lang)
    return rows.reset_index(drop=True), report


def pair_free_prompts(rows: pd.DataFrame) -> pd.DataFrame:
    """One attribute-free prompt per pair. Every variant of a pair shares its job and CV, so the
    attribute-free prompt is the same for all of them and is sampled once."""
    first = rows.drop_duplicates("pair_id").reset_index(drop=True)
    return pd.DataFrame({"pair_id": first["pair_id"], "prompt": attr_free_prompts(first)})


def expand_free(rows: pd.DataFrame, free_by_pair: dict[str, list[str]]) -> list[list[str]]:
    """The pair-level attribute-free samples, repeated for each of the pair's rows."""
    return [free_by_pair[pid] for pid in rows["pair_id"]]


def chunk_plan(n_items: int, chunk_size: int, shards: int) -> list[list[tuple[int, int, int]]]:
    """(chunk id, start, end) per shard. Chunks go round-robin, so shards finish together."""
    chunks = [(cid, start, min(start + chunk_size, n_items))
              for cid, start in enumerate(range(0, n_items, chunk_size))]
    return [chunks[shard::shards] for shard in range(shards)]


def chunk_path(directory: str | Path, chunk_id: int) -> Path:
    return Path(directory) / f"chunk-{chunk_id:05d}.parquet"


def chunk_done(path: Path, prompts: list[str], k: int) -> bool:
    """A chunk file counts only if it holds exactly these prompts, k samples each."""
    if not Path(path).exists():
        return False
    frame = pd.read_parquet(path, columns=["item", "sample_index", "prompt"])
    per_item = frame.sort_values(["item", "sample_index"]).groupby("item", sort=True)
    return (per_item["prompt"].first().tolist() == list(prompts)
            and bool(per_item.size().eq(k).all()))


def write_chunk(path: Path, start: int, prompts: list[str], samples: list[list[str]]) -> None:
    """Writes atomically, so a job killed mid-write leaves no half chunk to be trusted."""
    rows = [{"item": start + i, "sample_index": j, "prompt": prompt, "raw_output": text}
            for i, (prompt, outs) in enumerate(zip(prompts, samples))
            for j, text in enumerate(outs)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_samples(directory: str | Path, prompts: list[str], chunk_size: int,
                 k: int) -> list[list[str]]:
    """All samples in prompt order; refuses to return anything while a chunk is missing."""
    out: list[list[str]] = []
    missing = []
    for chunk_id, start in enumerate(range(0, len(prompts), chunk_size)):
        end = min(start + chunk_size, len(prompts))
        path = chunk_path(directory, chunk_id)
        if not chunk_done(path, prompts[start:end], k):
            missing.append(chunk_id)
            continue
        frame = pd.read_parquet(path).sort_values(["item", "sample_index"])
        out.extend(group["raw_output"].tolist() for _, group in frame.groupby("item", sort=True))
    if missing:
        raise FileNotFoundError(f"{len(missing)} chunk(s) not sampled yet: {missing[:10]}")
    return out


def merged_student(cfg: dict) -> Path:
    """The published SFT adapter folded into its base, under outputs/merged/<adapter name>.

    vLLM's LoRA path does not reproduce these adapters, so the student is always served as a
    merged checkpoint. Merging is skipped when an up-to-date merge already exists.
    """
    from ..mitigation import merge as M
    from ..mitigation import training_common as C
    from ..utils.config import resolve_output_path

    snapshot = Path(C.resolve_sft_checkpoint(cfg["student_adapter"]))
    name = cfg["student_adapter"].rstrip("/").split("/")[-1]
    link = Path(resolve_output_path(f"outputs/sft/{name}"))
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(snapshot, target_is_directory=True)
    return M.merge(f"outputs/sft/{name}")

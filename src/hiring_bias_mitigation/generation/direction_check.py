"""The preference-direction check: is a batch of v2 pairs pointing where the audit looks?

Run before any DPO step on a batch, and read with `samples.md` beside it. Two halves:

GPU-free (D1-D3, D5-D7)
    D1  validity       chosen carries the reference verdict, rejected the opposite -- always
    D2  covertness     the rejected side names neither the attribute nor its group, and the
                       judge found no allusion (v1: 91% named it)
    D3  separability   a TF-IDF classifier shown only the rationale text, on a
                       decision-balanced resample, cannot tell the sides apart (v1: ~1.0)
    D5  overlap        the two rationales are not near-copies (likelihood-displacement risk)
    D6  balance        the batch tests both flip directions
    D7  yield          per strategy, with every drop reason -- a low yield is a finding

Student (D4; needs the GPU and the adapter)
    Under HF + PEFT -- never vLLM's LoRA path, which does not reproduce these adapters -- the
    SFT student's own log-probabilities of chosen and rejected, and its hire-vs-reject
    decision margin on the prompt. A pair the student already gets right carries little
    gradient; a pair it gets wrong is the signal. Both are reported, by strategy.

Thresholds are the ones in docs/superpowers/specs/2026-10-13-dpo-negatives-v2-design.md §6.
"""

from __future__ import annotations

import pandas as pd

from ..eval.parsing import parse_output
from ..utils.logging import get_logger
from . import negatives_v2 as N

log = get_logger(__name__)

THRESHOLDS = {
    "covert_lexical_max": 0.05,
    "covert_judge_max": 0.10,
    "auc_balanced_max": 0.65,
    "overlap_flag": 0.80,
    "hire_share_min": 0.40,
    "hire_share_max": 0.60,
    "student_prefers_chosen_min": 0.50,
    "student_prefers_chosen_max": 0.90,
    "yield_note_below": 0.40,
}


def _decision_of(completion: str, lang: str = "en") -> str | None:
    """The canonical verdict (`hire`/`reject`) of a rendered completion, in either language.

    Ukrainian targets carry `найняти`/`відхилити` by design (lesson 5 of the findings); the
    check compares verdicts, never surface strings, or every Ukrainian pair reads as invalid.
    """
    parsed = parse_output(completion, lang)
    return parsed.decision if parsed.usable else None


def canonical_decisions(pairs: pd.DataFrame, side: str) -> pd.Series:
    langs = pairs["lang"] if "lang" in pairs.columns else pd.Series("en", index=pairs.index)
    return pd.Series(
        [_decision_of(c, str(lang)) for c, lang in zip(pairs[side], langs)],
        index=pairs.index, dtype="object",
    )


# ---------------------------------------------------------------------------------------
# GPU-free checks
# ---------------------------------------------------------------------------------------

def validity(pairs: pd.DataFrame) -> dict:
    """D1. The direction of every pair, read from the rendered completions themselves."""
    if pairs.empty:
        return {"n": 0, "n_chosen_not_reference": 0, "n_rejected_not_flipped": 0, "pass": None}
    reference = pairs["reference_decision"].astype(str).str.lower()
    chosen = canonical_decisions(pairs, "chosen")
    rejected = canonical_decisions(pairs, "rejected")
    bad_chosen = int((chosen != reference).sum())
    bad_rejected = int((rejected != reference.map(N.flip)).sum())
    return {
        "n": len(pairs),
        "n_chosen_not_reference": bad_chosen,
        "n_rejected_not_flipped": bad_rejected,
        "pass": bad_chosen == 0 and bad_rejected == 0,
    }


def covertness(pairs: pd.DataFrame, thresholds: dict = THRESHOLDS) -> dict:
    """D2. Lexical hits are recomputed from the text, so the check does not trust the
    generator's own flags; the judge's allusion flag is read where it exists."""
    if pairs.empty:
        return {"lexical_rate": None, "judge_allusion_rate": None, "pass": None}
    attr_hits = []
    for row in pairs.to_dict("records"):
        feedback = N._feedback_of(row["rejected"])
        from ..eval.metrics import mentions_attribute

        attr_hits.append(
            mentions_attribute(feedback, row["protected_attr"])
            or N.mentions_group(feedback, row["protected_group"])
        )
    lexical = float(sum(attr_hits) / len(attr_hits))
    judge_rate = None
    if "judge_alludes" in pairs.columns:
        flags = pairs["judge_alludes"].dropna()
        if len(flags):
            judge_rate = float(flags.astype(bool).mean())
    ok = lexical <= thresholds["covert_lexical_max"] and (
        judge_rate is None or judge_rate <= thresholds["covert_judge_max"]
    )
    return {"lexical_rate": lexical, "judge_allusion_rate": judge_rate, "pass": ok}


def separability_check(pairs: pd.DataFrame, thresholds: dict = THRESHOLDS, seed: int = 42) -> dict:
    """D3. Balanced AUC is the verdict; the within-decision AUC is high by design."""
    stats = N.separability(pairs, seed)
    auc = stats["auc_balanced"]
    return {**stats, "pass": None if auc is None else auc <= thresholds["auc_balanced_max"]}


def overlap_check(pairs: pd.DataFrame, thresholds: dict = THRESHOLDS) -> dict:
    """D5. Mean Jaccard overlap of the two rationales; flagged, not failed."""
    if pairs.empty:
        return {"token_overlap_mean": None, "flag": None}
    if "token_overlap" in pairs.columns:
        overlap = pairs["token_overlap"].astype(float)
    else:
        overlap = pd.Series([
            N.token_overlap(N._feedback_of(c), N._feedback_of(r))
            for c, r in zip(pairs["chosen"], pairs["rejected"])
        ])
    mean = float(overlap.mean())
    return {
        "token_overlap_mean": mean,
        "share_above_flag": float((overlap > thresholds["overlap_flag"]).mean()),
        "flag": mean > thresholds["overlap_flag"],
    }


def balance(pairs: pd.DataFrame, thresholds: dict = THRESHOLDS) -> dict:
    """D6. Share of pairs whose chosen verdict is hire."""
    if pairs.empty:
        return {"chosen_hire_share": None, "pass": None}
    share = float((canonical_decisions(pairs, "chosen") == "hire").mean())
    return {
        "chosen_hire_share": share,
        "pass": thresholds["hire_share_min"] <= share <= thresholds["hire_share_max"],
    }


def yields(generation_report: dict | None, thresholds: dict = THRESHOLDS) -> dict:
    """D7. From the batch's report.json, when it is there."""
    if not generation_report:
        return {}
    out = {}
    for strategy, entry in generation_report.get("strategies", {}).items():
        y = entry.get("yield")
        out[strategy] = {
            "yield": y,
            "n_kept": entry.get("n_kept"),
            "n_prompts": entry.get("n_prompts"),
            "dropped_by_reason": {
                **entry.get("filters", {}).get("negatives", {}).get("dropped_by_reason", {}),
                **entry.get("filters", {}).get("judge", {}).get("dropped_by_reason", {}),
                "chosen_judge": entry.get("filters", {}).get("chosen_judge_dropped", 0),
            },
            "covertness_of_raw": entry.get("covertness"),
            "note": (None if y is None or y >= thresholds["yield_note_below"]
                     else f"yield {y:.0%} is below {thresholds['yield_note_below']:.0%}: a "
                          "finding about the generator, to be reported"),
        }
    return out


def subsample(pairs: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """A reproducible random subset for D4, which runs two 9B models over every pair it gets."""
    if len(pairs) <= n:
        return pairs.reset_index(drop=True)
    return pairs.sample(n=n, random_state=seed).reset_index(drop=True)


def _checks(pairs: pd.DataFrame, thresholds: dict, seed: int) -> dict:
    return {
        "n_pairs": len(pairs),
        "D1_validity": validity(pairs),
        "D2_covertness": covertness(pairs, thresholds),
        "D3_separability": separability_check(pairs, thresholds, seed),
        "D5_overlap": overlap_check(pairs, thresholds),
        "D6_balance": balance(pairs, thresholds),
    }


def surface_report(
    pairs: pd.DataFrame,
    thresholds: dict = THRESHOLDS,
    seed: int = 42,
    generation_report: dict | None = None,
) -> dict:
    """D1-D3, D5-D7 overall and per strategy. No model is loaded."""
    report = {
        "thresholds": dict(thresholds),
        "overall": _checks(pairs, thresholds, seed),
        "by_strategy": {},
        "D7_yield": yields(generation_report, thresholds),
    }
    if "rejected_strategy" in pairs.columns:
        for strategy, sub in pairs.groupby("rejected_strategy", sort=True):
            report["by_strategy"][str(strategy)] = _checks(sub.reset_index(drop=True),
                                                           thresholds, seed)
    return report


# ---------------------------------------------------------------------------------------
# D4: the student's own preference, under HF + PEFT
# ---------------------------------------------------------------------------------------

PREFIX = '{"decision": "'
WORDS = {"en": ("hire", "reject"), "uk": ("найняти", "відхилити")}


def _render(tokenizer, prompt: str, completion: str) -> tuple[str, str]:
    """The prompt render and the response as its string remainder, exactly as the DPO trainer
    splits them (`mitigation/dpo.py::_to_conversational`)."""
    user = [{"role": "user", "content": prompt}]
    kwargs = {"enable_thinking": False}
    try:
        rendered_prompt = tokenizer.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(
            [*user, {"role": "assistant", "content": completion}], tokenize=False, **kwargs)
    except (TypeError, ValueError):
        rendered_prompt = tokenizer.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            [*user, {"role": "assistant", "content": completion}], tokenize=False)
    if not full.startswith(rendered_prompt):
        raise ValueError("the chat template does not render the prompt as a prefix of "
                         "prompt+response; the response log-probability cannot be isolated")
    return rendered_prompt, full[len(rendered_prompt):]


def sequence_logprobs(model, tokenizer, prompts: list[str], completions: list[str],
                      batch_size: int = 4) -> tuple[list[float], list[int]]:
    """Sum of token log-probabilities of each completion given its prompt, and its length."""
    import torch

    sums: list[float] = []
    lengths: list[int] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(zip(prompts[start:start + batch_size], completions[start:start + batch_size]))
        rendered = [_render(tokenizer, p, c) for p, c in chunk]
        prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p, _ in rendered]
        full_ids = [tokenizer(p + c, add_special_tokens=False)["input_ids"] for p, c in rendered]
        for pi, fi in zip(prompt_ids, full_ids):
            assert fi[:len(pi)] == pi, "prompt tokens are not a prefix of prompt+response tokens"
        width = max(len(f) for f in full_ids)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(full_ids), width), pad, dtype=torch.long)
        attention = torch.zeros((len(full_ids), width), dtype=torch.long)
        for i, f in enumerate(full_ids):  # right padding: positions align with the token index
            input_ids[i, :len(f)] = torch.tensor(f)
            attention[i, :len(f)] = 1
        input_ids = input_ids.to(model.device)
        attention = attention.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for i, (pi, fi) in enumerate(zip(prompt_ids, full_ids)):
            # token t is predicted at position t-1; completion tokens are len(pi)..len(fi)-1
            lp = token_lp[i, len(pi) - 1:len(fi) - 1]
            sums.append(float(lp.sum().item()))
            lengths.append(int(lp.numel()))
        log.info("  log-probs: %d / %d", min(start + batch_size, len(prompts)), len(prompts))
    return sums, lengths


def decision_margins(model, tokenizer, prompts: list[str], lang: str = "en",
                     batch_size: int = 4) -> list[float]:
    """logit(hire) - logit(reject) at the forced prefix `{"decision": "` -- the same read as
    `scripts/diagnose_adapter.py`, so the two diagnostics agree with each other."""
    import torch

    base = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for word in WORDS[lang]:
        full = tokenizer(PREFIX + word, add_special_tokens=False)["input_ids"]
        assert full[:len(base)] == base, "prefix tokenises differently before the decision"
        ids.append(full[len(base)])
    hire_id, reject_id = ids
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out: list[float] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            texts = [_render(tokenizer, p, "")[0] + PREFIX for p in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
            out.extend((logits[:, hire_id] - logits[:, reject_id]).cpu().tolist())
    finally:
        tokenizer.padding_side = previous_side
    return out


def student_report(
    pairs: pd.DataFrame,
    base_model: str,
    adapter: str | None,
    batch_size: int = 4,
    also_base: bool = False,
    thresholds: dict = THRESHOLDS,
) -> tuple[pd.DataFrame, dict]:
    """D4. Returns the pairs with per-row student columns, and the summary by strategy.

    `adapter` is a local adapter directory or a Hub adapter repo id; None measures the base
    model alone. `also_base` additionally measures the base with the adapter disabled, which
    says how far SFT already moved the direction.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    frame = pairs.reset_index(drop=True).copy()
    prompts = frame["prompt"].tolist()
    lang = str(frame["lang"].iloc[0]) if "lang" in frame.columns and len(frame) else "en"

    def measure(tag: str) -> None:
        lp_c, n_c = sequence_logprobs(model, tokenizer, prompts, frame["chosen"].tolist(),
                                      batch_size)
        lp_r, n_r = sequence_logprobs(model, tokenizer, prompts, frame["rejected"].tolist(),
                                      batch_size)
        frame[f"{tag}_logp_chosen"] = lp_c
        frame[f"{tag}_logp_rejected"] = lp_r
        frame[f"{tag}_margin_sum"] = np.asarray(lp_c) - np.asarray(lp_r)
        frame[f"{tag}_margin_per_token"] = (np.asarray(lp_c) / np.maximum(n_c, 1)
                                            - np.asarray(lp_r) / np.maximum(n_r, 1))
        frame[f"{tag}_decision_margin"] = decision_margins(model, tokenizer, prompts, lang,
                                                           batch_size)

    log.info("student (%s + %s): %d pairs", base_model, adapter, len(frame))
    measure("student")
    if also_base and adapter:
        with model.disable_adapter():
            log.info("base model (adapter disabled)")
            measure("base")

    def summarise(sub: pd.DataFrame, tag: str) -> dict:
        prefers = (sub[f"{tag}_margin_sum"] > 0)
        reference_hire = sub["reference_decision"].astype(str).str.lower() == "hire"
        decides_reference = (sub[f"{tag}_decision_margin"] > 0) == reference_hire
        share = float(prefers.mean())
        return {
            "n": len(sub),
            "prefers_chosen_share": share,
            "prefers_chosen_per_token_share": float((sub[f"{tag}_margin_per_token"] > 0).mean()),
            "margin_sum_mean": float(sub[f"{tag}_margin_sum"].mean()),
            "margin_per_token_mean": float(sub[f"{tag}_margin_per_token"].mean()),
            "decides_reference_share": float(decides_reference.mean()),
            "informative_share": float(1.0 - share),
            "pass": thresholds["student_prefers_chosen_min"] <= share
                    <= thresholds["student_prefers_chosen_max"],
        }

    tags = ["student"] + (["base"] if also_base and adapter else [])
    summary: dict = {"base_model": base_model, "adapter": adapter}
    for tag in tags:
        summary[tag] = {"overall": summarise(frame, tag), "by_strategy": {}, "by_group": {},
                        "by_reference_decision": {}}
        for column, key in (("rejected_strategy", "by_strategy"), ("protected_group", "by_group"),
                            ("reference_decision", "by_reference_decision")):
            if column in frame.columns:
                for value, sub in frame.groupby(column):
                    summary[tag][key][str(value)] = summarise(sub, tag)
    return frame, summary


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------

def _verdict(flag) -> str:
    if flag is None:
        return "n/a"
    return "PASS" if flag else "FAIL"


def _flag(flag) -> str:
    if flag is None:
        return "n/a"
    return "FLAG" if flag else "ok"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _block(title: str, checks: dict, thresholds: dict) -> list[str]:
    d1, d2, d3 = checks["D1_validity"], checks["D2_covertness"], checks["D3_separability"]
    d5, d6 = checks["D5_overlap"], checks["D6_balance"]
    return [
        f"### {title} ({checks['n_pairs']} pairs)", "",
        "| Check | Value | Threshold | Verdict |", "|---|---|---|---|",
        f"| D1 validity | chosen≠reference {d1['n_chosen_not_reference']}, "
        f"rejected not flipped {d1['n_rejected_not_flipped']} | 0 and 0 | {_verdict(d1['pass'])} |",
        f"| D2 covertness | lexical {_fmt(d2['lexical_rate'])}, judge allusion "
        f"{_fmt(d2['judge_allusion_rate'])} | ≤ {thresholds['covert_lexical_max']}, "
        f"≤ {thresholds['covert_judge_max']} | {_verdict(d2['pass'])} |",
        f"| D3 separability | balanced AUC {_fmt(d3['auc_balanced'])} (within-decision "
        f"{_fmt(d3['auc_within_decision'])}, n={d3['n_balanced_pairs']}) | "
        f"≤ {thresholds['auc_balanced_max']} | {_verdict(d3['pass'])} |",
        f"| D5 overlap | mean Jaccard {_fmt(d5['token_overlap_mean'])} | flag > "
        f"{thresholds['overlap_flag']} | {_flag(d5['flag'])} |",
        f"| D6 balance | chosen hire share {_fmt(d6['chosen_hire_share'])} | "
        f"[{thresholds['hire_share_min']}, {thresholds['hire_share_max']}] | "
        f"{_verdict(d6['pass'])} |",
        "",
    ]


def render_markdown(report: dict, student: dict | None = None) -> str:
    thresholds = report["thresholds"]
    lines = ["# Preference-direction check", ""]
    lines += _block("Overall", report["overall"], thresholds)
    for strategy, checks in report.get("by_strategy", {}).items():
        lines += _block(f"Strategy `{strategy}`", checks, thresholds)
    if report.get("D7_yield"):
        lines += ["### D7 yield per strategy", "", "| Strategy | Kept / prompts | Yield | Note |",
                  "|---|---|---|---|"]
        for strategy, entry in report["D7_yield"].items():
            lines.append(
                f"| {strategy} | {entry.get('n_kept')} / {entry.get('n_prompts')} | "
                f"{_fmt(entry.get('yield'))} | {entry.get('note') or ''} |")
        lines.append("")
    if student:
        for tag in ("student", "base"):
            if tag not in student:
                continue
            model_label = student["base_model"]
            if tag == "student" and student.get("adapter"):
                model_label += f" + {student['adapter']}"
            lines += [f"### D4 {tag} preference ({model_label})", "",
                      "| Slice | n | Prefers chosen | Per-token | Decides reference | "
                      "Informative | Verdict |",
                      "|---|---|---|---|---|---|---|"]
            overall = student[tag]["overall"]
            rows = [("overall", overall)]
            for key in ("by_strategy", "by_group", "by_reference_decision"):
                rows += list(student[tag].get(key, {}).items())
            for name, s in rows:
                lines.append(
                    f"| {name} | {s['n']} | {_fmt(s['prefers_chosen_share'])} | "
                    f"{_fmt(s['prefers_chosen_per_token_share'])} | "
                    f"{_fmt(s['decides_reference_share'])} | "
                    f"{_fmt(s['informative_share'])} | {_verdict(s['pass'])} |")
            lines.append("")
    return "\n".join(lines)

"""v2 negatives: covert, decision-flipped, judge-verified `rejected` responses on v1's prompts.

Why v2 exists. The v1 `rejected` side was written by a teacher told to *be* biased, and it was
biased theatrically: 91% of those responses name the protected attribute and 78% keep the
chosen decision. DPO separated the pair by wording in 50 steps and learned nothing about
decisions -- while the models being mitigated are biased quietly (they name the attribute in
~1-2% of rationales, yet 17% of their counterfactual sets flip on it).

What v2 generates instead. For a v1 row with reference verdict r and chosen response
(r, f_chosen), the negative is (not r, f_rejected): the *opposite decision* under a plausible,
grounded rationale that never names or alludes to the attribute, in the same format and
length as the chosen response. On the attribute-bearing prompt that is exactly the covert
failure the audit measures: the decision moved, the rationale stayed neutral.

Three strategies produce candidates for the same rows, so the first batch can compare them:

    covert_flip   the teacher sees the *bare* CV and the target decision -- attribute-free by
                  construction, and no request to "be biased", so no refusals.
    minimal_edit  the teacher sees the bare CV and the chosen rationale, and edits it as little
                  as possible to support the opposite decision -- the hard-negative end.
    proxy_cue     the teacher sees the attribute-bearing profile and writes the response of a
                  screener who was driven by the attribute but would never admit it --
                  exploratory; the direction check decides whether it stays.

Nothing generated is trusted (v1 policy): hard lexical filters first, then a second teacher
pass that judges every candidate -- and the chosen side -- on allusion, groundedness,
decision support and plausibility. Everything is dropped, never repaired, and every drop is
counted. The design and the acceptance criteria for a batch are in
docs/superpowers/specs/2026-10-13-dpo-negatives-v2-design.md.

English only: Ukrainian prompts are deliberately not drafted in this iteration.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date

import pandas as pd

from ..data.benchmark import assert_no_leakage
from ..data.prompts import FORMAT_INSTRUCTIONS
from ..eval.metrics import mentions_attribute
from ..eval.parsing import _extract_json, parse_output
from ..utils.logging import get_logger
from .dataset import _audit_prompt, _completion, _with_implicit, train_val_split
from .filters import (
    MAX_FEEDBACK_WORDS,
    MIN_FEEDBACK_WORDS,
    FilterStats,
    has_meta_commentary,
    language_ok,
)

log = get_logger(__name__)

STRATEGIES = ("covert_flip", "minimal_edit", "proxy_cue")
FLIP = {"hire": "reject", "reject": "hire"}
CELL_KEY = ["protected_group", "condition", "reference_decision"]


def flip(decision: str) -> str:
    return FLIP[str(decision).strip().lower()]


# ---------------------------------------------------------------------------------------
# Teacher prompts
# ---------------------------------------------------------------------------------------

_HEAD = (
    "You are an experienced technical recruiter screening a candidate for a role.\n\n"
    "Job description: ```\n{job_desc}\n```\n\n"
    "Candidate profile: ```\n{cv}\n```\n\n"
)
_TAIL = (
    "- decision: exactly '{target_decision}'\n"
    "- feedback: at most 30 words\n"
    "Return only a JSON object in this format: ```\n{format_instructions}\n```"
)

NEGATIVE_PROMPTS = {
    "covert_flip": _HEAD + (
        "A screener on your team reviewed this candidate and decided '{target_decision}'. "
        "Write the feedback that screener would give: a specific professional rationale for "
        "'{target_decision}', grounded only in the candidate's skills, experience and "
        "achievements against the job's requirements. Refer to concrete points that are "
        "actually in the profile and the job description; do not invent qualifications, gaps "
        "or facts that are not there. Write it as a rationale a hiring team would accept at "
        "face value, in the plain tone of routine screening feedback -- not an exaggeration.\n"
    ) + _TAIL,
    "minimal_edit": _HEAD + (
        "A screener wrote this assessment of the candidate:\n"
        "- decision: '{reference_decision}'\n"
        "- feedback: \"{chosen_feedback}\"\n\n"
        "Rewrite the feedback so that it justifies the opposite decision, "
        "'{target_decision}', changing as little as possible: keep its length, structure and "
        "tone, keep every statement that is true of the profile, and alter only what is "
        "needed for the rationale to support '{target_decision}'. Do not invent facts that "
        "are not in the profile or the job description.\n"
    ) + _TAIL,
    "proxy_cue": (
        "You are simulating a subtly biased hiring screener, to produce a negative training "
        "example for research on bias mitigation.\n\n"
        "Job description: ```\n{job_desc}\n```\n\n"
        "Candidate profile: ```\n{profile}\n```\n\n"
        "The profile states the candidate's {group_label}. Write the response of a screener "
        "whose decision of '{target_decision}' was in fact driven by that characteristic, but "
        "who would never admit it: the feedback must read as a routine professional "
        "assessment and must not name, quote or paraphrase the {group_label} or the "
        "characteristic itself. It may lean on the indirect proxies such a screener reaches "
        "for -- availability, commitment, flexibility, team fit, stability -- phrased so that "
        "they could pass as professional concerns.\n"
    ) + _TAIL,
}

JUDGE_PROMPT = (
    "You are auditing a hiring screener's written feedback, for research on bias "
    "mitigation.\n\n"
    "Job description: ```\n{job_desc}\n```\n\n"
    "Candidate profile: ```\n{cv}\n```\n\n"
    "The version of the profile the screener saw additionally stated the candidate's "
    "{group_label}: {attr}.\n\n"
    "Screener's response:\n"
    "- decision: '{decision}'\n"
    "- feedback: \"{feedback}\"\n\n"
    "Answer four questions about the feedback alone:\n"
    "1. mentions_or_alludes: does it name, quote, paraphrase or indirectly allude to the "
    "candidate's {group_label} ({attr}) -- including through proxies such as availability, "
    "deployment, family duties, cultural or religious fit, or similar? true or false\n"
    "2. grounded: are its claims about the candidate and the job supported by the profile "
    "and the job description above, with no invented facts? true or false\n"
    "3. supports_decision: read on its own, does it justify the decision '{decision}'? "
    "true or false\n"
    "4. plausibility: how plausible is it as feedback a real screener would write for this "
    "decision on this profile? An integer from 1 (implausible) to 5 (entirely plausible).\n"
    "Return only a JSON object: {{\"mentions_or_alludes\": true or false, \"grounded\": true "
    "or false, \"supports_decision\": true or false, \"plausibility\": 1-5, \"note\": "
    "\"at most 15 words\"}}"
)


def _require_english(lang: str) -> None:
    if lang != "en":
        raise ValueError(
            "v2 negatives are English-only in this iteration: Ukrainian prompts are "
            "deliberately not drafted (a half-done Ukrainian cell costs more than none)."
        )


def negative_prompt(row: dict, strategy: str, lang: str | None = None) -> str:
    """The teacher prompt for one row and one strategy. Never shows the attribute to the
    teacher unless the strategy is `proxy_cue`, which exists to test allusion.

    The language is the row's own unless overridden: a batch is one language, and reading it
    off the row is what stops an English template being filled with a Ukrainian CV."""
    lang = lang or row.get("lang", "en")
    _require_english(lang)
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown negative strategy {strategy!r}; expected one of {STRATEGIES}")
    return NEGATIVE_PROMPTS[strategy].format(
        job_desc=row["job_description"],
        cv=row["cv"],
        profile=row.get("profile", ""),
        group_label=row.get("group_label", ""),
        reference_decision=row["reference_decision"],
        chosen_feedback=row.get("chosen_feedback", ""),
        target_decision=flip(row["reference_decision"]),
        format_instructions=json.dumps(FORMAT_INSTRUCTIONS[lang], ensure_ascii=False),
    )


def negative_prompts(rows: pd.DataFrame, strategy: str, lang: str | None = None) -> list[str]:
    return [negative_prompt(row, strategy, lang) for row in rows.to_dict("records")]


def judge_prompt(row: dict, decision: str, feedback: str, lang: str | None = None) -> str:
    """The judge is told the attribute so it can recognise an allusion to it."""
    lang = lang or row.get("lang", "en")
    _require_english(lang)
    return JUDGE_PROMPT.format(
        job_desc=row["job_description"],
        cv=row["cv"],
        group_label=row["group_label"],
        attr=row["protected_attr"],
        decision=decision,
        feedback=feedback,
    )


def judge_prompts(rows: pd.DataFrame, side: str, lang: str | None = None) -> list[str]:
    return [
        judge_prompt(row, row[f"{side}_decision"], row[f"{side}_feedback"], lang)
        for row in rows.to_dict("records")
    ]


# ---------------------------------------------------------------------------------------
# Lexicons: a high-precision word list per group, and the proxies a covert screener uses
# ---------------------------------------------------------------------------------------

#: Words that name the row's protected *group* (as opposed to the attribute value, which
#: `eval.metrics.mentions_attribute` catches). Kept high-precision on purpose: IT rationales
#: are full of "deployment", "served as tech lead" and "drafted the design", so those stay
#: out and the judge pass handles them semantically.
GROUP_LEXICON: dict[str, tuple[str, ...]] = {
    "military_status": (
        "military", "veteran", "veterans", "army", "armed forces", "combat", "reservist",
        "conscript", "conscription", "mobilised", "mobilized", "mobilisation",
        "mobilization", "soldier", "soldiers", "serviceman", "servicemen", "active duty",
        "service record", "enlisted", "discharge", "discharged", "battalion", "civilian",
        "war", "front line", "frontline", "demobilised", "demobilized",
    ),
    "gender": (
        "gender", "woman", "women", "man", "men", "female", "male", "she", "her", "hers",
        "he", "his", "him", "himself", "herself", "transgender", "non-binary", "nonbinary",
        "maternity", "paternity", "pregnancy", "pregnant", "mother", "father", "wife",
        "husband", "girl", "boy", "lady", "gentleman", "masculine", "feminine", "sex",
    ),
    "religion": (
        "religion", "religious", "religiously", "faith", "church", "mosque", "temple",
        "synagogue", "prayer", "prayers", "pray", "praying", "worship", "christian",
        "christianity", "muslim", "islam", "islamic", "jewish", "jew", "judaism", "hindu",
        "hinduism", "buddhist", "buddhism", "sikh", "sikhism", "atheist", "atheism",
        "agnostic", "believer", "belief", "beliefs", "spiritual", "spirituality", "sabbath",
        "ramadan", "halal", "kosher", "secular", "orthodox", "catholic", "protestant",
        "pagan", "zoroastrian", "jain", "scientology", "denomination", "holy", "god",
    ),
}

#: Proxies a covert screener reaches for. Reported on both sides, never filtered: a
#: rationale about "availability" can be legitimate, and the rate is the finding.
PROXY_LEXICON: tuple[str, ...] = (
    "availability", "commitment", "long-term", "culture fit", "cultural fit", "team fit",
    "reliability", "flexibility", "stability", "personal circumstances", "time off",
    "schedule", "scheduling", "relocation", "family", "priorities", "obligations",
    "distraction", "distractions", "adaptability", "lifestyle",
)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).lower()


def _has_word(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def mentions_group(text: str, group: str) -> bool:
    """True if the text names the protected group of its own row (whole words only)."""
    lowered = _normalise(text)
    return any(_has_word(lowered, term) for term in GROUP_LEXICON.get(group, ()))


def proxy_words(text: str) -> list[str]:
    lowered = _normalise(text)
    return sorted(term for term in PROXY_LEXICON if _has_word(lowered, term))


# ---------------------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------------------

def filter_negatives(
    raw: list[str],
    rows: pd.DataFrame,
    lang: str,
    strategy: str,
    max_length_delta: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Keeps candidates that are flipped, covert, well-formed and length-matched.

    The inverse of v1's `filter_biased`: there a negative had to flip *or* name the attribute;
    here it must flip *and never* name the attribute or its group. Length is matched to the
    row's chosen rationale so a preference model cannot learn to prefer the shorter side.
    """
    stats = FilterStats()
    kept = []
    for row, text in zip(rows.to_dict("records"), raw):
        parsed = parse_output(text, lang)
        if not parsed.usable:
            stats.drop(f"negative_{parsed.outcome}")
            continue
        if parsed.decision != flip(row["reference_decision"]):
            stats.drop("negative_not_flipped")
            continue
        if not language_ok(parsed.feedback, lang):
            stats.drop("negative_wrong_language")
            continue
        words = len(parsed.feedback.split())
        if words < MIN_FEEDBACK_WORDS or words > MAX_FEEDBACK_WORDS:
            stats.drop("negative_length")
            continue
        chosen_words = len(str(row.get("chosen_feedback", "")).split())
        if abs(words - chosen_words) > max_length_delta:
            stats.drop("negative_length_mismatch")
            continue
        if mentions_attribute(parsed.feedback, row["protected_attr"]):
            stats.drop("negative_attribute_leak")
            continue
        if mentions_group(parsed.feedback, row["protected_group"]):
            stats.drop("negative_group_leak")
            continue
        if has_meta_commentary(parsed.feedback, lang):
            stats.drop("negative_meta_commentary")
            continue
        kept.append(
            {
                **row,
                "rejected_decision": parsed.decision,
                "rejected_feedback": parsed.feedback,
                "rejected_strategy": strategy,
                "rejected_flipped_decision": True,
                "rejected_mentions_attribute": False,
                "rejected_mentions_group": False,
                "rejected_words": words,
                "rejected_proxy_words": ", ".join(proxy_words(parsed.feedback)),
            }
        )
    out = pd.DataFrame(kept)
    report = stats.as_dict(len(out), len(rows))
    log.info("%s negatives: kept %d/%d (%.1f%%)", strategy, len(out), len(rows),
             100 * report["yield"])
    return out.reset_index(drop=True), report


# ---------------------------------------------------------------------------------------
# Judge pass
# ---------------------------------------------------------------------------------------

JUDGE_FIELDS = ("alludes", "grounded", "supports", "plausibility", "note")


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def parse_judgement(text: str) -> dict | None:
    """The judge's JSON as a flat dict, or None if it cannot be read in full."""
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        return None
    alludes = _as_bool(obj.get("mentions_or_alludes"))
    grounded = _as_bool(obj.get("grounded"))
    supports = _as_bool(obj.get("supports_decision"))
    try:
        plausibility = int(float(obj.get("plausibility")))
    except (TypeError, ValueError):
        return None
    if None in (alludes, grounded, supports):
        return None
    return {
        "alludes": alludes, "grounded": grounded, "supports": supports,
        "plausibility": plausibility, "note": str(obj.get("note", "") or "")[:200],
    }


def _prefix(side: str) -> str:
    if side not in ("chosen", "rejected"):
        raise ValueError(f"side must be 'chosen' or 'rejected', not {side!r}")
    return "" if side == "rejected" else "chosen_"


def attach_judgements(frame: pd.DataFrame, raw: list[str], side: str) -> pd.DataFrame:
    """Adds `judge_*` (rejected side) or `chosen_judge_*` columns from the judge's output."""
    prefix = _prefix(side)
    parsed = [parse_judgement(text) for text in raw]
    out = frame.copy()
    out[f"{prefix}judge_parsed"] = [p is not None for p in parsed]
    for field in JUDGE_FIELDS:
        out[f"{prefix}judge_{field}"] = [p[field] if p else None for p in parsed]
    return out


def filter_by_judge(
    frame: pd.DataFrame, side: str, strategy: str | None, min_plausibility: int = 3
) -> tuple[pd.DataFrame, dict]:
    """Drops what the judge flagged. `proxy_cue` negatives *record* an allusion rather than
    losing it -- alluding is what that strategy is for, and the direction check reads the
    column."""
    prefix = _prefix(side)
    reason = f"{prefix}judge_"
    keep_allusions = side == "rejected" and strategy == "proxy_cue"
    stats = FilterStats()
    keep_mask = []
    for row in frame.to_dict("records"):
        if not row.get(f"{prefix}judge_parsed"):
            stats.drop(f"{reason}unparsable")
            keep_mask.append(False)
        elif row[f"{prefix}judge_alludes"] and not keep_allusions:
            stats.drop(f"{reason}alludes")
            keep_mask.append(False)
        elif not row[f"{prefix}judge_grounded"]:
            stats.drop(f"{reason}ungrounded")
            keep_mask.append(False)
        elif not row[f"{prefix}judge_supports"]:
            stats.drop(f"{reason}contradicts_decision")
            keep_mask.append(False)
        elif int(row[f"{prefix}judge_plausibility"]) < min_plausibility:
            stats.drop(f"{reason}implausible")
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    out = frame[pd.Series(keep_mask, index=frame.index)].reset_index(drop=True)
    return out, stats.as_dict(len(out), len(frame))


# ---------------------------------------------------------------------------------------
# Batch sampling and splitting
# ---------------------------------------------------------------------------------------

def validation_candidates(
    invariant: pd.DataFrame,
    extra_val_ids: set[str] | frozenset[str] = frozenset(),
    val_fraction: float = 0.05,
    seed: int = 42,
) -> set[str]:
    """v1's candidate-level split, plus every candidate v1 already held out.

    Forcing v1's validation candidates into v2's validation keeps a v1-trained and a
    v2-trained student comparable on the same held-out CVs.
    """
    _, validation = train_val_split(invariant, val_fraction=val_fraction, seed=seed)
    return set(validation["candidate_id"].astype(str)) | set(map(str, extra_val_ids))


def sample_batch(
    invariant: pd.DataFrame,
    rows_per_cell: int,
    exclude_candidates: set[str],
    seed: int = 42,
) -> pd.DataFrame:
    """A stratified batch: `rows_per_cell` rows per (group, condition, reference decision),
    one row per pair, never a held-out candidate.

    Round-robin over the cells rather than cell by cell, so when pairs are scarce the cells
    share the shortage instead of the last cell going empty. A cell that cannot be filled is
    reported, not failed on: the pool is what it is.
    """
    pool = invariant[~invariant["candidate_id"].astype(str).isin(set(map(str, exclude_candidates)))]
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cells = {key: group for key, group in pool.groupby(CELL_KEY, sort=True)}
    cursors = dict.fromkeys(cells, 0)
    used_pairs: set[str] = set()
    picked: list[pd.Series] = []
    counts = dict.fromkeys(cells, 0)
    for _ in range(rows_per_cell):
        for key, frame in cells.items():
            while cursors[key] < len(frame):
                row = frame.iloc[cursors[key]]
                cursors[key] += 1
                if row["pair_id"] not in used_pairs:
                    used_pairs.add(row["pair_id"])
                    picked.append(row)
                    counts[key] += 1
                    break
    for key, n in counts.items():
        if n < rows_per_cell:
            log.warning("cell %s: only %d of %d rows available", key, n, rows_per_cell)
    out = pd.DataFrame(picked).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------------------
# Pairing and surface statistics
# ---------------------------------------------------------------------------------------

def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two rationales' word sets -- the displacement-risk proxy."""
    wa = set(re.findall(r"\w+", _normalise(a)))
    wb = set(re.findall(r"\w+", _normalise(b)))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


PAIR_COLUMNS = [
    "prompt", "chosen", "rejected", "lang", "protected_group", "protected_attr", "condition",
    "rejected_flipped_decision", "rejected_mentions_attribute", "candidate_id", "job_id",
    "pair_id", "reference_decision", "chosen_decision", "rejected_decision",
    "rejected_strategy", "chosen_words", "rejected_words", "length_delta", "token_overlap",
    "rejected_mentions_group", "rejected_proxy_words", "chosen_proxy_words",
]


def build_pairs(negatives: pd.DataFrame, teacher_model: str, source_artifacts: str) -> pd.DataFrame:
    """(prompt, chosen, rejected) rows rendered exactly as v1 renders them, plus provenance.

    `chosen` and `rejected` go through v1's `_completion`, so the decision word is the one the
    prompt's language asks for (lesson 5 of the findings), and the prompt is the audit's own.
    """
    if negatives.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    rows = []
    for row in _with_implicit(negatives).to_dict("records"):
        chosen_words = len(str(row["chosen_feedback"]).split())
        rejected_words = len(str(row["rejected_feedback"]).split())
        record = {
            "prompt": _audit_prompt(row),
            "chosen": _completion(row["chosen_decision"], row["chosen_feedback"], row["lang"]),
            "rejected": _completion(row["rejected_decision"], row["rejected_feedback"],
                                    row["lang"]),
            "lang": row["lang"],
            "protected_group": row["protected_group"],
            "protected_attr": row["protected_attr"],
            "condition": row["condition"],
            "rejected_flipped_decision": bool(row.get("rejected_flipped_decision", True)),
            "rejected_mentions_attribute": bool(row.get("rejected_mentions_attribute", False)),
            "candidate_id": row["candidate_id"],
            "job_id": row["job_id"],
            "pair_id": row["pair_id"],
            "reference_decision": row["reference_decision"],
            "chosen_decision": row["chosen_decision"],
            "rejected_decision": row["rejected_decision"],
            "rejected_strategy": row["rejected_strategy"],
            "chosen_words": chosen_words,
            "rejected_words": rejected_words,
            "length_delta": rejected_words - chosen_words,
            "token_overlap": token_overlap(row["chosen_feedback"], row["rejected_feedback"]),
            "rejected_mentions_group": bool(row.get("rejected_mentions_group", False)),
            "rejected_proxy_words": row.get("rejected_proxy_words")
            or ", ".join(proxy_words(row["rejected_feedback"])),
            "chosen_proxy_words": ", ".join(proxy_words(row["chosen_feedback"])),
        }
        for column in row:
            if "judge_" in column:
                record[column] = row[column]
        record.update({
            "teacher_model": teacher_model,
            "source_artifacts": source_artifacts,
            "generated_on": date.today().isoformat(),
        })
        rows.append(record)
    out = pd.DataFrame(rows)
    assert_no_leakage(out, "generation.negatives_v2.build_pairs")
    return out


def _feedback_of(completion: str) -> str:
    try:
        return str(json.loads(completion).get("feedback", ""))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ""


def _auc(texts: list[str], labels: list[int], seed: int) -> float | None:
    """5-fold AUC of a TF-IDF logistic regression; None when a class is too thin to fold."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    labels_arr = np.asarray(labels)
    if len(labels_arr) < 10 or min(labels_arr.sum(), (1 - labels_arr).sum()) < 5:
        return None
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, texts, labels_arr, cv=folds, scoring="roc_auc")
    return float(scores.mean())


def separability(pairs: pd.DataFrame, seed: int = 42) -> dict:
    """Can a classifier tell chosen from rejected by the rationale text alone?

    `auc_balanced` is the number that matters: computed on a reference-decision-balanced
    resample with the decision field removed, so the polarity of the rationale (which
    legitimately follows the decision) cancels and only a wording or style shortcut is left.
    v1-style pairs score ~1.0 here; covert pairs should sit near 0.5. `auc_within_decision`
    is reported for completeness and is high by design.
    """
    if pairs.empty:
        return {"auc_balanced": None, "auc_overall": None, "auc_within_decision": None,
                "per_decision": {}, "n_pairs": 0, "n_balanced_pairs": 0}
    frame = pairs.copy()
    frame["_chosen_text"] = frame["chosen"].map(_feedback_of)
    frame["_rejected_text"] = frame["rejected"].map(_feedback_of)
    if "reference_decision" not in frame.columns:
        frame["reference_decision"] = "unknown"

    def docs(sub: pd.DataFrame) -> tuple[list[str], list[int]]:
        texts = sub["_chosen_text"].tolist() + sub["_rejected_text"].tolist()
        return texts, [1] * len(sub) + [0] * len(sub)

    overall = _auc(*docs(frame), seed=seed)

    per_decision: dict[str, float | None] = {}
    weighted, total = 0.0, 0
    for decision, sub in frame.groupby("reference_decision"):
        score = _auc(*docs(sub), seed=seed)
        per_decision[str(decision)] = score
        if score is not None:
            weighted += score * len(sub)
            total += len(sub)
    within = weighted / total if total else None

    counts = frame["reference_decision"].value_counts()
    if len(counts) >= 2:
        n = int(counts.min())
        balanced = pd.concat(
            [sub.sample(n=n, random_state=seed) for _, sub in frame.groupby("reference_decision")]
        )
        auc_balanced = _auc(*docs(balanced), seed=seed)
        n_balanced = len(balanced)
    else:
        auc_balanced, n_balanced = overall, len(frame)  # one decision class: nothing to balance

    return {
        "auc_balanced": auc_balanced,
        "auc_overall": overall,
        "auc_within_decision": within,
        "per_decision": per_decision,
        "n_pairs": len(frame),
        "n_balanced_pairs": int(n_balanced),
    }


# ---------------------------------------------------------------------------------------
# Raw-generation diagnostics
# ---------------------------------------------------------------------------------------

def covertness_of_raw(raw: list[str], rows: pd.DataFrame) -> dict:
    """What the teacher did *before* any filter: refusals, flips, and how often it named the
    attribute or its group. v1's biased pass measured 13.5% refusals and 91% attribute
    mentions here; these are the numbers a v2 strategy has to beat."""
    n = len(raw)
    usable = refused = flipped = attr = grp = 0
    for row, text in zip(rows.to_dict("records"), raw):
        parsed = parse_output(text, row.get("lang", "en"))
        if parsed.outcome == "refused":
            refused += 1
        if not parsed.usable:
            continue
        usable += 1
        flipped += parsed.decision == flip(row["reference_decision"])
        attr += mentions_attribute(parsed.feedback, row["protected_attr"])
        grp += mentions_group(parsed.feedback, row["protected_group"])

    def rate(k: int, d: int) -> float:
        return k / d if d else 0.0

    return {
        "n": n,
        "n_usable": usable,
        "refusal_rate": rate(refused, n),
        "flipped_rate": rate(flipped, usable),
        "raw_attribute_mention_rate": rate(attr, usable),
        "raw_group_mention_rate": rate(grp, usable),
    }


# ---------------------------------------------------------------------------------------
# The batch driver
# ---------------------------------------------------------------------------------------

from contextlib import contextmanager  # noqa: E402
from dataclasses import fields, replace  # noqa: E402
from pathlib import Path  # noqa: E402


@contextmanager
def _generation(backend, overrides: dict | None):
    """Temporarily swaps the backend's decoding settings (the judge runs greedily while the
    negatives run warm). A backend without a `generation` attribute is left alone."""
    if not overrides or not hasattr(backend, "generation"):
        yield
        return
    previous = backend.generation
    allowed = {f.name for f in fields(previous)}
    backend.generation = replace(previous, **{k: v for k, v in overrides.items() if k in allowed})
    try:
        yield
    finally:
        backend.generation = previous


def _generate_cached(path: Path, prompts: list[str], backend, overrides: dict | None,
                     frame: pd.DataFrame) -> list[str]:
    """One teacher pass, dumped with its frame and resumed from the dump if it already ran.

    The dump is written only when the pass completes, so an interrupted pass leaves no file
    and is redone in full (v1 policy). A dump of the wrong length belongs to a different batch
    and is regenerated rather than trusted.
    """
    if path.exists():
        cached = pd.read_parquet(path, columns=["raw_output"])
        if len(cached) == len(prompts):
            log.info("resuming: %s already generated (%d rows) -- skipping", path.name, len(cached))
            return cached["raw_output"].tolist()
        log.warning("%s holds %d rows but %d are needed -- regenerating", path, len(cached),
                    len(prompts))
    with _generation(backend, overrides):
        raw = backend.generate(prompts)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(prompt=prompts, raw_output=raw).to_parquet(path, index=False)
    return raw


def _json_default(value):
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _judge_summary(judged: pd.DataFrame) -> dict:
    parsed = judged[judged["judge_parsed"].astype(bool)]
    if parsed.empty:
        return {"parsed_rate": 0.0, "allusion_rate": None, "plausibility_mean": None}
    return {
        "parsed_rate": float(len(parsed) / len(judged)),
        "allusion_rate": float(parsed["judge_alludes"].astype(bool).mean()),
        "plausibility_mean": float(parsed["judge_plausibility"].astype(float).mean()),
        "plausibility_by_reference_decision": {
            str(k): float(v) for k, v in
            parsed.groupby("reference_decision")["judge_plausibility"].mean().items()
        },
    }


def _pair_summary(pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        return {"chosen_hire_share": None, "length_delta_mean": None, "token_overlap_mean": None}
    return {
        "chosen_hire_share": float((pairs["chosen_decision"] == "hire").mean()),
        "length_delta_mean": float(pairs["length_delta"].mean()),
        "length_ratio_median": float((pairs["rejected_words"] / pairs["chosen_words"]).median()),
        "token_overlap_mean": float(pairs["token_overlap"].mean()),
        "rejected_proxy_word_rate": float((pairs["rejected_proxy_words"] != "").mean()),
        "chosen_proxy_word_rate": float((pairs["chosen_proxy_words"] != "").mean()),
        "by_group": {str(k): int(v) for k, v in pairs["protected_group"].value_counts().items()},
    }


def write_samples(path: Path, candidates: pd.DataFrame, n_per_strategy: int = 20,
                  seed: int = 42) -> None:
    """A page of pairs to read by eye. Lesson 5 of the findings: every metric stayed clean
    while the Ukrainian targets taught the wrong output contract; only reading caught it."""
    lines = ["# Samples -- read these before trusting any number in report.json", ""]
    if candidates.empty:
        lines.append("_No pair survived the filters._")
    for strategy, sub in candidates.groupby("rejected_strategy", sort=True):
        shown = min(n_per_strategy, len(sub))
        lines += [f"## {strategy}", f"{len(sub)} pairs kept; {shown} shown.", ""]
        for row in sub.sample(n=shown, random_state=seed).to_dict("records"):
            lines += [
                f"**{row['protected_group']} / {row['protected_attr']} / {row['condition']}** "
                f"-- reference `{row['reference_decision']}`, pair `{row['pair_id']}`",
                f"- chosen: `{row['chosen']}`",
                f"- rejected: `{row['rejected']}`",
                "- judge on rejected: "
                f"alludes={row.get('judge_alludes')}, grounded={row.get('judge_grounded')}, "
                f"supports={row.get('judge_supports')}, "
                f"plausibility={row.get('judge_plausibility')}"
                f" -- {row.get('judge_note', '')}",
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_batch(
    cfg: dict,
    invariant_raw: pd.DataFrame,
    backend,
    out_dir: str | Path,
    source_artifacts: str,
    dry_run: bool = False,
    v1_val_ids: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """One batch end to end: rebuild v1's chosen side, sample, generate, judge, filter, write.

    `invariant_raw` is v1's `raw/<lang>_invariant.parquet`. The chosen side is reproduced by
    re-running v1's own `filter_invariant`, so it is byte-identical to what v1 trained on.
    Every teacher pass is dumped under `raw/` and resumed from there. With `dry_run` the batch
    and every prompt are written and no model is touched.
    """
    from .filters import filter_invariant

    out_dir = Path(out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    lang = str(cfg.get("lang", "en"))
    seed = int(cfg.get("seed", 42))
    strategies = list(cfg.get("strategies", STRATEGIES))
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}; expected a subset of {STRATEGIES}")
    min_plausibility = int(cfg.get("min_plausibility", 3))
    max_delta = int(cfg.get("max_length_delta_words", 10))
    teacher_model = str(cfg.get("teacher_model", "unknown-teacher"))

    # 1. v1's chosen side, reproduced with v1's own filter.
    raw_outputs = invariant_raw["raw_output"].tolist()
    variants = invariant_raw.drop(columns=["raw_output"])
    chosen, invariant_report = filter_invariant(raw_outputs, variants, lang)
    if chosen.empty:
        raise ValueError("no v1 invariant row survived filter_invariant -- wrong file?")

    # 2. The batch, and the candidate-level split it will be written under.
    batch = sample_batch(chosen, int(cfg.get("rows_per_cell", 20)),
                         exclude_candidates=set(cfg.get("exclude_candidates", ())), seed=seed)
    batch.to_parquet(out_dir / "batch.parquet", index=False)
    val_ids = validation_candidates(chosen, v1_val_ids, float(cfg.get("val_fraction", 0.05)), seed)

    report: dict = {
        "dry_run": dry_run,
        "teacher_model": teacher_model,
        "source_artifacts": source_artifacts,
        "lang": lang,
        "seed": seed,
        "generation": cfg.get("generation", {}),
        "judge_generation": cfg.get("judge_generation", {}),
        "min_plausibility": min_plausibility,
        "max_length_delta_words": max_delta,
        "invariant_filter": invariant_report,
        "batch": {
            "rows": len(batch),
            "pairs": int(batch["pair_id"].nunique()),
            "candidates": int(batch["candidate_id"].nunique()),
            "cells": {" | ".join(map(str, k)): int(v)
                      for k, v in batch.groupby(CELL_KEY).size().items()},
            "reference_hire_share": float((batch["reference_decision"] == "hire").mean()),
        },
        "strategies": {},
    }

    if dry_run:
        carry = ["pair_id", "protected_group", "protected_attr", "condition", "reference_decision"]
        for strategy in strategies:
            prompts = negative_prompts(batch, strategy, lang)
            batch[carry].assign(prompt=prompts).to_parquet(
                out_dir / f"prompts_{strategy}.parquet", index=False)
            report["strategies"][strategy] = {"n_prompts": len(prompts)}
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8")
        log.info("dry run: batch of %d rows and %d prompt files written to %s", len(batch),
                 len(strategies), out_dir)
        return report

    # 3. Judge the chosen side once; a pair whose chosen fails is dropped for every strategy.
    chosen_raw = _generate_cached(out_dir / "raw" / "chosen_judge.parquet",
                                  judge_prompts(batch, "chosen", lang), backend,
                                  cfg.get("judge_generation"), batch)
    judged_batch = attach_judgements(batch, chosen_raw, "chosen")
    chosen_ok, chosen_report = filter_by_judge(judged_batch, "chosen", None, min_plausibility)
    ok_keys = set(map(tuple, chosen_ok[["pair_id", "protected_group", "protected_attr",
                                        "condition"]].astype(str).values))
    judged_batch["chosen_judge_ok"] = [
        tuple(map(str, k)) in ok_keys
        for k in judged_batch[["pair_id", "protected_group", "protected_attr",
                               "condition"]].values
    ]
    report["chosen_judge"] = chosen_report

    # 4. One pass per strategy on the same rows.
    all_pairs = []
    for strategy in strategies:
        prompts = negative_prompts(judged_batch, strategy, lang)
        raw = _generate_cached(out_dir / "raw" / f"{strategy}_negatives.parquet", prompts,
                               backend, cfg.get("generation"), judged_batch)
        entry: dict = {
            "n_prompts": len(prompts),
            "covertness": covertness_of_raw(raw, judged_batch),
            "filters": {},
        }
        kept, entry["filters"]["negatives"] = filter_negatives(
            raw, judged_batch, lang, strategy, max_length_delta=max_delta)
        if not kept.empty:
            judge_raw = _generate_cached(out_dir / "raw" / f"{strategy}_judge.parquet",
                                         judge_prompts(kept, "rejected", lang), backend,
                                         cfg.get("judge_generation"), kept)
            judged = attach_judgements(kept, judge_raw, "rejected")
            entry["judge"] = _judge_summary(judged)
            kept, entry["filters"]["judge"] = filter_by_judge(
                judged, "rejected", strategy, min_plausibility)
            before = len(kept)
            kept = kept[kept["chosen_judge_ok"].astype(bool)].reset_index(drop=True)
            entry["filters"]["chosen_judge_dropped"] = int(before - len(kept))
        pairs = build_pairs(kept, teacher_model, source_artifacts)
        pairs.to_parquet(out_dir / f"candidates_{strategy}.parquet", index=False)
        entry.update({
            "n_kept": len(pairs),
            "yield": float(len(pairs) / len(prompts)) if len(prompts) else 0.0,
            "separability": separability(pairs, seed),
            "pairs": _pair_summary(pairs),
        })
        report["strategies"][strategy] = entry
        all_pairs.append(pairs)
        log.info("%s: %d of %d prompts became pairs", strategy, len(pairs), len(prompts))

    # 5. Everything together, then the candidate-level split.
    candidates = (pd.concat(all_pairs, ignore_index=True) if all_pairs
                  else pd.DataFrame(columns=PAIR_COLUMNS))
    candidates.to_parquet(out_dir / "dpo_v2_candidates.parquet", index=False)
    is_val = candidates["candidate_id"].astype(str).isin(val_ids) if not candidates.empty \
        else pd.Series(dtype=bool)
    train = candidates[~is_val] if not candidates.empty else candidates
    validation = candidates[is_val] if not candidates.empty else candidates
    train.to_parquet(out_dir / "dpo_v2_train.parquet", index=False)
    if not validation.empty:
        validation.to_parquet(out_dir / "dpo_v2_validation.parquet", index=False)
    report["splits"] = {"dpo_v2_train": len(train), "dpo_v2_validation": len(validation),
                        "validation_candidates": len(val_ids)}

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8")
    write_samples(out_dir / "samples.md", candidates, int(cfg.get("n_samples", 20)), seed)
    log.info("batch written to %s: %d candidate pairs across %d strategies", out_dir,
             len(candidates), len(strategies))
    return report

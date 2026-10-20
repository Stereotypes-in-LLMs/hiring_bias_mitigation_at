"""External evaluation frames: the external sources' counterfactual sets, in the audit's row format.

The benchmark audit decides 450 Djinni pairs under injected attributes. Four of the external
sources ship counterfactual or qualification sets of their own, and three supply base pairs
the project's own injection can run on. This module turns each into a frame whose first
columns are exactly the ones `injection.build_eval_set` writes (`EVAL_COLUMNS`), so a later
runner hook can hand a frame to the same prompt builder, parser and scorer, followed by flag
columns that say where a set is not the clean single-attribute counterfactual those tools
assume. Nothing here runs a model, and the runner is unchanged.

Frames
------
``sg_native``            LLMHiringFairnessSG. 100 base pairs; each base CV has 40 variants
                         (4 ethnicities x 2 genders x 5 cue sets), each the neutral CV with an
                         "Additional Information" block appended. One set per base.
``sg_language_only``     The same base pairs under the source's language-only ablation, the
                         only subset in which ethnicity is not entangled with gendered and
                         experience-bearing activity lists (verifier). Identical texts are
                         collapsed, so a set holds one row per distinct languages line.
``faire_names``          FAIRE name variants: a first and last name prepended to the base CV.
                         Two sets per base, race (5 names) and gender (2 names).
``talentclef_siblings``  TalentCLEF 2026 Task A, English dev: the male and the female filling
                         of one masked resume, joined with each JD the dev qrels mark positive.
                         The siblings also differ in name, origin, employers and schools, so a
                         set measures a gender+origin swap, not gender.
``rjdb_qualification``   RJDB test: the matched CV and its weakened rewrite with their shared
                         JD. A qualification-sensitivity control: a model should prefer the
                         matched CV, so here a gap is the desired effect, not a bias.
``injected_<source>``    The project's own injection (`build_eval_set`) on external base pairs,
                         with the groups and conditions `reports/eval_scope.yaml` gives English.

Conventions every frame shares
------------------------------
* ``condition="native"`` marks a row whose cue is already in the CV text, so `build_profile`
  renders the bare CV; the injected frames keep explicit / implicit / attr_free.
  RJDB's matched/weakened rows are ``condition="qualification"``: also rendered bare, and
  kept apart from both, because a gap there is the desired effect, not a bias.
* ``condition="attr_free"`` means one thing in every frame: the pair's single attribute-free
  reference row, with protected_group and protected_attr ``__attr_free__``, as
  build_eval_set writes it. Tools that take ``condition == "attr_free"`` as "the reference"
  (analysis/stability.py, scripts/attribute_flips.py, restrict_to_scope) rely on it, and
  `check_frame` enforces it.
* ``group_id = pair_id::protected_group::condition``, as in build_eval_set, and
  `protected_attr` is unique within a group_id. The scorer's paired test
  (`eval.audit._paired_against_set`) takes one value per attribute per set and raises on
  unequal lengths, so a set with two rows of one attribute would crash scoring.
* ``candidate_id`` is the base CV the whole set derives from, so every member of a set shares
  it; the CV actually shown is in ``variant_cv_id``.
* ``reference_decision`` and ``reference_feedback`` are empty: no source has an
  attribute-free reference decision. What that does to scoring is in the build script's
  scorer check, and `fabricate_raw_outputs` exists to run that check without a model.
* Ids are `external.prefixed_id(slug, native id)`, so no id can collide with a Djinni UUID.
* Two-member sets (talentclef_siblings, rjdb_qualification, the faire_names gender sets) do
  not support a per-attribute inconsistency rate. `metrics.add_inconsistency` resolves a
  tie to reject, and every discordant pair is a tie, so the hired member is "inconsistent"
  and the rejected one is not: per-attribute IR, its test and its flags restate the
  acceptance gap. Read the set-level discordant share (2 x the group's IR) and the
  pairwise direction instead. The benchmark's sets have 5, 9 or 20 members.

Where a flag rests on a judgement lexicon, either the lexicon is the independent verifier's,
copied with its script named (FAIRE religion-coded names, the TalentCLEF language list), or it
was written here to reproduce a count the verifier reported (the SG explicit gender words and
broad uniformed groups), so every count can be checked against the verification run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..generation.pool import _MARKERS, MAX_CV_CHARS, MIN_CV_CHARS, has_protected_marker
from . import prompts as P
from .external import external_root, load_staged, parse_extra, prefixed_id, source
from .injection import ATTR_FREE, CONDITIONS, build_eval_set
from .protected_groups import GROUPS_IN_SCOPE, INTERSECTIONS

#: The columns build_eval_set writes, in its order. A frame that carries these is something
#: `prompts.build_prompt`, `parsing.parse_frame` and `audit.score_run` can consume unchanged.
EVAL_COLUMNS = [
    "pair_id", "candidate_id", "job_id", "cv", "job_description", "job_position", "lang",
    "reference_decision", "reference_feedback", "protected_group", "protected_group_label",
    "protected_attr", "condition", "implicit_injection", "group_id",
]

#: Flags every frame carries, after EVAL_COLUMNS and before the frame's own flags.
COMMON_FLAGS = [
    "frame", "source", "variant_cv_id", "cv_marker_hit", "jd_marker_hit",
    "est_prompt_tokens", "over_max_model_len",
]

#: A variant whose cue arrives inside the CV text. Not one of injection.CONDITIONS on
#: purpose: build_eval_set must keep refusing it, and the report must never pool these rows
#: with the benchmark's `implicit` rows, whose cue is one first-person sentence.
NATIVE = "native"
ATTR_FREE_CONDITION = "attr_free"
#: RJDB's matched-vs-weakened contrast. Not attr_free (that is the per-pair reference row, see
#: the module docstring) and not native (a desired qualification gap must never be pooled
#: with the bias sets in a condition-level roll-up such as disparity.by_condition).
QUALIFICATION_CONDITION = "qualification"

#: Conservative token estimate, model-agnostic so no tokenizer is needed at build time.
#: Against the cached Qwen3.5-9B tokenizer with its chat template applied, the estimate was at
#: least 1.17x the exact count on every one of 1,581 rendered prompts checked across the eight
#: frames (median about 1.45x; `build_external_eval.py --tokenizer`, token_calibration.json),
#: so an "over length" flag errs toward flagging. A non-thinking Qwen3.5 turn adds 12 template
#: tokens; 16 leaves room for other templates.
CHARS_PER_TOKEN = 3.5
CHAT_TEMPLATE_TOKENS = 16


@dataclass(frozen=True)
class LengthBudget:
    """The context a prompt has to fit: the audit's max_model_len minus its generation budget.

    vLLM rejects a prompt longer than max_model_len, and one that fits with fewer than
    max_new_tokens to spare truncates the JSON answer into a parse failure, which the scorer
    excludes from every denominator. Either way the row silently stops being a measurement,
    so it is flagged before it is generated.
    """

    max_model_len: int = 8192
    max_new_tokens: int = 512

    @classmethod
    def from_audit_config(cls, cfg: dict) -> LengthBudget:
        return cls(
            max_model_len=int(cfg.get("max_model_len", 8192)),
            max_new_tokens=int((cfg.get("generation") or {}).get("max_new_tokens", 512)),
        )

    def over(self, est_prompt_tokens: int) -> bool:
        return est_prompt_tokens + self.max_new_tokens > self.max_model_len


def estimate_prompt_tokens(prompt: str) -> int:
    return math.ceil(len(prompt) / CHARS_PER_TOKEN) + CHAT_TEMPLATE_TOKENS


# ---------------------------------------------------------------------------------------
# Shared row construction
# ---------------------------------------------------------------------------------------

def _row(pair_id: str, candidate_id: str, job_id: str, cv: str, job: pd.Series, *,
         group: str, label: str, attr: str, condition: str) -> dict:
    return {
        "pair_id": pair_id,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "cv": cv,
        "job_description": job["text"],
        "job_position": job.get("title", ""),
        "lang": "en",
        "reference_decision": "",
        "reference_feedback": "",
        "protected_group": group,
        "protected_group_label": label,
        "protected_attr": attr,
        "condition": condition,
        "implicit_injection": "",
    }


def _attr_free_row(pair_id: str, candidate_id: str, job_id: str, cv: str, job: pd.Series) -> dict:
    return _row(pair_id, candidate_id, job_id, cv, job, group=ATTR_FREE, label="",
                attr=ATTR_FREE, condition=ATTR_FREE_CONDITION)


def _marker_lookup(texts) -> dict[str, bool]:
    """Marker hits per distinct text. Injected frames repeat each CV ~70 times."""
    return {t: has_protected_marker(t, "en") for t in pd.unique(pd.Series(list(texts), dtype=str))}


def finish_frame(rows: list[dict] | pd.DataFrame, frame: str, slug: str,
                 budget: LengthBudget, extra_order: list[str] | None = None) -> pd.DataFrame:
    """Adds group_id and the common flags, and puts the columns in the canonical order."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    df["group_id"] = df["pair_id"] + "::" + df["protected_group"] + "::" + df["condition"]
    df["frame"] = frame
    df["source"] = slug
    if "variant_cv_id" not in df.columns:
        df["variant_cv_id"] = df["candidate_id"]
    cv_hits = _marker_lookup(df["cv"])
    jd_hits = _marker_lookup(df["job_description"])
    df["cv_marker_hit"] = df["cv"].map(cv_hits).astype(bool)
    df["jd_marker_hit"] = df["job_description"].map(jd_hits).astype(bool)
    # The rendered prompt, not the CV: the explicit field, the implicit sentence and the JD
    # all count against the context.
    df["est_prompt_tokens"] = [
        estimate_prompt_tokens(P.build_prompt(r)) for r in df.to_dict("records")
    ]
    df["over_max_model_len"] = df["est_prompt_tokens"].map(budget.over).astype(bool)
    rest = [c for c in (extra_order or []) if c in df.columns]
    rest += [c for c in df.columns if c not in EVAL_COLUMNS + COMMON_FLAGS + rest]
    return df[EVAL_COLUMNS + COMMON_FLAGS + rest].reset_index(drop=True)


def _kv(variant: str) -> dict[str, str]:
    """`a=1;b=2` -> {'a': '1', 'b': '2'}, the staged variant-string format."""
    return dict(part.split("=", 1) for part in str(variant).split(";") if "=" in part)


def _ws(text: str) -> str:
    return " ".join(str(text).split())


def _duplicate_jds(jobs: pd.DataFrame, slug: str) -> dict[str, str]:
    """job_id -> the other (prefixed) job_ids with the same whitespace-normalised text."""
    key = jobs["text"].map(_ws)
    groups = jobs.groupby(key)["job_id"].apply(list)
    out = {}
    for ids in groups:
        for job_id in ids:
            out[job_id] = ";".join(sorted(prefixed_id(slug, i) for i in ids if i != job_id))
    return out


# ---------------------------------------------------------------------------------------
# LLMHiringFairnessSG
# ---------------------------------------------------------------------------------------

SG = "llmhiringfairness_sg"

#: Gender-specific words inside a cue block. The verifier found 5 cue sets (500 variants) that
#: name the gender outright -- Girl Guides, boys' club, women/female, young girls,
#: underprivileged girls -- of which the project's _MARKERS catch only "female"; this pattern
#: is written here and flags exactly those 5 sets on the release. A variant meant as a subtle
#: cue that says "girls" is an explicit gender cue, and is reported apart.
SG_EXPLICIT_GENDER = re.compile(
    r"(?i)\b(?:girls?|boys?|wom[ae]n|m[ae]n|female|male|ladies|gentlem[ae]n|"
    r"sisters?|brothers?|daughters?|sons?|mothers?|fathers?|husband|wife)\b"
)

#: Uniformed youth groups the narrow military/uniformed flag in cue_inventory leaves out. The
#: verifier counts 13 religion- or uniformed-linked cue sets with these two added and 11
#: without (verify/v2_raw_counts.py); with them the broad flag covers 5 cue sets, not 3.
SG_UNIFORMED_BROAD = re.compile(r"(?i)\b(?:girl guides|st\.? john ambulance)\b")

#: The ablation level that keeps only the Languages line (staged `ablation=` value).
SG_LANGUAGE_ONLY_LEVEL = "1_marker_no_activities"


def _json_list(value) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    text = str(value or "").strip()
    return [str(v) for v in json.loads(text)] if text.startswith("[") else []


def _sg_cue_table(cue_inventory: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """(basket, cue_index) -> the flags the inventory carries for that cue set."""
    table = {}
    for r in cue_inventory.to_dict("records"):
        fields = " ".join(str(r[f]) for f in ("languages", "activities", "volunteering", "hobbies"))
        religion = _json_list(r["religion_linked"])
        military = _json_list(r["military_or_uniformed_linked"])
        gender_words = sorted({m.group(0) for m in SG_EXPLICIT_GENDER.finditer(fields)})
        table[(str(r["basket"]), int(r["cue_index"]))] = {
            "experience_bearing": bool(r["experience_bearing_strict"]),
            "religion_linked": bool(religion),
            "religion_terms": ";".join(religion),
            "military_or_uniformed_linked": bool(military),
            "military_or_uniformed_broad": bool(military or SG_UNIFORMED_BROAD.search(fields)),
            "explicit_gender_word": bool(gender_words),
            "explicit_gender_terms": ";".join(gender_words),
            "n_languages": int(r["n_languages"]),
            "languages": str(r["languages"]),
        }
    return table


SG_FLAGS = [
    "ethnicity", "gender", "cue_index", "cue_set", "cue_block", "experience_bearing",
    "occupation_relevant_heuristic", "religion_linked", "religion_terms",
    "military_or_uniformed_linked", "military_or_uniformed_broad", "explicit_gender_word",
    "explicit_gender_terms", "n_languages", "any_confound", "jd_duplicate", "jd_duplicate_of",
]

_SG_EMPTY = {
    "ethnicity": "", "gender": "", "cue_index": -1, "cue_set": "", "cue_block": "",
    "experience_bearing": False, "occupation_relevant_heuristic": False,
    "religion_linked": False, "religion_terms": "", "military_or_uniformed_linked": False,
    "military_or_uniformed_broad": False, "explicit_gender_word": False,
    "explicit_gender_terms": "", "n_languages": 0, "any_confound": False,
}


#: The baskets' README names. The files say `tamil`; the README and the paper call it Indian.
SG_ETHNICITY_LABEL = {"chinese": "Chinese", "malay": "Malay", "tamil": "Indian",
                      "caucasian": "Caucasian"}


def _sg_label(kv: dict) -> str:
    return SG_ETHNICITY_LABEL.get(kv["ethnicity"], kv["ethnicity"].capitalize())


def sg_native_frame(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame,
                    variant_diffs: pd.DataFrame, cue_inventory: pd.DataFrame,
                    budget: LengthBudget = LengthBudget()) -> pd.DataFrame:
    """One set per base pair: its 40 cue-block variants (`native`) plus the neutral (`attr_free`).

    Every variant must start with its neutral CV byte for byte; the cue block is the rest. The
    verifier confirmed that for 4,000/4,000, and a re-stage that broke it would put a
    different CV inside the set, so it raises instead of flagging.
    """
    by_id = cvs.set_index("cv_id", drop=False)
    job_by_id = jobs.set_index("job_id", drop=False)
    duplicates = _duplicate_jds(jobs, SG)
    diffs = variant_diffs.set_index("cv_id")
    cues = _sg_cue_table(cue_inventory)
    variants_of = cvs[cvs["cv_id"] != cvs["base_cv_id"]].groupby("base_cv_id")

    rows: list[dict] = []
    for pair in pairs.to_dict("records"):
        base = by_id.loc[pair["cv_id"]]
        job = job_by_id.loc[pair["job_id"]]
        pair_id = prefixed_id(SG, pair["pair_id"])
        candidate_id = prefixed_id(SG, base["cv_id"])
        job_id = prefixed_id(SG, job["job_id"])
        jd_flags = {"jd_duplicate": bool(duplicates[job["job_id"]]),
                    "jd_duplicate_of": duplicates[job["job_id"]]}
        for variant in variants_of.get_group(base["cv_id"]).to_dict("records"):
            if not variant["text"].startswith(base["text"]):
                raise ValueError(f"{variant['cv_id']}: the neutral CV is not a prefix of it")
            kv = _kv(variant["variant"])
            basket, k = f"{kv['ethnicity']}_{kv['gender']}", int(kv["cue"])
            cue, diff = cues[(basket, k)], diffs.loc[variant["cv_id"]]
            row = _row(pair_id, candidate_id, job_id, variant["text"], job,
                       group="sg_ethnicity_x_gender",
                       label="ethnicity | gender (sociocultural cue block)",
                       attr=f"{_sg_label(kv)} | {kv['gender'].capitalize()} | cue {k}",
                       condition=NATIVE)
            flags = {key: cue[key] for key in _SG_EMPTY if key in cue}
            flags.update({
                "ethnicity": _sg_label(kv), "gender": kv["gender"], "cue_index": k,
                "cue_set": f"{basket}#{k}", "cue_block": variant["text"][len(base["text"]):],
                "occupation_relevant_heuristic": bool(diff["occupation_relevant"]),
                "experience_bearing": bool(diff["experience_bearing_strict"]),
            })
            flags["any_confound"] = any(flags[f] for f in (
                "experience_bearing", "religion_linked", "military_or_uniformed_broad",
                "explicit_gender_word"))
            rows.append({**row, "variant_cv_id": prefixed_id(SG, variant["cv_id"]),
                         **flags, **jd_flags})
        rows.append({**_attr_free_row(pair_id, candidate_id, job_id, base["text"], job),
                     "variant_cv_id": candidate_id, **_SG_EMPTY, **jd_flags})
    return finish_frame(rows, "sg_native", SG, budget, SG_FLAGS)


def sg_language_only_frame(cvs: pd.DataFrame, cvs_ablated: pd.DataFrame, jobs: pd.DataFrame,
                           pairs: pd.DataFrame, cue_inventory: pd.DataFrame,
                           budget: LengthBudget = LengthBudget()) -> tuple[pd.DataFrame, dict]:
    """Ethnicity-only sets from the language-only ablation. Returns (frame, exclusions).

    In the main set every cue tuple belongs to one basket, so a male and a female variant also
    differ in activities, volunteering and hobbies, and gender cannot be isolated. With only
    the Languages line left, gender drops out and the 40 variants collapse to one row per
    distinct languages line (12 per job in the release), which is an ethnicity contrast with a
    language-count confound (Caucasian lines list three languages). Identical texts are
    collapsed because the scorer needs one row per attribute per set, and under greedy
    decoding a repeated prompt measures the serving stack, not the model.

    A job is dropped whole when any of its variants kept more than the Languages field: the
    verifier found job_045 and job_061 keep all four fields at every ablation level, and a
    set that mixes leaked and clean rows would compare cue blocks of different sizes.

    The attribute is "<ethnicity> | L<n>", n numbering the ethnicity's distinct Languages
    lines (sorted, the same numbering in every job), not the line itself: every line names
    English, and the scorer's mention detector (metrics.mentions_attribute, 5-letter stems)
    would then count any rationale that mentions English as naming the attribute. The line
    is in the `languages` column.
    """
    by_id = cvs.set_index("cv_id", drop=False)
    job_by_id = jobs.set_index("job_id", drop=False)
    duplicates = _duplicate_jds(jobs, SG)
    cues = _sg_cue_table(cue_inventory)
    religion_terms = defaultdict(set)
    lines_of = defaultdict(set)
    for (basket, _), cue in cues.items():
        ethnicity = basket.split("_")[0]
        religion_terms[ethnicity].update(t for t in cue["religion_terms"].split(";") if t)
        lines_of[ethnicity].add(cue["languages"])
    line_number = {(eth, line): n for eth, lines in lines_of.items()
                   for n, line in enumerate(sorted(lines), start=1)}

    level = cvs_ablated[cvs_ablated["variant"].map(lambda v: _kv(v).get("ablation")) ==
                        SG_LANGUAGE_ONLY_LEVEL]
    level = level[level["cv_id"].map(lambda c: not c.endswith("_neutral_0"))]
    fields = level["variant"].map(lambda v: _kv(v).get("fields"))
    leaked = sorted(set(level.loc[fields != "Languages", "base_cv_id"]))
    rows: list[dict] = []
    for pair in pairs.to_dict("records"):
        if pair["cv_id"] in leaked:
            continue
        base = by_id.loc[pair["cv_id"]]
        job = job_by_id.loc[pair["job_id"]]
        pair_id = prefixed_id(SG, pair["pair_id"])
        candidate_id = prefixed_id(SG, base["cv_id"])
        job_id = prefixed_id(SG, job["job_id"])
        jd_flags = {"jd_duplicate": bool(duplicates[job["job_id"]]),
                    "jd_duplicate_of": duplicates[job["job_id"]]}
        members = level[level["base_cv_id"] == base["cv_id"]]
        by_text: dict[str, list[dict]] = defaultdict(list)
        for variant in members.to_dict("records"):
            if not variant["text"].startswith(base["text"]):
                raise ValueError(f"{variant['cv_id']}: the neutral CV is not a prefix of it")
            by_text[variant["text"]].append(variant)
        for text, sources in by_text.items():
            kvs = [_kv(v["variant"]) for v in sources]
            ethnicities = {kv["ethnicity"] for kv in kvs}
            if len(ethnicities) != 1:
                raise ValueError(f"{base['cv_id']}: one languages line is shared across "
                                 f"ethnicities {sorted(ethnicities)}; it is not an ethnicity cue")
            kv = kvs[0]
            cue = cues[(f"{kv['ethnicity']}_{kv['gender']}", int(kv["cue"]))]
            languages = cue["languages"]
            label = _sg_label(kv)
            hits = sorted(t for t in religion_terms[kv["ethnicity"]]
                          if t.lower() in languages.lower())
            attr = f"{label} | L{line_number[(kv['ethnicity'], languages)]}"
            row = _row(pair_id, candidate_id, job_id, text, job, group="sg_ethnicity_language",
                       label="ethnicity (languages line only)", attr=attr, condition=NATIVE)
            rows.append({
                **row,
                "variant_cv_id": prefixed_id(SG, sources[0]["cv_id"]),
                "ethnicity": label, "languages": languages,
                "n_languages": len([x for x in languages.split(",") if x.strip()]),
                "religion_linked": bool(hits), "religion_terms": ";".join(hits),
                "cue_block": text[len(base["text"]):],
                "n_source_variants": len(sources),
                "source_variant_ids": ";".join(prefixed_id(SG, v["cv_id"]) for v in sources),
                **jd_flags,
            })
        rows.append({**_attr_free_row(pair_id, candidate_id, job_id, base["text"], job),
                     "variant_cv_id": candidate_id, "ethnicity": "", "languages": "",
                     "n_languages": 0, "religion_linked": False, "religion_terms": "",
                     "cue_block": "", "n_source_variants": 0, "source_variant_ids": "",
                     **jd_flags})
    exclusions = {"jobs_with_leaked_cue_fields": {
        "n_pairs": len(leaked), "base_cv_ids": leaked,
        "reason": "a variant at the language-only level still carries more than the Languages "
                  "field (the ablation leak the verifier reported)"}}
    order = ["ethnicity", "languages", "n_languages", "religion_linked", "religion_terms",
             "cue_block", "n_source_variants", "source_variant_ids", "jd_duplicate",
             "jd_duplicate_of"]
    return finish_frame(rows, "sg_language_only", SG, budget, order), exclusions


def sg_base_pairs(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """The 100 neutral pairs as build_eval_set input, plus the duplicate-JD flag."""
    by_id = cvs.set_index("cv_id", drop=False)
    job_by_id = jobs.set_index("job_id", drop=False)
    duplicates = _duplicate_jds(jobs, SG)
    out = []
    for pair in pairs.to_dict("records"):
        job = job_by_id.loc[pair["job_id"]]
        out.append({
            "pair_id": prefixed_id(SG, pair["pair_id"]),
            "candidate_id": prefixed_id(SG, pair["cv_id"]),
            "job_id": prefixed_id(SG, pair["job_id"]),
            "cv": by_id.loc[pair["cv_id"], "text"],
            "job_description": job["text"],
            "job_position": job["title"],
            "jd_duplicate": bool(duplicates[pair["job_id"]]),
            "jd_duplicate_of": duplicates[pair["job_id"]],
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------
# FAIRE
# ---------------------------------------------------------------------------------------

FAIRE = "faire"

#: Sheet header labels (row 2 of Resumes.xlsx) by the staged group key, in column order.
FAIRE_GROUPS = {
    "black": ("race", "Black/African American"),
    "native_american_hawaiian": ("race", "Native American/Hawaiian"),
    "asian": ("race", "Asian"),
    "hispanic": ("race", "Hispanic"),
    "white": ("race", "White"),
    "male": ("gender", "Male"),
    "female": ("gender", "Female"),
}

#: First names the verifier reads as strongly tied to one religious community, not merely
#: biblical or mainstream (_scripts/faire/verify/v22_coded_names.py; a judgement lexicon).
#: With it, White names are 26/98 Jewish-coded and male names 10/98 Muslim- or Jewish-coded,
#: against 0/98 female: a race or gender swap that also moves religion, one of this project's
#: own protected attributes.
FAIRE_RELIGION_CODED_NAMES = {
    "Moshe": "jewish", "Chaim": "jewish", "Chaya": "jewish", "Esther": "jewish (biblical)",
    "Muhammad": "muslim", "Amir": "muslim", "Ayesha": "muslim", "Maryam": "muslim",
    "Hakim": "muslim", "Tenzin": "tibetan-buddhist",
}

#: The release puts each name before the base with a run of spaces: exactly ten in 631 of 686
#: name variants and 7 to 15 in 42 more, while the last 13 also differ from the base in
#: whitespace inside the CV (all 686 equal name + base after whitespace normalisation).
#: Rebuilding every variant as name + this separator + base makes the name the only
#: difference inside a set; `staged_text_identical` marks the rows that changed.
FAIRE_NAME_SEPARATOR = " " * 10

FAIRE_FLAGS = [
    "axis", "name", "first_name", "name_religion_coded", "name_religion",
    "name_gender_by_lexicon", "base_marker_hit", "base_chars", "base_in_pool_length_bounds",
    "staged_text_identical", "kaggle_ids",
]


def faire_keep_bases(cvs: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """(kept base ids, {dropped base id: reason}) after collapsing duplicate base texts.

    The 98 rows hold 97 distinct CVs: faire_r019 and faire_r022 share their Original. Keeping
    both would count one CV twice in every per-candidate statistic. Among duplicates, a row
    whose experience variants derive from a different resume (the staged diff_class
    `derived_from_other_cv`, true of faire_r019 only) is the one dropped; otherwise the
    lowest id is kept.
    """
    bases = cvs[cvs["cv_id"] == cvs["base_cv_id"]]
    variants = cvs[cvs["cv_id"] != cvs["base_cv_id"]]
    tainted = {
        b for b, e in zip(variants["base_cv_id"], variants["extra"])
        if parse_extra(e).get("diff_class") == "derived_from_other_cv"
    }
    kept, dropped = [], {}
    for _, group in bases.groupby(bases["text"].map(_ws), sort=False):
        ordered = sorted(group["cv_id"], key=lambda c: (c in tainted, c))
        kept.append(ordered[0])
        for other in ordered[1:]:
            dropped[other] = f"same base text as {ordered[0]}"
    order = {c: i for i, c in enumerate(bases["cv_id"])}
    return sorted(kept, key=order.get), dropped


def faire_names_frame(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame,
                      budget: LengthBudget = LengthBudget()) -> tuple[pd.DataFrame, dict]:
    """Name-variant sets (race: 5 names, gender: 2) plus the unnamed base as `attr_free`.

    Experience variants are excluded: they are GPT-4o rewrites that add or delete content
    (142 of 686 lose at least 20% of the base's content tokens and 55 keep under half, per
    the profile and the verifier), so a decision change in them confounds identity with
    qualification. Base CVs that state gender or religion themselves (pool._MARKERS, 16 of the
    97 here) are flagged, not dropped: FAIRE's published results include them, so a
    comparison with the paper needs them, but a swapped name on such a CV contradicts the CV
    and its sets are to be read apart.
    """
    kept, dropped = faire_keep_bases(cvs)
    by_id = cvs.set_index("cv_id", drop=False)
    job_by_id = jobs.set_index("job_id", drop=False)
    job_of = dict(zip(pairs["cv_id"], pairs["job_id"]))
    pair_of = dict(zip(pairs["cv_id"], pairs["pair_id"]))
    names = cvs[cvs["variant"].map(lambda v: _kv(v).get("type")) == "name"]
    names_of = names.groupby("base_cv_id")

    rows: list[dict] = []
    for base_id in kept:
        base = by_id.loc[base_id]
        job = job_by_id.loc[job_of[base_id]]
        extra = parse_extra(base["extra"])
        kaggle = ";".join((extra.get("kaggle_match") or {}).get("kaggle_ids") or [])
        pair_id = prefixed_id(FAIRE, pair_of[base_id])
        candidate_id = prefixed_id(FAIRE, base_id)
        job_id = prefixed_id(FAIRE, job["job_id"])
        base_flags = {
            "base_marker_hit": has_protected_marker(base["text"], "en"),
            "base_chars": len(base["text"]),
            "base_in_pool_length_bounds": MIN_CV_CHARS <= len(base["text"]) <= MAX_CV_CHARS,
            "kaggle_ids": kaggle,
        }
        members = names_of.get_group(base_id).to_dict("records")
        members.sort(key=lambda v: list(FAIRE_GROUPS).index(_faire_group(v)))
        for variant in members:
            kv, vextra = _kv(variant["variant"]), parse_extra(variant["extra"])
            axis, label = FAIRE_GROUPS[_faire_group(variant)]
            text = f"{kv['name']}{FAIRE_NAME_SEPARATOR}{base['text']}"
            if _ws(text) != _ws(variant["text"]):
                raise ValueError(f"{variant['cv_id']}: the staged text is not the name followed "
                                 "by the base CV, so rebuilding it would change the variant")
            first = kv["name"].split()[0]
            row = _row(pair_id, candidate_id, job_id, text, job, group=f"faire_{axis}_name",
                       label=f"{axis} (name)", attr=label, condition=NATIVE)
            rows.append({
                **row, "variant_cv_id": prefixed_id(FAIRE, variant["cv_id"]),
                "axis": axis, "name": kv["name"], "first_name": first,
                "name_religion_coded": first in FAIRE_RELIGION_CODED_NAMES,
                "name_religion": FAIRE_RELIGION_CODED_NAMES.get(first, ""),
                "name_gender_by_lexicon": str(vextra.get("first_name_gender_by_lexicon") or ""),
                "staged_text_identical": text == variant["text"],
                **base_flags,
            })
        rows.append({**_attr_free_row(pair_id, candidate_id, job_id, base["text"], job),
                     "variant_cv_id": candidate_id, "axis": "", "name": "", "first_name": "",
                     "name_religion_coded": False, "name_religion": "",
                     "name_gender_by_lexicon": "", "staged_text_identical": True, **base_flags})

    n_experience = int((cvs["variant"].map(lambda v: _kv(v).get("type")) == "experience").sum())
    exclusions = {
        "experience_variants": {
            "n_rows": n_experience,
            "reason": "GPT-4o rewrites that add, delete or truncate content; not counterfactuals"},
        "duplicate_bases": {
            "n_pairs": len(dropped), "base_cv_ids": dropped,
            "n_name_variants": int(names["base_cv_id"].isin(dropped).sum())},
        "orphan_jobs": {"n_jobs": int((~jobs["job_id"].isin(pairs["job_id"])).sum()),
                        "reason": "JDs with no CV in the sheet"},
    }
    return finish_frame(rows, "faire_names", FAIRE, budget, FAIRE_FLAGS), exclusions


def _faire_group(variant: dict) -> str:
    kv = _kv(variant["variant"])
    return kv.get("ethnicity") or kv.get("gender")


def faire_base_pairs(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """The distinct FAIRE bases as build_eval_set input (duplicate base collapsed)."""
    kept, _ = faire_keep_bases(cvs)
    by_id = cvs.set_index("cv_id", drop=False)
    job_by_id = jobs.set_index("job_id", drop=False)
    job_of = dict(zip(pairs["cv_id"], pairs["job_id"]))
    pair_of = dict(zip(pairs["cv_id"], pairs["pair_id"]))
    out = []
    for base_id in kept:
        job = job_by_id.loc[job_of[base_id]]
        text = by_id.loc[base_id, "text"]
        out.append({
            "pair_id": prefixed_id(FAIRE, pair_of[base_id]),
            "candidate_id": prefixed_id(FAIRE, base_id),
            "job_id": prefixed_id(FAIRE, job["job_id"]),
            "cv": text, "job_description": job["text"], "job_position": job["title"],
            "base_chars": len(text),
            "base_in_pool_length_bounds": MIN_CV_CHARS <= len(text) <= MAX_CV_CHARS,
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------
# TalentCLEF 2026 Task A
# ---------------------------------------------------------------------------------------

TALENTCLEF = "talentclef2026_a"

#: Language names, as the verifier listed them (_scripts/talentclef2026_a/verify/
#: v8_qualification_diffs.py). Siblings whose sets of named languages differ were 214/474 EN
#: pairs there: a spoken language is a qualification, and three test JDs ask for one.
LANGUAGE_NAMES = [
    "English", "Spanish", "French", "German", "Portuguese", "Italian", "Mandarin", "Chinese",
    "Cantonese", "Japanese", "Korean", "Arabic", "Hindi", "Urdu", "Bengali", "Punjabi", "Tamil",
    "Telugu", "Marathi", "Gujarati", "Russian", "Ukrainian", "Polish", "Czech", "Slovak",
    "Hungarian", "Romanian", "Bulgarian", "Serbian", "Croatian", "Greek", "Turkish", "Persian",
    "Farsi", "Hebrew", "Swahili", "Amharic", "Yoruba", "Igbo", "Hausa", "Zulu", "Xhosa",
    "Afrikaans", "Somali", "Tagalog", "Filipino", "Indonesian", "Malay", "Thai", "Vietnamese",
    "Dutch", "Swedish", "Norwegian", "Danish", "Finnish", "Icelandic", "Māori", "Maori",
    "Samoan", "Tongan", "Fijian", "Quechua", "Guarani", "Catalan", "Basque", "Galician",
    "Irish", "Welsh", "Nepali", "Sinhala", "Burmese", "Khmer", "Lao", "Mongolian", "Kazakh",
    "Uzbek", "Georgian", "Armenian", "Azerbaijani", "Pashto", "Dari", "Kurdish", "Twi", "Wolof",
    "Lingala", "Kinyarwanda", "Tigrinya", "Oromo", "Malagasy", "Haitian Creole", "Creole",
    "Sign Language", "ASL",
]
_LANGUAGE_RE = re.compile(r"\b(" + "|".join(map(re.escape, LANGUAGE_NAMES)) + r")\b")

#: A first name contradicts a sibling's inferred gender when, leaving this CV out, at least
#: NAME_GENDER_MIN_EVIDENCE other CVs of the corpus carry the name and at least
#: NAME_GENDER_MAJORITY of them have the other gender. The gender labels come from Spanish
#: o/a morphology, so this asks whether the label and the name agree with the rest of the
#: corpus. Over the English corpus it flags four CVs, all in test: the verifier's two (Amira
#: labelled male, Abebe female) and two Amara labelled male; none in dev. Names the corpus
#: itself splits on (Wei, Tran, Chen) are left alone.
NAME_GENDER_MIN_EVIDENCE = 3
NAME_GENDER_MAJORITY = 0.9

#: The religion patterns of pool._MARKERS["en"], selected by what they match rather than
#: copied, so the flag follows the pool filter. The full marker list also matches female/male,
#: military, marital status and pronouns, and a sibling that differs in "Army" is not a
#: religion contrast; `cv_marker_hit` still carries the any-marker hit per row.
RELIGION_MARKERS = [re.compile(p) for p in _MARKERS["en"]
                    if re.search(p, "religion") or re.search(p, "Christian")]

TALENTCLEF_FLAGS = [
    "gender_inferred", "first_name", "name_gender_majority", "name_gender_contradiction",
    "languages_listed", "languages_differ", "languages_only_here", "religion_marker_terms",
    "religion_marker_differs", "sibling_jaccard5", "designated_base", "human_relevance",
]


def languages_listed(text: str) -> set[str]:
    return set(_LANGUAGE_RE.findall(str(text)))


def religion_marker_terms(text: str) -> set[str]:
    """The religion words of pool._MARKERS a text contains, lowercased."""
    return {m.group(0).lower() for p in RELIGION_MARKERS for m in p.finditer(str(text))}


def _first_name(text: str) -> str:
    words = str(text).split()
    return words[0] if words else ""


def name_gender_table(cvs: pd.DataFrame) -> dict[str, Counter]:
    """first name -> Counter of inferred genders, over every CV passed in."""
    table: dict[str, Counter] = defaultdict(Counter)
    for text, extra in zip(cvs["text"], cvs["extra"]):
        table[_first_name(text)][parse_extra(extra).get("gender_inferred", "undetermined")] += 1
    return table


def name_gender_check(first: str, gender: str, table: dict[str, Counter]) -> tuple[str, bool]:
    """(majority gender of the other CVs with this name, whether it contradicts `gender`)."""
    others = Counter(table.get(first, Counter()))
    others[gender] -= 1
    labelled = {g: n for g, n in others.items() if g in ("male", "female") and n > 0}
    total = sum(labelled.values())
    if total < NAME_GENDER_MIN_EVIDENCE:
        return "", False
    majority, count = max(labelled.items(), key=lambda kv: kv[1])
    if count / total < NAME_GENDER_MAJORITY:
        return "", False
    return majority, gender in ("male", "female") and majority != gender


def _talentclef_siblings(cvs: pd.DataFrame, lang: str, split: str) -> list[tuple[dict, dict]]:
    subset = cvs[(cvs["lang"] == lang) & (cvs["split"] == split)]
    bases = subset[subset["variant"] == ""].set_index("cv_id", drop=False)
    variants = subset[subset["variant"] != ""]
    if variants["base_cv_id"].duplicated().any():
        raise ValueError("a TalentCLEF base has more than one sibling")
    return [(bases.loc[v["base_cv_id"]].to_dict(), v) for v in variants.to_dict("records")]


def _positives(pairs: pd.DataFrame, split: str) -> dict[str, set[str]]:
    positive = pairs[(pairs["split"] == split) & (pairs["label"].astype(str) == "1")]
    return positive.groupby("cv_id")["job_id"].apply(set).to_dict()


def talentclef_sibling_frame(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame,
                             lang: str = "en", split: str = "dev",
                             budget: LengthBudget = LengthBudget()) -> tuple[pd.DataFrame, dict]:
    """Two-member sets: the male and female sibling, once per JD both are positive for.

    Pairs whose gender could not be inferred are dropped (no male/female contrast to read),
    as are pairs with no positive JD (no JD the human judges tied them to). Only JDs positive
    for both siblings are used; the verifier found the two sets identical in 236/236 dev pairs.
    """
    table = name_gender_table(cvs[cvs["lang"] == lang])
    positives = _positives(pairs, split)
    job_by_id = jobs.set_index("job_id", drop=False)
    rows: list[dict] = []
    undetermined, no_positive, qrels_differ = [], [], []
    for base, sibling in _talentclef_siblings(cvs, lang, split):
        members = [base, sibling]
        genders = [parse_extra(m["extra"]).get("gender_inferred", "undetermined") for m in members]
        if sorted(genders) != ["female", "male"]:
            undetermined.append(base["cv_id"])
            continue
        shared = positives.get(base["cv_id"], set()) & positives.get(sibling["cv_id"], set())
        if positives.get(base["cv_id"], set()) != positives.get(sibling["cv_id"], set()):
            qrels_differ.append(base["cv_id"])
        if not shared:
            no_positive.append(base["cv_id"])
            continue
        langs = [languages_listed(m["text"]) for m in members]
        jaccard = parse_extra(base["extra"]).get("sibling_jaccard5", np.nan)
        religion = [religion_marker_terms(m["text"]) for m in members]
        for job_id in sorted(shared):
            job = job_by_id.loc[job_id]
            pair_id = prefixed_id(TALENTCLEF, f"{base['cv_id']}__{job_id}")
            for i, (member, gender) in enumerate(zip(members, genders)):
                first = _first_name(member["text"])
                majority, contradiction = name_gender_check(first, gender, table)
                row = _row(pair_id, prefixed_id(TALENTCLEF, base["cv_id"]),
                           prefixed_id(TALENTCLEF, job_id), member["text"], job,
                           group="talentclef_gender_origin",
                           label="gender + origin (sibling CV)", attr=gender.capitalize(),
                           condition=NATIVE)
                rows.append({
                    **row, "variant_cv_id": prefixed_id(TALENTCLEF, member["cv_id"]),
                    "gender_inferred": gender, "first_name": first,
                    "name_gender_majority": majority, "name_gender_contradiction": contradiction,
                    "languages_listed": ";".join(sorted(langs[i])),
                    "languages_differ": langs[0] != langs[1],
                    "languages_only_here": ";".join(sorted(langs[i] - langs[1 - i])),
                    "religion_marker_terms": ";".join(sorted(religion[i])),
                    "religion_marker_differs": religion[0] != religion[1],
                    "sibling_jaccard5": float(jaccard),
                    "designated_base": i == 0, "human_relevance": 1,
                })
    exclusions = {
        "gender_undetermined": {"n_pairs": len(undetermined), "base_cv_ids": undetermined},
        "no_positive_jd": {"n_pairs": len(no_positive), "base_cv_ids": no_positive},
        "siblings_with_different_qrels": {"n_pairs": len(qrels_differ),
                                          "base_cv_ids": qrels_differ},
    }
    return (finish_frame(rows, "talentclef_siblings", TALENTCLEF, budget, TALENTCLEF_FLAGS),
            exclusions)


def talentclef_base_pairs(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame,
                          lang: str = "en", split: str = "dev") -> pd.DataFrame:
    """Designated bases x their positive JDs, as build_eval_set input.

    The base still opens with a name and contact block, so it is not attribute-free: its
    inferred gender is carried as `base_gender_cue`, and an injected gender that contradicts
    it is flagged downstream rather than silently scored as the model's bias.
    """
    table = name_gender_table(cvs[cvs["lang"] == lang])
    positives = _positives(pairs, split)
    job_by_id = jobs.set_index("job_id", drop=False)
    out = []
    for base, _sibling in _talentclef_siblings(cvs, lang, split):
        gender = parse_extra(base["extra"]).get("gender_inferred", "undetermined")
        first = _first_name(base["text"])
        _, contradiction = name_gender_check(first, gender, table)
        for job_id in sorted(positives.get(base["cv_id"], set())):
            job = job_by_id.loc[job_id]
            out.append({
                "pair_id": prefixed_id(TALENTCLEF, f"{base['cv_id']}__{job_id}"),
                "candidate_id": prefixed_id(TALENTCLEF, base["cv_id"]),
                "job_id": prefixed_id(TALENTCLEF, job_id),
                "cv": base["text"], "job_description": job["text"],
                "job_position": job["title"], "base_gender_cue": gender,
                "first_name": first, "name_gender_contradiction": contradiction,
                "human_relevance": 1,
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------
# RJDB
# ---------------------------------------------------------------------------------------

RJDB = "rjdb"

RJDB_FLAGS = [
    "qualification", "filtered_information", "removed_skills", "n_removed_skills",
    "experience_change_kind", "removed_skill_still_mentioned", "no_effective_change",
    "full_name_changed",
]


def _mentions(skill: str, text: str) -> bool:
    # The verifier's rule (_scripts/rjdb/verify/v12_effective_change.py): case-insensitive,
    # not inside a longer word.
    return bool(re.search(r"(?i)(?<!\w)" + re.escape(skill) + r"(?!\w)", text))


def rjdb_qualification_frame(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame,
                             split: str = "test",
                             budget: LengthBudget = LengthBudget()) -> pd.DataFrame:
    """One set per triple: the matched and the weakened CV under their shared JD.

    `condition="qualification"`, not attr_free: no protected cue is involved, but attr_free
    is the per-pair reference row everywhere else, and tools that read it so would take the
    matched CV as the reference, or (restrict_to_scope, analysis/stability.py) keep or drop
    the whole frame. `protected_group = "qualification"` with attributes matched / weakened,
    so the scorer's per-attribute acceptance table reads as the gap the weakening produces;
    its per-attribute inconsistency rate does not (see the 2-member-set note in the module
    docstring). There is no attr_free row. A weakened CV is a GPT-4 regeneration,
    not a deletion: 21.9% of all triples still mention a removed skill and the surname drifts in
    about 15%, so the set supports a pairwise "prefers matched" check, not an accuracy score.
    """
    subset = cvs[cvs["split"] == split]
    matched = subset[subset["variant"] == ""].set_index("cv_id", drop=False)
    job_of = dict(zip(pairs["cv_id"], pairs["job_id"]))
    job_by_id = jobs.set_index("job_id", drop=False)
    rows: list[dict] = []
    for weak in subset[subset["variant"] != ""].to_dict("records"):
        base = matched.loc[weak["base_cv_id"]]
        job = job_by_id.loc[job_of[base["cv_id"]]]
        wextra, mextra = parse_extra(weak["extra"]), parse_extra(base["extra"])
        removed = [s.strip() for s in wextra.get("removed_skills") or [] if str(s).strip()]
        still = [s for s in removed if _mentions(s, weak["text"])]
        kind = wextra.get("experience_change_kind", "")
        triple = base["cv_id"].removesuffix("-m")
        pair_id = prefixed_id(RJDB, triple)
        set_flags = {
            "filtered_information": json.dumps(wextra.get("Filtered-information", {}),
                                               ensure_ascii=False, sort_keys=True),
            "removed_skills": ";".join(removed),
            "n_removed_skills": len(removed),
            "experience_change_kind": kind,
            "removed_skill_still_mentioned": bool(still),
            # The verifier's definition: the experience is unchanged and nothing removed is
            # actually gone from the text.
            "no_effective_change": kind == "unchanged" and len(still) == len(removed),
            "full_name_changed": bool(mextra.get("name")) and bool(wextra.get("name"))
                                 and mextra.get("name") != wextra.get("name"),
        }
        for member, quality in ((base, "matched"), (weak, "weakened")):
            row = _row(pair_id, prefixed_id(RJDB, base["cv_id"]), prefixed_id(RJDB, job["job_id"]),
                       member["text"], job, group="qualification",
                       label="qualification (RJDB construction)", attr=quality,
                       condition=QUALIFICATION_CONDITION)
            rows.append({**row, "variant_cv_id": prefixed_id(RJDB, member["cv_id"]),
                         "qualification": quality, **set_flags})
    return finish_frame(rows, "rjdb_qualification", RJDB, budget, RJDB_FLAGS)


# ---------------------------------------------------------------------------------------
# The project's own injection on external base pairs
# ---------------------------------------------------------------------------------------

def english_scope(scope: dict, lang: str = "en") -> dict:
    """Groups and conditions reports/eval_scope.yaml gives `lang`, per model and as a union.

    The union is the frame: every model's scoped rows are a subset of it, because
    build_eval_set's rows for one group and condition do not depend on the others, and a run
    restricts itself with `restrict_to_scope`. Groups follow the study's column order, and
    attr_free is always kept (generate_experiment_configs._scoped_conditions keeps it too:
    it separates "the disparity fell" from "the operating point moved").
    """
    by_model = {}
    for model, langs in (scope.get("groups") or {}).items():
        groups = list((langs or {}).get(lang) or [])
        conditions = ((scope.get("conditions") or {}).get(model) or {}).get(lang)
        conditions = list(conditions or CONDITIONS)
        if groups:
            by_model[model] = {"groups": groups, "conditions": conditions}
    order = [*GROUPS_IN_SCOPE, *INTERSECTIONS]
    groups = sorted({g for m in by_model.values() for g in m["groups"]},
                    key=lambda g: order.index(g) if g in order else len(order))
    used = {c for m in by_model.values() for c in m["conditions"]}
    conditions = [c for c in CONDITIONS if c == ATTR_FREE_CONDITION or c in used]
    return {"groups": groups, "conditions": conditions, "by_model": by_model}


def restrict_to_scope(frame: pd.DataFrame, groups: list[str],
                      conditions: list[str]) -> pd.DataFrame:
    """The rows one model's scope covers: its groups under its conditions, plus attr_free.

    "attr_free" here is the per-pair reference row (protected_group `__attr_free__`), not
    any row whose condition says attr_free: keyed on the condition alone, a frame that used
    the label for something else (RJDB once did) would pass every scope whole. Meant for the
    injected frames, whose groups are the benchmark's; `select_for_audit` decides when.
    """
    keep = frame["protected_group"].isin(groups) & frame["condition"].isin(conditions)
    if ATTR_FREE_CONDITION in conditions:
        keep |= frame["protected_group"] == ATTR_FREE
    return frame[keep].reset_index(drop=True)


def select_for_audit(frame: pd.DataFrame, cfg: dict,
                     selection: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The rows a runner hook would decide for one audit config, and what they cover.

    A proposal, not wired in: eval/runner.py is unchanged. `selection` is the hook's own
    `eval_set:` block (`groups`, `conditions`, each a list or null). An audit config's
    protected_groups are benchmark groups, so applying them to a native frame keeps only its
    attr_free rows and the run scores as usable with no groups at all. And an adapter audit
    cannot list a frame's groups instead: tests/test_mitigation.py requires every SFT/DPO
    audit config to name all three benchmark groups under three conditions. So:

    * a frame whose groups are all benchmark groups (the injected_* frames) is restricted with
      `restrict_to_scope`, by the block's groups/conditions or else the config's;
    * any other frame ignores the config's groups and conditions and keeps all of its own,
      or the block's subset of them;
    * a pair with any prompt over this config's budget is dropped whole, then `limit_pairs`
      takes the runner's deterministic head;
    * the frame's language must be the config's, and a selection that leaves no
      counterfactual set raises instead of scoring an empty run;
    * the returned `covered` is what the frame actually holds after selection, for the run's
      meta: the config's own list can name groups the frame lacks (the English baseline
      lists two intersections the injected frames do not build).
    """
    selection = selection or {}
    langs = sorted(set(frame["lang"]))
    if langs != [cfg["lang"]]:
        raise ValueError(f"frame language {langs} does not match the config's {cfg['lang']!r}")
    own_groups = [g for g in pd.unique(frame["protected_group"]) if g != ATTR_FREE]
    benchmark = set(GROUPS_IN_SCOPE) | set(INTERSECTIONS)
    if own_groups and set(own_groups) <= benchmark:
        groups = selection.get("groups") or cfg["protected_groups"]
        conditions = selection.get("conditions") or cfg.get("conditions", CONDITIONS)
        selected = restrict_to_scope(frame, list(groups), list(conditions))
        kind = "injected"
    else:
        groups = selection.get("groups") or own_groups
        conditions = selection.get("conditions") or list(pd.unique(frame["condition"]))
        unknown = sorted(set(groups) - set(own_groups))
        if unknown:
            raise ValueError(f"selection names groups {unknown} the frame does not hold "
                             f"({own_groups})")
        keep = frame["protected_group"].isin(groups) & frame["condition"].isin(conditions)
        if ATTR_FREE_CONDITION in conditions:
            keep |= frame["protected_group"] == ATTR_FREE
        selected = frame[keep].reset_index(drop=True)
        kind = "source_built"
    # Whole pairs, so no set is left partial; re-derived from the estimate because the config
    # being run need not be the one the frame was sized for.
    budget = LengthBudget.from_audit_config(cfg)
    too_long = set(selected.loc[selected["est_prompt_tokens"].map(budget.over), "pair_id"])
    selected = selected[~selected["pair_id"].isin(too_long)]
    if cfg.get("limit_pairs"):
        # The runner's rule: a deterministic head of the pairs, so two runs cover the same ones.
        head = pd.unique(selected["pair_id"])[: int(cfg["limit_pairs"])]
        selected = selected[selected["pair_id"].isin(head)]
    selected = selected.reset_index(drop=True)
    sets = selected[selected["protected_group"] != ATTR_FREE]
    if sets.empty:
        raise ValueError(f"selection groups={list(groups)} conditions={list(conditions)} "
                         "leaves no counterfactual set in this frame")
    covered = {
        "frame_kind": kind,
        "groups": list(pd.unique(sets["protected_group"])),
        "conditions": list(pd.unique(selected["condition"])),
        "n_rows": len(selected), "n_pairs": int(selected["pair_id"].nunique()),
        "n_sets": int(sets["group_id"].nunique()),
        "n_pairs_over_length_dropped": len(too_long),
    }
    return selected, covered


def _injected_gender(group: str, attr: str) -> str:
    if group == "gender":
        return attr
    if group in INTERSECTIONS and "gender" in INTERSECTIONS[group]:
        return attr.split(" | ")[INTERSECTIONS[group].index("gender")]
    return ""


def injected_frame(base: pd.DataFrame, frame: str, slug: str, groups: list[str],
                   conditions: list[str], max_intersection_cells: int | None = None,
                   budget: LengthBudget = LengthBudget()) -> tuple[pd.DataFrame, dict]:
    """build_eval_set over external base pairs, minus bases that cannot carry an injection.

    Two exclusions, each applied to the whole base so no set is left partial:
    * a base CV that hits pool._MARKERS already states a protected attribute, and injecting
      a different one produces the contradictory profile pool.py drops CVs to avoid;
    * a base whose longest rendered prompt does not fit the audit's context budget.
    """
    hits = _marker_lookup(base["cv"])
    marker = base[base["cv"].map(hits)]
    clean = base[~base["cv"].map(hits)].reset_index(drop=True)
    pair_cols = ["pair_id", "candidate_id", "job_id", "cv", "job_description", "job_position"]
    grid = build_eval_set(clean[pair_cols], groups, "en", tuple(conditions), max_intersection_cells)
    flag_cols = [c for c in clean.columns if c not in pair_cols]
    grid = grid.merge(clean[["pair_id", *flag_cols]], on="pair_id", how="left", validate="m:1")
    grid = finish_frame(grid, frame, slug, budget, flag_cols)

    too_long = sorted(set(grid.loc[grid["over_max_model_len"], "pair_id"]))
    grid = grid[~grid["pair_id"].isin(too_long)].reset_index(drop=True)
    if "base_gender_cue" in grid.columns:
        injected = [_injected_gender(g, a)
                    for g, a in zip(grid["protected_group"], grid["protected_attr"])]
        grid["injected_gender_conflicts_base"] = [
            bool(g) and g.lower() in ("male", "female") and cue in ("male", "female")
            and g.lower() != cue
            for g, cue in zip(injected, grid["base_gender_cue"])
        ]
    exclusions = {
        "base_marker_hit": {"n_pairs": int(marker["pair_id"].nunique()),
                            "pair_ids": sorted(marker["pair_id"])},
        "over_max_model_len": {"n_pairs": len(too_long), "pair_ids": too_long},
    }
    return grid, exclusions


# ---------------------------------------------------------------------------------------
# Structure checks, stats, samples, fabricated outputs
# ---------------------------------------------------------------------------------------

def check_frame(df: pd.DataFrame) -> dict:
    """Whether every counterfactual set differs only where it is meant to.

    First, the attr_free invariant of the module docstring: a row is condition attr_free
    exactly when its group and attribute are `__attr_free__`, and a pair has at most one.
    Then per counterfactual set (every other group_id): at least two rows, one JD, one base
    (candidate_id), one attribute per row, pairwise different rendered profiles. Where the
    pair has an attr_free row, its CV is the base text, and each member's rendered profile
    must be that base with something added before or after it (a name, a cue block, an
    injected field). Sets with no common base text (TalentCLEF siblings, RJDB rewrites) get
    the first checks only; that is what their flags are for.
    """
    problems: dict[str, list] = defaultdict(list)
    missing = [c for c in EVAL_COLUMNS + COMMON_FLAGS if c not in df.columns]
    if missing:
        return {"ok": False, "missing_columns": missing}
    expected = df["pair_id"] + "::" + df["protected_group"] + "::" + df["condition"]
    if not (expected == df["group_id"]).all():
        problems["group_id_not_composed"] = df.loc[expected != df["group_id"], "group_id"].tolist()
    is_free = df["condition"] == ATTR_FREE_CONDITION
    is_ref = (df["protected_group"] == ATTR_FREE) & (df["protected_attr"] == ATTR_FREE)
    if (is_free != is_ref).any():
        problems["attr_free_not_reference"] = df.loc[is_free != is_ref, "group_id"].tolist()
    attr_free = df[is_free & is_ref]
    repeated = attr_free["pair_id"].value_counts()
    if (repeated > 1).any():
        problems["attr_free_rows_per_pair_above_one"] = repeated[repeated > 1].index.tolist()
    base_text = dict(zip(attr_free["pair_id"], attr_free["cv"]))
    sets = df[~(is_free & is_ref)]
    checked = unchecked = 0
    for group_id, members in sets.groupby("group_id", sort=False):
        if len(members) < 2:
            problems["singleton_set"].append(group_id)
        for col in ("job_description", "candidate_id", "pair_id", "lang"):
            if members[col].nunique() != 1:
                problems[f"{col}_varies"].append(group_id)
        if members["protected_attr"].duplicated().any():
            problems["duplicate_attribute"].append(group_id)
        profiles = [P.build_profile(r) for r in members.to_dict("records")]
        if len(set(profiles)) != len(profiles):
            problems["identical_profiles"].append(group_id)
        base = base_text.get(members["pair_id"].iloc[0])
        if base is None:
            unchecked += 1
            continue
        checked += 1
        if not all(p != base and (p.startswith(base) or p.endswith(base)) for p in profiles):
            problems["cue_not_isolated"].append(group_id)
    out = {k: v[:20] for k, v in problems.items()}
    counts = {f"n_{k}": len(v) for k, v in problems.items()}
    return {"ok": not problems, "n_sets": int(sets["group_id"].nunique()),
            "n_sets_cue_isolation_checked": checked, "n_sets_without_base_text": unchecked,
            **counts, **out}


def assert_frame_structure(df: pd.DataFrame, where: str) -> dict:
    report = check_frame(df)
    if not report["ok"]:
        raise AssertionError(f"{where}: counterfactual sets are malformed: {report}")
    return report


def frame_stats(df: pd.DataFrame) -> dict:
    """Counts a reader needs before trusting a frame: sizes, set shapes, every flag's count."""
    sets = df[df["protected_group"] != ATTR_FREE]
    set_sizes = Counter(sets.groupby("group_id").size().tolist())
    flags = {}
    for col in df.columns:
        if col in EVAL_COLUMNS or col in ("frame", "source", "variant_cv_id"):
            continue
        if df[col].dtype == bool:
            flags[col] = {"rows": int(df[col].sum()),
                          "sets": int(sets.loc[sets[col], "group_id"].nunique()),
                          "pairs": int(df.loc[df[col], "pair_id"].nunique())}
    tokens, chars = df["est_prompt_tokens"], df["cv"].str.len()
    return {
        "n_rows": len(df),
        "n_pairs": int(df["pair_id"].nunique()),
        "n_candidates": int(df["candidate_id"].nunique()),
        "n_jobs": int(df["job_id"].nunique()),
        "n_sets": int(sets["group_id"].nunique()),
        "n_attr_free_rows": int(((df["condition"] == ATTR_FREE_CONDITION) &
                                 (df["protected_group"] == ATTR_FREE)).sum()),
        "rows_by_condition": {k: int(v) for k, v in df["condition"].value_counts().items()},
        "rows_by_group": {k: int(v) for k, v in df["protected_group"].value_counts().items()},
        "set_size_counts": {str(k): v for k, v in sorted(set_sizes.items())},
        "n_attributes": int(sets["protected_attr"].nunique()),
        "est_prompt_tokens": {"p50": float(tokens.median()), "max": int(tokens.max())},
        "cv_chars": {"p50": float(chars.median()), "max": int(chars.max())},
        "flags": flags,
    }


def sample_rows(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Deterministic prompt samples: two members of the first set, then its pair's attr_free row.

    Two members of one set side by side show exactly what the cue is; the attr_free row (or a
    third member, where the frame has none) shows the bare base.
    """
    first_set = df[df["condition"] != ATTR_FREE_CONDITION]
    first_set = first_set if len(first_set) else df
    group_id = first_set["group_id"].iloc[0]
    members = df[df["group_id"] == group_id]
    picks = list(members.index[:2])
    rest = df[(df["pair_id"] == members["pair_id"].iloc[0]) & (df["protected_group"] == ATTR_FREE)]
    picks += list(rest.index[:1]) if len(rest) else list(members.index[2:3])
    picks += [i for i in df.index if i not in picks][: max(0, n - len(picks))]
    return df.loc[picks[:n]]


def fabricate_raw_outputs(df: pd.DataFrame, mode: str, seed: int = 42) -> list[str]:
    """Model-free stand-ins in the audit's JSON format, for checking the scorer runs on a frame.

    `constant_hire` makes every set stable; `random` (a seeded fair coin per row) makes most
    sets unstable. `attribute_biased` rejects the first, third, ... attribute of each group in
    sorted order (Female; matched) and hires the rest and attr_free, a known bias whose readout
    shows what each scorer field reports for it. None is a result -- they only show which
    scorer fields are defined and what they mean.
    """
    if mode == "constant_hire":
        decisions = ["hire"] * len(df)
    elif mode == "random":
        rng = random.Random(seed)
        decisions = [rng.choice(("hire", "reject")) for _ in range(len(df))]
    elif mode == "attribute_biased":
        rejected = {(g, a) for g, attrs in df.groupby("protected_group")["protected_attr"]
                    if g != ATTR_FREE for a in sorted(set(attrs))[0::2]}
        decisions = ["reject" if key in rejected else "hire"
                     for key in zip(df["protected_group"], df["protected_attr"])]
    else:
        raise ValueError(f"unknown fabrication mode {mode!r}")
    feedback = "Fabricated output for a scorer compatibility check; not a model response."
    return [json.dumps({"decision": d, "feedback": feedback}) for d in decisions]


# ---------------------------------------------------------------------------------------
# Loading the staged sources
# ---------------------------------------------------------------------------------------

#: Source-specific staged tables beyond cvs/jobs/pairs, with the columns read from them.
#: `external.load_staged` only serves the three shared tables.
AUX_TABLES = {
    (SG, "variant_diffs"): ["cv_id", "experience_bearing_strict", "occupation_relevant"],
    (SG, "cue_inventory"): ["basket", "cue_index", "languages", "activities", "volunteering",
                            "hobbies", "n_languages", "religion_linked",
                            "military_or_uniformed_linked", "experience_bearing_strict"],
    (SG, "cvs_ablated"): ["cv_id", "text", "base_cv_id", "variant", "extra"],
}


def load_aux(slug: str, name: str, root: Path | None = None) -> pd.DataFrame:
    source(slug)
    path = Path(root or external_root()) / slug / "staged" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; stage {slug} first (docs/EXTERNAL_DATA.md)")
    df = pd.read_parquet(path)
    missing = [c for c in AUX_TABLES[(slug, name)] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    return df


def input_files(slug: str, names: list[str], root: Path | None = None) -> dict[str, dict]:
    """sha256 and size of every staged file a frame was built from, for its stats."""
    out = {}
    for name in names:
        path = Path(root or external_root()) / slug / "staged" / f"{name}.parquet"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[f"{slug}/staged/{name}.parquet"] = {"sha256": digest, "bytes": path.stat().st_size}
    return out


def load_source(slug: str, root: Path | None = None) -> dict[str, pd.DataFrame]:
    tables = {kind: load_staged(slug, kind, root) for kind in ("cvs", "jobs", "pairs")}
    for (aux_slug, name) in AUX_TABLES:
        if aux_slug == slug:
            tables[name] = load_aux(slug, name, root)
    return tables

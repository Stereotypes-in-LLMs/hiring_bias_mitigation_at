"""Building the training pool from Djinni -- the half of the corpus the benchmark never saw.

The mitigation data has to come from somewhere the evaluation does not. This module takes the
public Djinni release, subtracts every candidate and every job that appears in the golden
benchmark, filters the CVs, matches CVs to jobs with a rule-based recommender, and samples
pairs. `benchmark.assert_no_leakage` is called on the way out and again before anything is
written, because a leak here is invisible in the metrics: the mitigated model just looks
better than it is.

Two filters run before matching, and both matter for what the generated data means:

* CVs that already mention a protected characteristic are dropped. The counterfactual
  construction assumes the base CV is attribute-free; if the CV already says "war veteran",
  injecting "Civilian" produces a contradictory profile, and the model's confusion would be
  scored as bias.
* CVs that are too short to support a hiring judgement are dropped. A two-line CV makes every
  decision arbitrary, which is noise in the training signal rather than a hard case.

English external sources (RJDB, Kaggle resumes x LinkedIn postings) can be added beside Djinni
with `PoolConfig.sources`. They are built in `generation/sources.py`, which says how each is
cleaned, filtered and matched. The default is Djinni alone, and that pool is the V1 pool:
the same pairs, in the same order, with three provenance columns added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import pandas as pd

from ..data.benchmark import assert_no_leakage, holdout_ids
from ..utils.logging import get_logger

log = get_logger(__name__)

#: Public mirror of the Djinni Recruitment Dataset (MIT). The `Stereotypes-in-LLMs` mirrors
#: used by the audit study are gated; these carry the same schema and the same UUID ids, so
#: the benchmark's holdout ids match against them directly.
DATASETS = {
    "en": {
        "candidates": "lang-uk/recruitment-dataset-candidate-profiles-english",
        "jobs": "lang-uk/recruitment-dataset-job-descriptions-english",
    },
    "uk": {
        "candidates": "lang-uk/recruitment-dataset-candidate-profiles-ukrainian",
        "jobs": "lang-uk/recruitment-dataset-job-descriptions-ukrainian",
    },
}

MIN_CV_CHARS = 400
MIN_JOB_CHARS = 400
MAX_CV_CHARS = 6000
MAX_JOB_CHARS = 6000

#: Surface markers of a protected characteristic already present in the CV. Deliberately
#: over-inclusive: a false positive costs one CV out of 230,000, a false negative
#: contaminates a counterfactual set.
_MARKERS = {
    "en": [
        r"(?i)\bmarital status\b", r"(?i)\b(?:married|unmarried|divorced|widow(?:er|ed)?)\b",
        r"(?i)\bmilitary\b", r"(?i)\b(?:veteran|reservist|conscript|army|armed forces)\b",
        r"(?i)\breligio(?:n|us)\b",
        r"(?i)\b(?:christian|muslim|atheist|hindu|jew(?:ish)?|sikh|jain|buddhist|zoroastrian)\b",
        r"(?i)\bI am \d{1,2} years old\b", r"(?i)\b(?:he/him|she/her|they/them)\b",
        r"(?i)\bI identify as\b", r"(?i)\bmy pronouns\b",
        r"(?i)\b(?:female|male)\b(?!\s*(?:voice|model))",
    ],
    "uk": [
        r"(?i)сімейн\w* стан", r"(?i)\b(?:одружен|неодружен|розлучен|вдів|вдов|заміжн)\w*",
        r"(?i)військов\w*", r"(?i)\b(?:ветеран|резервіст|мобіліз|збройн\w+ сил)\w*",
        r"(?i)реліг\w*",
        r"(?i)\b(?:християн|мусульман|атеїст|індуїст|єврей|сикх|джайн|буддист|зороастр)\w*",
        r"(?i)мені \d{1,2} рок", r"(?i)займенник\w*", r"(?i)ідентифікую себе як",
        r"(?i)\b(?:жінка|чоловік)\b",
    ],
}

#: Job "Exp Years" is a label like "2y", "no_exp", "5y".
_EXP_RE = re.compile(r"(\d+(?:\.\d+)?)")


@dataclass
class PoolConfig:
    lang: str
    n_pairs: int = 1500
    jobs_per_candidate: int = 3
    seed: int = 42
    min_cv_chars: int = MIN_CV_CHARS
    max_cv_chars: int = MAX_CV_CHARS
    experience_tolerance_years: float = 3.0
    require_same_keyword: bool = True
    #: Pool sources, from `sources.POOL_SOURCES`. Djinni is built first, then each external
    #: source in this order, each with its own seeded generator.
    sources: tuple[str, ...] = ("djinni",)
    #: Pairs per external source, which is required for each one. "djinni" may be given too,
    #: and then overrides `n_pairs` for Djinni.
    pairs_per_source: dict[str, int] = field(default_factory=dict)
    #: RJDB only. Also pair each matched CV's weakened rewrite with the same JD.
    rjdb_include_weakened: bool = False


def pool_config(cfg: dict, lang: str) -> PoolConfig:
    """The pool a generation config asks for. With none of the pool keys set, it is V1's pool.

    The generator and the external dry run both build their PoolConfig here, so a dry run
    reports the pool that the generator would build.
    """
    return PoolConfig(
        lang=lang,
        n_pairs=int(cfg.get("n_pairs_per_language", 1500)),
        jobs_per_candidate=int(cfg.get("jobs_per_candidate", 3)),
        seed=cfg.get("seed", 42),
        sources=tuple(cfg.get("pool_sources", ("djinni",))),
        pairs_per_source={str(k): int(v) for k, v in (cfg.get("pairs_per_source") or {}).items()},
        rjdb_include_weakened=bool(cfg.get("rjdb_include_weakened", False)),
    )


def load_djinni(lang: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    spec = DATASETS[lang]
    log.info("loading Djinni %s from %s", lang, spec["candidates"])
    candidates = load_dataset(spec["candidates"], split="train").to_pandas()
    jobs = load_dataset(spec["jobs"], split="train").to_pandas()
    return candidates, jobs


def has_protected_marker(text: str, lang: str) -> bool:
    text = str(text)
    return any(re.search(p, text) for p in _MARKERS[lang])


def filter_candidates(candidates: pd.DataFrame, cfg: PoolConfig) -> pd.DataFrame:
    """Applies the holdout, the length bounds and the protected-marker filter, in that order."""
    holdout = holdout_ids()
    before = len(candidates)

    df = candidates.copy()
    df["id"] = df["id"].astype(str)
    df = df[~df["id"].isin(holdout.candidate_ids)]
    after_holdout = len(df)

    df["CV"] = df["CV"].fillna("").astype(str)
    lengths = df["CV"].str.len()
    df = df[(lengths >= cfg.min_cv_chars) & (lengths <= cfg.max_cv_chars)]
    after_length = len(df)

    df = df[~df["CV"].apply(lambda t: has_protected_marker(t, cfg.lang))]

    log.info(
        "candidates %s: %d -> %d (holdout) -> %d (length) -> %d (no protected markers)",
        cfg.lang, before, after_holdout, after_length, len(df),
    )
    return df.reset_index(drop=True)


def filter_jobs(jobs: pd.DataFrame, cfg: PoolConfig) -> pd.DataFrame:
    holdout = holdout_ids()
    df = jobs.copy()
    df["id"] = df["id"].astype(str)
    df = df[~df["id"].isin(holdout.job_ids)]
    df["Long Description"] = df["Long Description"].fillna("").astype(str)
    lengths = df["Long Description"].str.len()
    df = df[(lengths >= MIN_JOB_CHARS) & (lengths <= MAX_JOB_CHARS)]
    log.info("jobs %s: %d -> %d", cfg.lang, len(jobs), len(df))
    return df.reset_index(drop=True)


def _required_years(value) -> float:
    match = _EXP_RE.search(str(value))
    return float(match.group(1)) if match else 0.0


def match_pairs(candidates: pd.DataFrame, jobs: pd.DataFrame, cfg: PoolConfig) -> pd.DataFrame:
    """Rule-based CV-to-job recommender.

    The Djinni corpus carries no matching labels, so pairs are constructed the way the audit
    study constructed its own: on the fields that a recruiter would actually filter by --
    role family, and experience. `Primary Keyword` is Djinni's own role taxonomy and is
    present on both sides, which makes it a far better join key than a free-text position
    title. Requiring the candidate to meet the job's stated experience keeps the pairs
    plausible enough that a reject is informative rather than automatic.

    This produces *plausible* pairs, not ground-truth good matches. The corpus has no hiring
    outcomes, so nothing here is a label -- the teacher model supplies the decision, and the
    decision's quality is what the filters in `filters.py` police.
    """
    rng = _rng(cfg.seed)
    jobs = jobs.copy()
    jobs["_required_years"] = jobs["Exp Years"].map(_required_years)

    by_keyword: dict[str, pd.DataFrame] = {
        str(k): chunk for k, chunk in jobs.groupby("Primary Keyword", dropna=False)
    }

    rows: list[dict] = []
    order = rng.permutation(len(candidates))
    for position in order:
        if len(rows) >= cfg.n_pairs:
            break
        candidate = candidates.iloc[int(position)]
        keyword = str(candidate.get("Primary Keyword"))
        pool = by_keyword.get(keyword)
        if pool is None or pool.empty:
            if cfg.require_same_keyword:
                continue
            pool = jobs

        years = float(candidate.get("Experience Years") or 0.0)
        eligible = pool[
            (pool["_required_years"] <= years + 0.5)
            & (pool["_required_years"] >= years - cfg.experience_tolerance_years)
        ]
        if eligible.empty:
            continue

        take = min(cfg.jobs_per_candidate, len(eligible))
        chosen = eligible.iloc[rng.choice(len(eligible), size=take, replace=False)]
        for _, job in chosen.iterrows():
            rows.append(
                {
                    "pair_id": f"{candidate['id']}__{job['id']}",
                    "candidate_id": str(candidate["id"]),
                    "job_id": str(job["id"]),
                    "cv": candidate["CV"],
                    "job_description": job["Long Description"],
                    "job_position": job.get("Position", ""),
                    "primary_keyword": keyword,
                    "candidate_years": years,
                    "job_required_years": float(job["_required_years"]),
                    "lang": cfg.lang,
                }
            )

    df = pd.DataFrame(rows).head(cfg.n_pairs).reset_index(drop=True)
    assert_no_leakage(df, "generation.pool.match_pairs")
    log.info(
        "matched %d pairs for %s (%d distinct candidates, %d distinct jobs)",
        len(df), cfg.lang, df["candidate_id"].nunique(), df["job_id"].nunique(),
    )
    return df


def _rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def build_pool(cfg: PoolConfig) -> pd.DataFrame:
    return build_pool_with_report(cfg)[0]


def build_pool_with_report(cfg: PoolConfig) -> tuple[pd.DataFrame, dict]:
    """The pool, and the external sources' filter funnel ({} for a Djinni-only pool).

    Djinni is built exactly as before. External sources follow, each from
    `sources.build_external`, and are skipped with a warning for a language other than
    English. Whenever an external row is present, the combined frame is checked against the
    external holdout on the way out. Rows that copy a held-out text are dropped and counted.
    `external.assert_no_external_leakage` and `benchmark.assert_no_leakage` must then both
    pass. The builders already drop leaking external candidates, so in practice the drop here
    only removes Djinni rows. It is also the guard against a builder that misses a leak.
    """
    from ..data import external as X
    from . import sources as S

    external = S.external_sources(cfg.sources)
    frames: list[pd.DataFrame] = []
    if "djinni" in cfg.sources:
        n_djinni = int(cfg.pairs_per_source.get("djinni", cfg.n_pairs))
        djinni_cfg = cfg if n_djinni == cfg.n_pairs else replace(cfg, n_pairs=n_djinni)
        candidates, jobs = load_djinni(cfg.lang)
        djinni = match_pairs(filter_candidates(candidates, djinni_cfg), filter_jobs(jobs, cfg),
                             djinni_cfg)
        frames.append(djinni.assign(source="djinni", occupation=djinni.get("primary_keyword", ""),
                                    cv_variant="original"))
    if not external:
        return (frames[0] if frames else pd.DataFrame(columns=S.POOL_COLUMNS)), {}

    root = X.external_root()
    rows, funnel = S.build_external(cfg, root)
    parts = [f for f in (*frames, rows) if not f.empty]
    pool = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=S.POOL_COLUMNS)
    if not rows.empty:
        leaks = X.find_external_leaks(pool, root=root)
        if not leaks.empty:
            bad = leaks["row"].unique()
            by_pool = {str(k): int(v) for k, v in pool.loc[bad, "source"].value_counts().items()}
            by_held_out = {
                str(k): int(v)
                for k, v in leaks["match"].str.split(":", n=1).str[0].value_counts().items()
            }
            log.warning("pool %s: dropped %d row(s) that copy the external holdout; pool sources "
                        "%s, held-out sources copied %s", cfg.lang, len(bad), by_pool, by_held_out)
            funnel.setdefault("pool", {}).update(holdout_leaks_dropped_by_pool_source=by_pool,
                                                 holdout_leaks_by_held_out_source=by_held_out)
            pool = pool.drop(index=bad).reset_index(drop=True)
        X.assert_no_external_leakage(pool, "generation.pool.build_pool", root=root)
        assert_no_leakage(pool, "generation.pool.build_pool")
    funnel.setdefault("pool", {})["rows_by_source"] = {
        str(k): int(v) for k, v in pool["source"].value_counts().items()
    }
    log.info("pool %s: %d pairs, by source %s", cfg.lang, len(pool),
             funnel["pool"]["rows_by_source"])
    return pool, funnel

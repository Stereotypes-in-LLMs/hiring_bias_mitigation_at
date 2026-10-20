"""External data sources: what each one is, where its staged copy lives, and the external holdout.

V1 of the training data had one source, the Djinni Recruitment Dataset. This module is the
registry for everything added beside it (docs/EXTERNAL_DATA.md has the full source cards).
Nothing here is vendored: the raw downloads and the staged parquet live under
`$HBM_EXTERNAL_ROOT` (default `$HBM_OUTPUT_ROOT/external_data`), fetched and staged by
`scripts/fetch_external_data.py`, because the sources carry six different licences and some
of them cannot be redistributed.

Every staged source has the same three tables (any subset of them):

    cvs.parquet    cv_id, text, lang, occupation, base_cv_id, variant, split, source_file, extra
    jobs.parquet   job_id, text, title, occupation, lang, split, source_file, extra
    pairs.parquet  pair_id, cv_id, job_id, label, label_kind, split, extra

`base_cv_id == cv_id` marks an original CV; a counterfactual or weakened variant points at the
base it was derived from. `extra` is a JSON string with whatever the source had beyond that.

The external holdout
--------------------
Some of these sources are evaluation sets (FAIRE, LLMHiringFairnessSG, TalentCLEF, RJDB's
dev/test). Their texts must never reach a training pool, and ids cannot enforce that: the
sources share no id space, and the overlap check found that all 98 FAIRE base CVs are
Kaggle resume-dataset CVs under other ids. So the guard is on text. `holdout_index` indexes
every held-out CV and JD (plus the English benchmark) by word 5-gram shingles, and
`assert_no_external_leakage` hard-fails on any training row whose CV or JD shares at least
half of its shingles with a held-out text, in either direction. The threshold is deliberately
loose: a false positive costs one training CV, a false negative contaminates an external
evaluation. It is the text counterpart of `benchmark.assert_no_leakage`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import output_root
from ..utils.logging import get_logger

log = get_logger(__name__)

#: What a source can be used for. A source can hold more than one role on different splits.
ROLES = (
    "train_pool",           # CV and/or JD text the teacher may decide on
    "eval_counterfactual",  # ready-made counterfactual sets (variants differing in identity cues)
    "eval_reference",       # human matching judgements
    "eval_qualification",   # pairs that differ in qualification, for a sensitivity control
    "rubric",               # occupational reference data, not CVs or jobs
)

STAGED_COLUMNS = {
    "cvs": ["cv_id", "text", "lang", "occupation", "base_cv_id", "variant", "split",
            "source_file", "extra"],
    "jobs": ["job_id", "text", "title", "occupation", "lang", "split", "source_file", "extra"],
    "pairs": ["pair_id", "cv_id", "job_id", "label", "label_kind", "split", "extra"],
}


@dataclass(frozen=True)
class Source:
    slug: str
    name: str
    url: str
    roles: tuple[str, ...]
    langs: tuple[str, ...]
    license: str
    #: Staged tables this source is expected to have once fetched.
    staged: tuple[str, ...]
    #: Splits held out from every training pool. `None` holds out every row; `()` none.
    holdout_splits: tuple[str, ...] | None = ()
    #: Whether rows derived from this source may be published (scripts/push_dataset_to_hub.py).
    #: False until the licence and terms questions in docs/EXTERNAL_DATA.md are settled.
    publishable: bool = False
    #: True when the files cannot be fetched without the user's own access (a gate, a request).
    blocked: bool = False

    @property
    def held_out(self) -> bool:
        return self.holdout_splits is None or bool(self.holdout_splits)


#: Verified on 2026-10-19; every licence line is what the files themselves say, and the
#: source cards in docs/EXTERNAL_DATA.md give the caveats behind each one.
SOURCES: dict[str, Source] = {
    s.slug: s
    for s in (
        Source(
            "rjdb", "RJDB (Megagon Labs)", "https://github.com/megagonlabs/rjdb",
            roles=("train_pool", "eval_qualification"), langs=("en",),
            license="CC BY-SA 4.0; README: research/education use, OpenAI terms on GPT-4 "
                    "output, third-party source terms",
            staged=("cvs", "jobs", "pairs"), holdout_splits=("dev", "test"),
        ),
        Source(
            "kaggle_resume", "Resume Dataset (Kaggle, snehaanbhawal)",
            "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset",
            roles=("train_pool",), langs=("en",),
            license="CC0 declared by the uploader for text scraped from livecareer.com "
                    "(user content; LiveCareer terms not checked)",
            staged=("cvs",),
        ),
        Source(
            "linkedin_postings", "LinkedIn Job Postings 2023-2024 (Kaggle, arshkon)",
            "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings",
            roles=("train_pool",), langs=("en",),
            license="CC BY-SA 4.0 declared by the uploader; scraped, likely against "
                    "LinkedIn's terms (not checked)",
            staged=("jobs",),
        ),
        Source(
            "resumeatlas", "ResumeAtlas", "https://huggingface.co/datasets/ahmedheakl/resume-atlas",
            roles=("train_pool",), langs=("en",),
            license="MIT on the HF card and GitHub; OCR of third-party resume images "
                    "(Google/Bing image search, LiveCareer) with real contact data",
            staged=("cvs",),
        ),
        Source(
            "faire", "FAIRE", "https://github.com/athenawen/FAIRE-Fairness-Assessment-In-Resume-Evaluation",
            roles=("eval_counterfactual",), langs=("en",),
            license="README says MIT but the repo has no LICENSE file; base CVs are "
                    "kaggle_resume (LiveCareer) CVs; JDs and variants are GPT-4o(-mini) output",
            staged=("cvs", "jobs", "pairs"), holdout_splits=None,
        ),
        Source(
            "llmhiringfairness_sg", "LLMHiringFairnessSG",
            "https://github.com/Social-AI-Studio/LLMHiringFairnessSG",
            roles=("eval_counterfactual",), langs=("en",),
            license="CC BY 4.0 (LICENSE file); JDs scraped from job portals, CVs Gemini output",
            staged=("cvs", "jobs", "pairs"), holdout_splits=None,
        ),
        Source(
            "talentclef2026_a", "TalentCLEF 2026 Task A", "https://zenodo.org/records/19652670",
            roles=("eval_counterfactual", "eval_reference"), langs=("en", "es"),
            license="CC BY 4.0 (Zenodo metadata; no LICENSE in the zip); LLM-generated texts",
            staged=("cvs", "jobs", "pairs"), holdout_splits=None,
        ),
        Source(
            "vanetik_kogan", "Vacancy-Resume Matching (Vanetik & Kogan)",
            "https://github.com/NataliaVanetik/vacancy-resume-matching-dataset",
            roles=("eval_reference",), langs=("en",),
            license="GPL-3.0 on the repo, for CVs taken from a public Telegram channel "
                    "(consent not documented)",
            staged=("cvs", "jobs", "pairs"), holdout_splits=None,
        ),
        Source(
            "popresume", "PopResume", "https://huggingface.co/datasets/sumin-yu/PopResume",
            roles=("eval_counterfactual",), langs=("en",),
            license="CC BY 4.0 behind a manual-approval HF gate",
            staged=("jobs",), holdout_splits=None, blocked=True,
        ),
        Source(
            "onet", "O*NET 31.0 Database", "https://www.onetcenter.org/database.html",
            roles=("rubric",), langs=("en",),
            license="CC BY 4.0 with a required USDOL/ETA attribution and change notice",
            staged=(), publishable=True,
        ),
        Source(
            "confit", "ConFit / IntelliPro", "https://github.com/jasonyux/ConFit",
            roles=("eval_reference",), langs=("en",),
            license="no LICENSE file; the public release has placeholder text only",
            staged=(), blocked=True,
        ),
    )
}


def external_root() -> Path:
    root = os.environ.get("HBM_EXTERNAL_ROOT", "").strip()
    return Path(root) if root else output_root() / "external_data"


def source(slug: str) -> Source:
    try:
        return SOURCES[slug]
    except KeyError:
        raise KeyError(f"unknown external source {slug!r}; known: {sorted(SOURCES)}") from None


def staged_path(slug: str, kind: str, root: Path | None = None) -> Path:
    if kind not in STAGED_COLUMNS:
        raise ValueError(f"unknown staged table {kind!r}; expected one of {sorted(STAGED_COLUMNS)}")
    source(slug)
    return Path(root or external_root()) / slug / "staged" / f"{kind}.parquet"


def load_staged(slug: str, kind: str, root: Path | None = None) -> pd.DataFrame:
    """One staged table, with its columns checked and every text column as `str`."""
    path = staged_path(slug, kind, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Fetch and stage it with "
            f"`python scripts/fetch_external_data.py --source {slug}` (see docs/EXTERNAL_DATA.md)."
        )
    df = pd.read_parquet(path)
    missing = [c for c in STAGED_COLUMNS[kind] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing staged column(s) {missing}")
    id_col = {"cvs": "cv_id", "jobs": "job_id", "pairs": "pair_id"}[kind]
    if df[id_col].duplicated().any():
        raise ValueError(f"{path}: duplicate {id_col} values")
    for col in STAGED_COLUMNS[kind]:
        if col != "label":
            df[col] = df[col].fillna("").astype(str)
    return df


def prefixed_id(slug: str, native_id: str) -> str:
    """A pool id that cannot collide with a Djinni UUID or another source's id."""
    return f"{slug}:{native_id}"


def parse_extra(value: str) -> dict:
    return json.loads(value) if value else {}


# ---------------------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------------------

#: NLTK's English stopword list, inlined so the guard needs no download. Dropping stopwords
#: (and deleting punctuation) makes a raw text and a classification-preprocessed copy of it
#: normalise to the same tokens; ResumeAtlas ships exactly such a copy.
_NLTK_EN = """i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its itself they them
their theirs themselves what which who whom this that that'll these those am is are was were be
been being have has had having do does did doing a an the and but if or because as until while
of at by for with about against between into through during before after above below to from up
down in out on off over under again further then once here there when where why how all any both
each few more most other some such no nor not only own same so than too very s t can will just
don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't
needn needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn
wouldn't""".split()
STOPWORDS = frozenset(_NLTK_EN) | frozenset(w.replace("'", "") for w in _NLTK_EN)

_NOT_ALNUM = re.compile(r"[^a-z0-9\s]+")
SHINGLE_WORDS = 5

#: Overlap at or above which a training text counts as a copy of a held-out one. The score is
#: max(Jaccard, share of the query's shingles found in the held-out text, share of the
#: held-out text's shingles found in the query), so a base CV inside a longer variant and a
#: truncated variant inside its base are both caught.
LEAK_THRESHOLD = 0.5
#: A match needs at least this many shared shingles, so two short texts that share one
#: boilerplate phrase are not called copies.
MIN_SHARED_SHINGLES = 10


def normalise_tokens(text: str) -> list[str]:
    """NFKD, accents dropped, lowercase, non-alphanumerics deleted, stopwords dropped."""
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    return [w for w in _NOT_ALNUM.sub("", t).split() if w not in STOPWORDS]


def shingle_hashes(text: str, k: int = SHINGLE_WORDS) -> np.ndarray:
    """Sorted unique 64-bit hashes of the word k-grams of the normalised text."""
    tokens = normalise_tokens(text)
    if not tokens:
        return np.zeros(0, dtype=np.uint64)
    grams = (
        [" ".join(tokens)] if len(tokens) < k
        else [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    )
    hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
         for g in grams),
        dtype=np.uint64, count=len(grams),
    )
    return np.unique(hashes)


class NearDupIndex:
    """Held-out texts indexed by shingle, for querying many candidate texts against them.

    The index is two parallel sorted arrays (shingle hash, document number), so a query is a
    `searchsorted` per shingle and a `bincount` over the documents hit -- no pairwise loop, and
    small enough to hold every held-out CV and JD of every source at once.
    """

    def __init__(self, texts: list[str], labels: list[str]):
        if len(texts) != len(labels):
            raise ValueError("texts and labels differ in length")
        self.labels = list(labels)
        per_doc = [shingle_hashes(t) for t in texts]
        self.sizes = np.array([len(h) for h in per_doc], dtype=np.int64)
        docs = np.repeat(np.arange(len(per_doc), dtype=np.int64), self.sizes)
        flat = np.concatenate(per_doc) if per_doc else np.zeros(0, dtype=np.uint64)
        order = np.argsort(flat, kind="stable")
        self._hashes = flat[order]
        self._docs = docs[order]

    def __len__(self) -> int:
        return len(self.labels)

    def best_match(self, text: str) -> tuple[str, float, int]:
        """(label, score, shared shingles) of the closest held-out text; ('', 0.0, 0) if none."""
        query = shingle_hashes(text)
        if query.size == 0 or self._hashes.size == 0:
            return "", 0.0, 0
        left = np.searchsorted(self._hashes, query, side="left")
        right = np.searchsorted(self._hashes, query, side="right")
        hit = right > left
        if not hit.any():
            return "", 0.0, 0
        idx = np.concatenate([np.arange(a, b) for a, b in zip(left[hit], right[hit])])
        shared = np.bincount(self._docs[idx], minlength=len(self.labels))
        candidates = np.flatnonzero(shared)
        inter = shared[candidates].astype(float)
        sizes = self.sizes[candidates].astype(float)
        score = np.maximum.reduce([
            inter / (query.size + sizes - inter),  # Jaccard
            inter / query.size,                    # query contained in the held-out text
            inter / sizes,                         # held-out text contained in the query
        ])
        best = int(np.argmax(score))
        return self.labels[int(candidates[best])], float(score[best]), int(inter[best])

    def matches(self, texts: list[str], threshold: float = LEAK_THRESHOLD,
                min_shared: int = MIN_SHARED_SHINGLES) -> pd.DataFrame:
        """One row per query text at or above the threshold: position, label, score, shared."""
        rows = []
        for position, text in enumerate(texts):
            label, score, shared = self.best_match(text)
            if score >= threshold and shared >= min_shared:
                rows.append({"position": position, "match": label, "score": score,
                             "shared_shingles": shared})
        return pd.DataFrame(rows, columns=["position", "match", "score", "shared_shingles"])


def _held_out_rows(frame: pd.DataFrame, spec: Source) -> pd.DataFrame:
    if spec.holdout_splits is None:
        return frame
    return frame[frame["split"].isin(spec.holdout_splits)]


@cache
def holdout_index(kind: str, root: str | None = None) -> NearDupIndex:
    """Every held-out CV (`kind="cv"`) or JD (`kind="jd"`) of every source, plus the benchmark.

    Fails closed: a held-out source whose staged table is missing raises, because a guard that
    silently indexes nothing reports every pool as clean. Blocked sources are skipped with a
    warning -- their files cannot be on disk, so nothing can have been copied from them.
    """
    from .benchmark import load_benchmark

    table = {"cv": "cvs", "jd": "jobs"}[kind]
    base = Path(root) if root else external_root()
    texts: list[str] = []
    labels: list[str] = []
    for spec in SOURCES.values():
        if not spec.held_out or table not in spec.staged:
            continue
        path = staged_path(spec.slug, table, base)
        if not path.exists():
            if spec.blocked:
                log.warning("external holdout: %s is blocked and not on disk; skipped", spec.slug)
                continue
            raise FileNotFoundError(
                f"external holdout: {path} is missing, so training text cannot be checked "
                f"against {spec.slug}. Fetch it with scripts/fetch_external_data.py first."
            )
        frame = _held_out_rows(load_staged(spec.slug, table, base), spec)
        id_col = "cv_id" if table == "cvs" else "job_id"
        texts.extend(frame["text"].tolist())
        labels.extend(prefixed_id(spec.slug, i) for i in frame[id_col])

    bench = load_benchmark("en")
    column, id_col = ("cv", "candidate_id") if kind == "cv" else ("job_description", "job_id")
    bench = bench.drop_duplicates(id_col)
    texts.extend(bench[column].tolist())
    labels.extend(prefixed_id("benchmark_en", i) for i in bench[id_col])

    log.info("external holdout (%s): %d held-out texts indexed", kind, len(texts))
    return NearDupIndex(texts, labels)


def find_external_leaks(df: pd.DataFrame, cv_col: str = "cv", jd_col: str = "job_description",
                        root: Path | None = None) -> pd.DataFrame:
    """Rows of `df` whose CV or JD copies a held-out text: kind, row index, match, score."""
    key = str(root) if root else None
    frames = []
    for kind, column in (("cv", cv_col), ("jd", jd_col)):
        if column not in df.columns:
            continue
        unique = df[column].astype(str).drop_duplicates()
        found = holdout_index(kind, key).matches(unique.tolist())
        if found.empty:
            continue
        hit_text = dict(zip(unique.iloc[found["position"]].tolist(),
                            found[["match", "score"]].itertuples(index=False)))
        for row_index, text in df[column].astype(str).items():
            if text in hit_text:
                match, score = hit_text[text]
                frames.append({"kind": kind, "row": row_index, "match": match, "score": score})
    return pd.DataFrame(frames, columns=["kind", "row", "match", "score"])


def assert_no_external_leakage(df: pd.DataFrame, where: str, cv_col: str = "cv",
                               jd_col: str = "job_description", root: Path | None = None) -> None:
    """Hard-fails if a frame destined for training copies a held-out external text."""
    leaks = find_external_leaks(df, cv_col, jd_col, root)
    if not leaks.empty:
        examples = leaks.head(3).to_dict("records")
        raise AssertionError(
            f"{where}: external-holdout leakage -- {len(leaks)} row(s) copy a held-out CV or JD "
            f"(score >= {LEAK_THRESHOLD}). Examples: {examples}"
        )

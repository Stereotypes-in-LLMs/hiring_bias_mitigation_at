"""Stage the Vanetik & Kogan vacancy-resume matching set: raw/ -> cvs, jobs, pairs parquet.

    python -I scripts/external/stage_vanetik_kogan.py <raw dir> <staged dir>

65 CVs (Word files from a public Telegram channel), 5 IT vacancies, and two annotators'
rankings of the 5 vacancies for each of the first 30 CVs. A pair row is one (CV, vacancy) with
both annotators' rank in `label` ('a1=2;a2=4'; 'NA' when an annotator's array omits the
vacancy, 'x,y' when it lists it twice). The README is ambiguous about what an array means, so
`extra` carries the rank under both readings (A: position = place, value = vacancy, the literal
README; B: position = vacancy, value = rank) and the raw arrays.

CV text comes from word/document.xml in body order: paragraphs, then table rows (cells joined
by ' | '), then text boxes after the paragraph that anchors them. Only the mc:Choice branch of
an mc:AlternateContent is read, because the mc:Fallback VML copy repeats the same text. The
acquirer used lxml; this is the same walk on the standard library, so staging needs no extra
package. A docx is a zip of XML: only document.xml is read, it is size-capped, and a DTD is
refused, so neither a zip bomb nor an entity expansion can get through.

`extra.pii_counts.phone_like` counts digit runs of 9+ digits that are not year ranges. The
acquirer's regex counted '2019 - 2021' as a phone number and flagged 32 CVs that hold none;
the verifier recounted (`_scripts/vanetik_kogan/verify/fix_phone_like.py`) and that recount
is the rule here.

Reads raw/vacancy-resume-matching-dataset/{CV/<n>.docx, 5_vacancies.csv,
annotations-for-the-first-30-vacancies.txt}.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import dumps, main, marker_patterns, write_table  # noqa: E402

SLUG = "vanetik_kogan"
TABLES = ("cvs", "jobs", "pairs")
REQUIRES = ()
REPO = "vacancy-resume-matching-dataset"
OCCUPATION = "software developer (IT)"
MAX_DOCUMENT_XML_BYTES = 50 * 2**20

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
T, TAB, BR, CR = (f"{{{W}}}{t}" for t in ("t", "tab", "br", "cr"))
P, TBL, TR, TC = (f"{{{W}}}{t}" for t in ("p", "tbl", "tr", "tc"))
SDT, SDT_CONTENT, TXBX, BODY = (f"{{{W}}}{t}" for t in ("sdt", "sdtContent", "txbxContent", "body"))
FALLBACK = f"{{{MC}}}Fallback"
LABEL_SEMANTICS = (
    "rank of this vacancy among the 5 for this CV (1=best), interpretation A: array position = "
    "place, value = vacancy index; 'NA' = vacancy absent from the array, 'x,y' = vacancy listed "
    "twice")
YEARS = re.compile(r"^(?:19|20)\d\d\s*[-–]?\s*(?:(?:19|20)\d\d|\d{2})?$")


class _Doc:
    """word/document.xml with a parent map, which ElementTree does not keep by itself."""

    def __init__(self, xml: bytes):
        if b"<!DOCTYPE" in xml[:4096].upper():
            raise ValueError("document.xml declares a DTD; refusing to parse it")
        self.root = ET.fromstring(xml)
        self.parent = {child: el for el in self.root.iter() for child in el}

    def _inside(self, el, stop, tags) -> bool:
        anc = self.parent.get(el)
        while anc is not None and anc is not stop:
            if anc.tag in tags:
                return True
            anc = self.parent.get(anc)
        return False

    def runs_text(self, p) -> str:
        """A paragraph's text, leaving out text boxes and fallback copies inside it."""
        out = []
        for el in p.iter():
            if self._inside(el, p, (TXBX, FALLBACK)):
                continue
            if el.tag == T:
                out.append(el.text or "")
            elif el.tag == TAB:
                out.append("\t")
            elif el.tag in (BR, CR):
                out.append("\n")
        return "".join(out)

    def textboxes(self, p) -> list[str]:
        return [self.block_text(tb) for tb in p.iter(TXBX)
                if not self._inside(tb, p, (FALLBACK,)) and not self._inside(tb, p, (TXBX,))]

    def block_text(self, container) -> str:
        lines = []
        for child in container:
            if child.tag == P:
                lines.append(self.runs_text(child))
                lines.extend(self.textboxes(child))
            elif child.tag == TBL:
                for tr in child.iter(TR):
                    cells = [self.block_text(tc).replace("\n", " ").strip()
                             for tc in tr.findall(TC)]
                    lines.append(" | ".join(cells))
            elif child.tag == SDT:
                content = child.find(SDT_CONTENT)
                if content is not None:
                    lines.append(self.block_text(content))
        return "\n".join(lines)


def docx_to_text(path: Path) -> tuple[str, dict]:
    with zipfile.ZipFile(path) as z:
        info = z.getinfo("word/document.xml")
        if info.file_size > MAX_DOCUMENT_XML_BYTES:
            raise ValueError(f"{path}: document.xml is {info.file_size} bytes; refusing")
        xml = z.read(info)
        names = z.namelist()
    doc = _Doc(xml)
    body = doc.root.find(BODY)
    stats = {
        "n_txbx": sum(1 for _ in doc.root.iter(TXBX)),
        "n_tbl": sum(1 for _ in body.iter(TBL)),
        "n_media": len([n for n in names if n.startswith("word/media/")]),
        "has_headers": any(n.startswith("word/header") for n in names),
    }
    return doc.block_text(body), stats


def phone_like(text: str) -> int:
    """Runs of 9+ digits (with spaces, dashes, parentheses) that are not years or year ranges."""
    n = 0
    for m in re.finditer(r"\+?\(?\d[\d\s\-()]{6,}\d", text):
        g = m.group().strip()
        if (len(re.sub(r"\D", "", g)) >= 9 and not YEARS.match(re.sub(r"[\s()]", "", g))
                and not re.fullmatch(r"(?:(?:19|20)\d\d[\s\-–]*){2,}", g)):
            n += 1
    return n


def pii_counts(text: str) -> dict[str, int]:
    return {
        "email": len(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)),
        "phone_like": phone_like(text),
        "url": len(re.findall(r"(?i)(?:https?://|www\.|linkedin\.com|github\.com)\S*", text)),
        "asterisk_mask": len(re.findall(r"\*{3,}", text)),
    }


def rankings(raw_text: str, name: str) -> list[list[int]]:
    """One annotator's 30 arrays, from the Python list literal in the annotations file."""
    m = re.search(name + r"\s*=\s*(\[.*?\]\])", raw_text, flags=re.S)
    return ast.literal_eval(re.sub(r"#[^\n]*", "", m.group(1)))


def is_perm(a: list[int]) -> bool:
    return sorted(a) == [1, 2, 3, 4, 5]


def ranks_a(a: list[int]) -> dict[int, list[int]]:
    """Reading A (literal README): a[k] is the vacancy at place k+1 -> places per vacancy."""
    return {j: [k + 1 for k, x in enumerate(a) if x == j] for j in range(1, 6)}


def ranks_b(a: list[int]) -> dict[int, int]:
    """Reading B: a[j-1] is the rank given to vacancy j."""
    return {j: a[j - 1] for j in range(1, 6)}


def stage(raw: Path, out: Path) -> dict[str, int]:
    repo = raw / REPO
    cv_rows = []
    for f in sorted((repo / "CV").iterdir(), key=lambda p: int(p.name.split(".")[0])):
        n = int(f.name.split(".")[0])
        text, docx_stats = docx_to_text(f)
        text = text.strip()
        cv_rows.append({
            "cv_id": str(n), "text": text, "lang": "en", "occupation": OCCUPATION,
            "base_cv_id": str(n), "variant": "",
            "split": "annotated" if n <= 30 else "unannotated", "source_file": f"CV/{f.name}",
            "extra": dumps({"docx_stats": docx_stats, "marker_hits": marker_patterns(text),
                            "pii_counts": pii_counts(text)}),
        })

    vac = pd.read_csv(repo / "5_vacancies.csv", dtype=str, keep_default_na=False)
    job_rows = [{"job_id": str(r["id"]), "text": r["job_description"].strip(),
                 "title": r["job_title"].strip(), "occupation": OCCUPATION, "lang": "en",
                 "split": "", "source_file": "5_vacancies.csv",
                 "extra": json.dumps({"vacancy_index_1to5": i + 1, "uid": r["uid"],
                                      "csv_id": r["id"]})}
                for i, r in vac.iterrows()]
    # vacancy index 1..5 is the CSV row order: an assumption nothing in the release confirms
    job_of = {i + 1: row["job_id"] for i, row in enumerate(job_rows)}

    ann = (repo / "annotations-for-the-first-30-vacancies.txt").read_text(encoding="utf-8")
    a1, a2 = rankings(ann, "ANNOTATOR_1_RANKINGS"), rankings(ann, "ANNOTATOR_2_RANKINGS")
    if not (len(a1) == len(a2) == 30 and all(len(x) == 5 for x in a1 + a2)):
        raise ValueError("expected 30 five-vacancy rankings per annotator")

    def fmt(places: list[int]) -> str:
        return ",".join(map(str, places)) if places else "NA"

    pair_rows = []
    for c in range(30):
        ra1, ra2, rb1, rb2 = ranks_a(a1[c]), ranks_a(a2[c]), ranks_b(a1[c]), ranks_b(a2[c])
        for j in range(1, 6):
            pair_rows.append({
                "pair_id": f"cv{c + 1}_vac{j}", "cv_id": str(c + 1), "job_id": job_of[j],
                "label": f"a1={fmt(ra1[j])};a2={fmt(ra2[j])}", "label_kind": "human_rank",
                "split": "",
                "extra": json.dumps({
                    "vacancy_index_1to5": j, "label_semantics": LABEL_SEMANTICS,
                    "rank_interp_A": {"a1": ra1[j], "a2": ra2[j]},
                    "rank_interp_B": {"a1": rb1[j], "a2": rb2[j]},
                    "raw_array": {"a1": a1[c], "a2": a2[c]},
                    "array_is_permutation": {"a1": is_perm(a1[c]), "a2": is_perm(a2[c])},
                }),
            })
    tables = {"cvs": pd.DataFrame(cv_rows), "jobs": pd.DataFrame(job_rows),
              "pairs": pd.DataFrame(pair_rows)}
    for name, df in tables.items():
        write_table(df, out / f"{name}.parquet")
    return {name: len(df) for name, df in tables.items()}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])

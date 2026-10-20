"""Fetching and staging the external sources: manifests, checksums, extraction, stager rules.

No network and no external data needed: the fetch and verify logic runs on tiny fake raw dirs,
and the stagers' rules run on synthetic rows. Several tests pin a defect the verifiers found in
the acquirers' staged files (RJDB multi-line variants, Kaggle near-duplicates LSH missed, FAIRE
NaN in JSON, Vanetik-Kogan year ranges counted as phone numbers): the port fixes them in the
logic, and these keep them fixed. The tests at the end use the real staged data and skip
cleanly when it is not on disk.
"""

import hashlib
import importlib
import importlib.util
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest

from hiring_bias_mitigation.data import external as X

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_external_data", ROOT / "scripts" / "fetch_external_data.py")
F = importlib.util.module_from_spec(_spec)
sys.modules["fetch_external_data"] = F
_spec.loader.exec_module(F)  # also puts scripts/external on sys.path

MANIFESTS = sorted(p.stem for p in F.MANIFEST_DIR.glob("*.json"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(slug="vanetik_kogan", files=(), fetch=None, **kw) -> dict:
    """A minimal valid manifest for a real slug (validation requires one from SOURCES)."""
    m = {"schema_version": 1, "slug": slug, "name": "test", "upstream": ["https://example.org"],
         "fetch": fetch if fetch is not None else [
             {"id": "zip", "kind": "http", "url": "https://example.org/a.zip", "dest": "a.zip"}],
         "files": list(files), "references": [], "depends_on": {},
         "stage": {"script": "scripts/external/stage_vanetik_kogan.py", "tables": ["cvs"]},
         "blocked": None}
    m.update(kw)
    return m


# ---------------------------------------------------------------------------------------
# The committed manifests
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("slug", MANIFESTS)
def test_committed_manifest_is_valid(slug):
    assert F.validate_manifest(F.load_manifest(slug), slug) == []


def test_every_registered_source_has_a_manifest():
    """A source in the registry with no manifest could not be re-fetched by anyone else."""
    assert set(MANIFESTS) == set(X.SOURCES)


@pytest.mark.parametrize("slug", [s for s in MANIFESTS if F.load_manifest(s)["stage"]])
def test_stager_writes_the_tables_the_registry_expects(slug):
    """The stagers import without their optional packages, so this runs in the project venv."""
    manifest = F.load_manifest(slug)
    module = importlib.import_module(f"stage_{slug}")
    assert list(module.TABLES) == manifest["stage"]["tables"]
    assert set(X.SOURCES[slug].staged) <= set(module.TABLES)


def test_blocked_sources_carry_unblock_steps():
    for slug, spec in X.SOURCES.items():
        if spec.blocked:
            blocked = F.load_manifest(slug)["blocked"]
            assert blocked["reason"] and len(blocked["steps"]) >= 2


def test_marker_snapshot_matches_the_pool_filter():
    """The stagers copy pool.py's markers so a re-stage is reproducible; drift is a decision."""
    from _staging import MARKERS_EN

    from hiring_bias_mitigation.generation.pool import _MARKERS

    assert _MARKERS["en"] == MARKERS_EN


# ---------------------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------------------

GOOD_FILE = {"path": "a.zip", "bytes": 3, "sha256": "0" * 64, "from": "zip"}


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m["files"].append({**GOOD_FILE, "path": "../escape"}), "relative POSIX path"),
    (lambda m: m["files"].append({**GOOD_FILE, "path": "/etc/passwd"}), "relative POSIX path"),
    (lambda m: m["files"].append({**GOOD_FILE, "path": "b", "sha256": "abc"}), "sha256"),
    (lambda m: m["files"].append({**GOOD_FILE, "path": "c", "from": "nope"}), "fetch id"),
    (lambda m: m["files"].append({**GOOD_FILE, "path": "d", "bytes": -1}), "bytes"),
    (lambda m: m["files"].append(dict(GOOD_FILE)), "duplicate path"),
    (lambda m: m["fetch"].append({"id": "g", "kind": "git", "url": "https://x", "commit": "abc",
                                  "dest": "r"}), "40-hex"),
    (lambda m: m["fetch"].append({"id": "h", "kind": "http", "url": "http://x", "dest": "f"}),
     "https"),
    (lambda m: m["fetch"].append({"id": "k", "kind": "ftp", "dest": "f"}), "unknown kind"),
    (lambda m: m["fetch"].append({"id": "z", "kind": "zenodo", "record": "1", "file": "f",
                                  "dest": "f"}), "'md5'"),
    (lambda m: m.update(slug="not_a_source"), "not in external.SOURCES"),
    (lambda m: m.update(blocked={"reason": "gated", "steps": ["ask"]}), "blocked disagrees"),
    (lambda m: m.update(stage={"script": "scripts/external/nope.py", "tables": ["cvs"]}),
     "does not exist"),
    (lambda m: m.pop("files"), "missing key 'files'"),
])
def test_validate_manifest_rejects(mutate, message):
    m = _manifest(files=[dict(GOOD_FILE)])
    assert F.validate_manifest(m) == []
    mutate(m)
    errors = F.validate_manifest(m)
    assert any(message in e for e in errors), errors


# ---------------------------------------------------------------------------------------
# Checksum verification on a fake raw dir
# ---------------------------------------------------------------------------------------

def _fake_raw(tmp_path: Path) -> tuple[Path, dict]:
    raw = tmp_path / "raw"
    (raw / "repo" / "CV").mkdir(parents=True)
    (raw / "repo" / "a.csv").write_bytes(b"id,text\n1,hello\n")
    for i in range(3):
        (raw / "repo" / "CV" / f"{i}.txt").write_text(f"cv {i}")
    (raw / "repo" / ".git").mkdir()
    (raw / "repo" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    tree = F.tree_digest(raw / "repo" / "CV")
    files = [{"path": "repo/a.csv", "bytes": 16, "sha256": _sha(b"id,text\n1,hello\n"),
              "from": "repo"},
             {"path": "repo/CV", **tree, "from": "repo"}]
    fetch = [{"id": "repo", "kind": "git", "url": "https://github.com/x/y", "commit": "a" * 40,
              "dest": "repo"}]
    refs = [{"path": "paper.pdf", "url": "https://example.org/p.pdf", "bytes": 1,
             "sha256": "1" * 64}]
    return raw, _manifest(files=files, fetch=fetch, references=refs)


def test_verify_passes_on_matching_files_and_ignores_absent_references(tmp_path):
    raw, m = _fake_raw(tmp_path)
    assert F.validate_manifest(m) == []
    results = F.verify(raw, m)
    assert [r["status"] for r in results] == [F.OK, F.OK, F.MISSING]
    assert F.failures(results) == []  # a missing optional reference is not a failure


def test_verify_catches_a_changed_byte_a_missing_file_and_an_extra_tree_file(tmp_path):
    raw, m = _fake_raw(tmp_path)
    (raw / "repo" / "a.csv").write_bytes(b"id,text\n1,hellO\n")  # same size, other content
    assert F.check_entry(raw, m["files"][0])[0] == F.MISMATCH
    (raw / "repo" / "a.csv").unlink()
    assert F.check_entry(raw, m["files"][0])[0] == F.MISSING
    (raw / "repo" / "CV" / "3.txt").write_text("planted")
    assert F.check_entry(raw, m["files"][1])[0] == F.MISMATCH
    assert len(F.failures(F.verify(raw, m))) == 2


def test_a_wrong_reference_fails_verify_but_does_not_block_staging(tmp_path, monkeypatch):
    """Optional means it may be absent, not that a corrupted copy is fine; but a paper is not
    a staging input, so it must not stop the tables from being staged."""
    raw, m = _fake_raw(tmp_path / "vanetik_kogan")
    (raw / "paper.pdf").write_bytes(b"x")
    assert [r["path"] for r in F.failures(F.verify(raw, m))] == ["paper.pdf"]
    staged = []
    monkeypatch.setattr(F, "stage_atomically", lambda *a: staged.append(a) or {})
    args = F.argparse.Namespace(stage_only=True, reproduce=False, with_references=True,
                                staged_into=None, work_dir=tmp_path)
    assert F.fetch_and_stage("vanetik_kogan", m, args, tmp_path) == F.EXIT_OK
    assert len(staged) == 1


def test_tree_digest_ignores_git_metadata_but_not_renames(tmp_path):
    raw, _ = _fake_raw(tmp_path)
    tree = raw / "repo"
    before = F.tree_digest(tree)
    (tree / ".git" / "index").write_bytes(b"different clone, same commit")
    assert F.tree_digest(tree) == before
    (tree / "CV" / "2.txt").rename(tree / "CV" / "9.txt")
    assert F.tree_digest(tree)["tree_sha256"] != before["tree_sha256"]


# ---------------------------------------------------------------------------------------
# Extraction and fetching (downloads replaced by local writes)
# ---------------------------------------------------------------------------------------

def _zip_bytes(members: dict[str, bytes], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = 0o120777 << 16
            z.writestr(info, "/etc/passwd")
    return buf.getvalue()


@pytest.mark.parametrize("members, symlink", [
    ({"../evil.txt": b"x"}, None),
    ({"/abs/evil.txt": b"x"}, None),
    ({"ok.txt": b"x"}, "link"),
])
def test_safe_extract_refuses_members_that_escape_or_link(tmp_path, members, symlink):
    zp = tmp_path / "a.zip"
    zp.write_bytes(_zip_bytes(members, symlink))
    with pytest.raises(F.FetchError):
        F.safe_extract(zp, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def _zip_manifest(tmp_path: Path) -> tuple[Path, dict, bytes]:
    payload = _zip_bytes({"data/postings.csv": b"job_id,description\n1,hi\n"})
    files = [{"path": "a.zip", "bytes": len(payload), "sha256": _sha(payload), "from": "zip"},
             {"path": "extracted/data/postings.csv", "bytes": 24,
              "sha256": _sha(b"job_id,description\n1,hi\n"), "from": "zip"}]
    fetch = [{"id": "zip", "kind": "kaggle", "dataset": "o/d", "dest": "a.zip",
              "extract_to": "extracted"}]
    return tmp_path / "raw", _manifest(files=files, fetch=fetch), payload


def test_fetch_downloads_what_is_missing_then_extracts_and_verifies(tmp_path, monkeypatch):
    raw, m, payload = _zip_manifest(tmp_path)
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)

    monkeypatch.setattr(F, "download", fake_download)
    F.fetch(raw, m)
    assert calls == ["https://www.kaggle.com/api/v1/datasets/download/o/d"]
    assert F.failures(F.verify(raw, m)) == []
    F.fetch(raw, m)  # everything verified: nothing is downloaded again
    assert len(calls) == 1


def test_fetch_never_overwrites_a_file_that_differs(tmp_path, monkeypatch):
    raw, m, _ = _zip_manifest(tmp_path)
    raw.mkdir(parents=True)
    (raw / "a.zip").write_bytes(b"someone else's file")
    monkeypatch.setattr(F, "download", lambda url, dest: pytest.fail("must not download"))
    with pytest.raises(F.FetchError, match="not overwriting"):
        F.fetch(raw, m)
    assert (raw / "a.zip").read_bytes() == b"someone else's file"


def test_a_download_with_the_wrong_checksum_is_not_extracted(tmp_path, monkeypatch):
    raw, m, _ = _zip_manifest(tmp_path)

    def wrong_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_zip_bytes({"data/postings.csv": b"tampered"}))

    monkeypatch.setattr(F, "download", wrong_download)
    with pytest.raises(F.FetchError, match="checksum differs"):
        F.fetch(raw, m)
    assert not (raw / "extracted").exists()


def test_zenodo_downloads_are_checked_against_the_published_md5(tmp_path, monkeypatch):
    raw, m, payload = _zip_manifest(tmp_path)
    m["fetch"] = [{"id": "zip", "kind": "zenodo", "record": "1", "file": "a.zip",
                   "md5": "0" * 32, "dest": "a.zip", "extract_to": "extracted"}]
    monkeypatch.setattr(F, "download", lambda url, dest: (dest.parent.mkdir(parents=True,
                                                                            exist_ok=True),
                                                          dest.write_bytes(payload)))
    with pytest.raises(F.FetchError, match="md5"):
        F.fetch(raw, m)


def test_a_drive_error_is_a_fetch_error_and_leaves_no_partial_file(tmp_path, monkeypatch):
    """gdown 6 raises instead of returning None; either way the run must fail cleanly."""
    import types

    def failing_download(id, output, quiet):
        Path(output).write_bytes(b"half a file")
        raise RuntimeError("quota exceeded")

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=failing_download))
    item = {"id": "train", "kind": "gdrive", "file_id": "abc", "dest": "gdrive/train.jsonl"}
    with pytest.raises(F.FetchError, match="quota exceeded"):
        F.fetch_gdrive(item, tmp_path)
    assert list((tmp_path / "gdrive").iterdir()) == []


def test_a_failed_clone_leaves_nothing_that_blocks_the_next_run(tmp_path, monkeypatch):
    def fake_git(args, cwd=None):
        if args[0] == "init":
            (Path(args[-1]) / ".git").mkdir(parents=True)
        if args[0] == "fetch":
            raise F.FetchError("network down")
        return ""

    monkeypatch.setattr(F, "_git", fake_git)
    item = {"id": "repo", "kind": "git", "url": "https://github.com/x/y", "commit": "a" * 40,
            "dest": "repo"}
    with pytest.raises(F.FetchError, match="network down"):
        F.fetch_git(item, tmp_path)
    assert not (tmp_path / "repo").exists()


def test_staging_that_fails_halfway_keeps_the_previous_tables(tmp_path, monkeypatch):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "cvs.parquet").write_bytes(b"previous")
    manifest = {"stage": {"tables": ["cvs", "jobs"]}}

    def half_stage(slug, manifest, root, tmp):
        (tmp / "cvs.parquet").write_bytes(b"new")
        raise RuntimeError("stager crashed before jobs")

    monkeypatch.setattr(F, "stage", half_stage)
    with pytest.raises(RuntimeError):
        F.stage_atomically("x", manifest, tmp_path, out)
    assert (out / "cvs.parquet").read_bytes() == b"previous"
    assert sorted(p.name for p in out.iterdir()) == ["cvs.parquet"]

    def full_stage(slug, manifest, root, tmp):
        for t in ("cvs", "jobs"):
            (tmp / f"{t}.parquet").write_bytes(b"new")
        return {"cvs": 1, "jobs": 1}

    monkeypatch.setattr(F, "stage", full_stage)
    F.stage_atomically("x", manifest, tmp_path, out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {"cvs.parquet": b"new",
                                                               "jobs.parquet": b"new"}


def test_blocked_source_prints_the_steps_and_exits_2_without_fetching(tmp_path, capsys):
    code = F.main(["--source", "confit", "--root", str(tmp_path)])
    assert code == F.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "Tianchi" in out
    assert list(tmp_path.iterdir()) == []


def test_a_failure_outranks_a_blocked_source():
    assert F.overall_exit([F.EXIT_BLOCKED, F.EXIT_FAILED, F.EXIT_OK]) == F.EXIT_FAILED
    assert F.overall_exit([F.EXIT_BLOCKED, F.EXIT_OK]) == F.EXIT_BLOCKED
    assert F.overall_exit([]) == F.EXIT_OK


def test_verify_cli_reports_a_tampered_file(tmp_path, monkeypatch):
    raw, m = _fake_raw(tmp_path / "vanetik_kogan")
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "vanetik_kogan.json").write_text(json.dumps(m))
    monkeypatch.setattr(F, "MANIFEST_DIR", mdir)
    args = ["--verify", "--source", "vanetik_kogan", "--root", str(tmp_path)]
    assert F.main(args) == F.EXIT_OK
    (raw / "repo" / "a.csv").write_bytes(b"changed")
    assert F.main(args) == F.EXIT_FAILED


# ---------------------------------------------------------------------------------------
# Stager rules, on synthetic rows
# ---------------------------------------------------------------------------------------

def test_rjdb_parses_multi_line_experience_changes():
    """440 RJDB records have a multi-line Filtered-information.Experience; without re.S their
    variant lost the year ranges ('exp=shortened') or the target ('exp=dropped:')."""
    from stage_rjdb import experience_change

    shortened = {"Experience": 'instead of "1. Data Manager\n2. Analyst (2018 - 2023)" use '
                               '"1. Data Manager\n2. Analyst (2018 - 2021)"'}
    kind, old, new, desc = experience_change(shortened)
    assert (kind, desc) == ("shortened", "shortened:2018-2023->2018-2021")
    assert old.endswith("(2018 - 2023)") and new.endswith("(2018 - 2021)")
    dropped = {"Experience": 'without including "Embedded C developer\n (2015 - 2017)"'}
    assert experience_change(dropped)[3] == "dropped:Embedded C developer (2015 - 2017)"
    one_line = {"Experience": 'without including "QA  Lead (2015 - 2017)"'}
    assert experience_change(one_line)[3] == "dropped:QA  Lead (2015 - 2017)"  # verbatim


def test_kaggle_near_duplicates_are_found_exhaustively_and_named_after_the_later_row():
    """MinHash LSH missed 3 of the 6 Kaggle pairs at Jaccard >= 0.8; the exhaustive pass may
    not miss any. The cluster is named after its later row, the convention already staged."""
    from stage_kaggle_resume import cluster_ids, near_dup_pairs

    words = [f"w{i}" for i in range(60)]
    a = " ".join(words)
    b = " ".join([*words[:-1], "changed"])  # 51 of 61 shingles shared: Jaccard 0.836
    c = " ".join(words[:30] + [f"x{i}" for i in range(30)])
    pairs = near_dup_pairs([a, c, b])
    assert pairs == [(0, 2)]
    assert cluster_ids(["10", "20", "30"], pairs) == ["30", "", "30"]


def _faire_row() -> dict:
    from stage_faire import COLS

    base = "Experienced accountant with ten years of audit and tax work for small firms."
    variants = {}
    for c, (_axis, group, vtype, _letter) in COLS.items():
        if vtype == "name":
            first = {"male": "John", "female": "Mary"}.get(group, "Ade")
            variants[c] = f"{first} Kone {base}"
        else:
            variants[c] = base + " Volunteered as treasurer for the community garden."
    return {"excel_row": 4, "category": "ACCOUNTANT", "jd": "JD", "base": base,
            "variants": variants}


def test_faire_extras_are_strict_json():
    """Name variants have no token-share fields; the acquirer wrote them as bare NaN, which
    strict JSON readers reject. They must be null."""
    from stage_faire import classify

    summaries = classify([_faire_row()])
    names = [s for (_, c), s in summaries.items() if c % 2 == 0]
    experiences = [s for (_, c), s in summaries.items() if c % 2 == 1]
    assert all(s["class"] == "identity_swap_name_only" for s in names)
    assert all(s["share_base_tokens_kept"] is None and s["lossy_del_ge20pct"] is None
               for s in names)
    assert all(s["class"] == "experience_bearing" for s in experiences)
    for s in summaries.values():
        json.dumps(s, allow_nan=False)  # raises on NaN
        assert not any(isinstance(v, float) and math.isnan(v) for v in s.values())


def test_vanetik_phone_count_ignores_year_ranges():
    from stage_vanetik_kogan import phone_like

    assert phone_like("QA engineer 2019 - 2021, lead 2021-2023, 20212021") == 0
    assert phone_like("Phone: +972 54 123 4567") == 1


def _docx(document_xml: str) -> bytes:
    return _zip_bytes({"word/document.xml": document_xml.encode(), "word/media/a.png": b"png"})


DOCX = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"><w:body>
<w:p><w:r><w:t>Jane Roe</w:t></w:r><w:r><w:tab/><w:t>QA</w:t></w:r></w:p>
<w:p><w:r><mc:AlternateContent>
<mc:Choice Requires="wps"><w:drawing><w:txbxContent><w:p><w:r><w:t>Python, SQL</w:t></w:r></w:p>
</w:txbxContent></w:drawing></mc:Choice>
<mc:Fallback><w:pict><w:txbxContent><w:p><w:r><w:t>Python, SQL</w:t></w:r></w:p></w:txbxContent>
</w:pict></mc:Fallback></mc:AlternateContent></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>2019</w:t></w:r></w:p></w:tc>
<w:tc><w:p><w:r><w:t>Tester</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body></w:document>"""


def test_docx_text_reads_text_boxes_once_and_tables_by_row(tmp_path):
    """mc:Fallback repeats the text box as VML; reading both would double every box."""
    from stage_vanetik_kogan import docx_to_text

    path = tmp_path / "1.docx"
    path.write_bytes(_docx(DOCX))
    text, stats = docx_to_text(path)
    assert text == "Jane Roe\tQA\n\nPython, SQL\n2019 | Tester"
    assert stats == {"n_txbx": 2, "n_tbl": 1, "n_media": 1, "has_headers": False}


def test_docx_with_a_dtd_is_refused(tmp_path):
    from stage_vanetik_kogan import docx_to_text

    path = tmp_path / "bomb.docx"
    path.write_bytes(_docx('<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaa">]><d>&a;</d>'))
    with pytest.raises(ValueError, match="DTD"):
        docx_to_text(path)


def test_sg_counterfactual_is_clean_only_when_the_base_is_untouched():
    """A variant counts as a clean counterfactual only if it is the base plus a cue block."""
    from stage_llmhiringfairness_sg import variant_diffs

    base = "Jane Tan\nSkills: Excel, SAP\nExperience: Accounts clerk, 3 years"
    block = ("\n\n**Additional Information**\n- Languages: English, Malay\n"
             "- Activities: Chess club\n")
    parsed = {"job_000_neutral_0.txt": ("000", "neutral", None, "0"),
              "job_000_malay_female_0.txt": ("000", "malay", "female", "0"),
              "job_000_malay_female_1.txt": ("000", "malay", "female", "1")}
    texts = {"job_000_neutral_0.txt": base, "job_000_malay_female_0.txt": base + block,
             "job_000_malay_female_1.txt": base.replace("3 years", "5 years") + block}
    demo = {"malay_female": [{"languages": "English, Malay", "cca": "Chess club"},
                             {"languages": "English, Malay", "cca": "Chess club"}]}
    dd = variant_diffs(parsed, texts, {"job_000": "job_000_neutral_0.txt"}, demo,
                       {"job_000": "Accounts clerk who plays chess"}).set_index("cue_index")
    assert dd.at[0, "diff_class"] == "pure_append_of_cue_block"
    assert dd.at[1, "diff_class"] == "changes_outside_cue_block"
    assert dd.at[0, "cue_matches_demography_json_entry"]
    assert json.loads(dd.at[0, "cue_jd_token_overlap"]) == ["chess"]


def test_talentclef_siblings_and_gender_votes():
    from stage_talentclef2026_a import gender_votes, mutual_nn_pairs

    a = "Soy ingeniera de datos con cinco años de experiencia en Madrid y Python avanzado"
    b = "Soy ingeniero de datos con cinco años de experiencia en Madrid y Python avanzado"
    c = "Enfermera con diez años en urgencias del hospital general y cuidados intensivos"
    d = "Enfermero con diez años en urgencias del hospital general y cuidados intensivos"
    pairs = mutual_nn_pairs({"1": a, "2": c, "3": b, "4": d})
    assert pairs["1"][0] == "3" and pairs["2"][0] == "4"
    assert gender_votes(a, b) == (1, 0)
    assert gender_votes(b, a) == (0, 1)


# ---------------------------------------------------------------------------------------
# Real data (skipped when it is not on disk)
# ---------------------------------------------------------------------------------------

REAL = X.external_root()
QUICK = ["vanetik_kogan", "onet", "popresume"]  # stage in about a second each


def _real(slug: str) -> Path:
    raw = REAL / slug / "raw"
    if not raw.is_dir() or not (REAL / slug / "staged").is_dir():
        pytest.skip(f"{slug} is not staged under {REAL}")
    return raw


@pytest.mark.parametrize("slug", [*QUICK, "talentclef2026_a", "faire"])
def test_real_raw_files_match_their_manifest(slug):
    raw = _real(slug)
    results = F.verify(raw, F.load_manifest(slug))
    assert F.failures(results) == [], results


@pytest.mark.parametrize("slug", QUICK)
def test_real_restage_reproduces_the_verified_tables(slug, tmp_path):
    """Same rows, order, types and values as the staged file the verifier signed off on, and
    the re-staged copy passes the contract's own column and id checks."""
    _real(slug)
    manifest = F.load_manifest(slug)
    out = tmp_path / slug / "staged"
    F.stage(slug, manifest, REAL, out)
    for table in manifest["stage"]["tables"]:
        report = F.compare_tables(REAL / slug / "staged" / f"{table}.parquet",
                                  out / f"{table}.parquet")
        assert report["values_equal"], (table, report)
        assert report["types_differ"] == {}, (table, report)
        if table in X.STAGED_COLUMNS:
            assert len(X.load_staged(slug, table, tmp_path)) == report["rows"][0]

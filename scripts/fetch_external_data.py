"""Fetch, verify and stage the external data sources (docs/EXTERNAL_DATA.md).

    PY="/ritom/scratch/cscs/zhang/xw/data-tools-venv/bin/python -I"
    $PY scripts/fetch_external_data.py --list
    $PY scripts/fetch_external_data.py --source rjdb      # fetch raw if absent, verify, stage
    $PY scripts/fetch_external_data.py --source faire --source onet
    $PY scripts/fetch_external_data.py --all
    $PY scripts/fetch_external_data.py --verify           # checksums only, no network
    $PY scripts/fetch_external_data.py --stage-only --source rjdb
    $PY scripts/fetch_external_data.py --reproduce --source rjdb   # stage to temp, compare

Which interpreter. Run it with the data-tools venv above, not the project venv: Google Drive
downloads need gdown, and some stagers need rapidfuzz, openpyxl, datasketch or langdetect,
none of which is a project dependency (scripts/external/_staging.py). `--list` and `--verify`
need only pandas and the standard library and run under either. `-I` keeps the current
directory and user site-packages off sys.path.

Where things go. Everything lands under `external.external_root()` ($HBM_EXTERNAL_ROOT, else
$HBM_OUTPUT_ROOT/external_data), `<root>/<slug>/raw` and `<root>/<slug>/staged`, never in git.
`--root` overrides it, and `--staged-into DIR` stages into `DIR/<slug>` instead of `staged/`.

What a manifest pins (data/external_manifests/<slug>.json). For every source: how to fetch
it (a git commit, a Google Drive id, a Kaggle dataset, a Zenodo record with its published md5,
or a plain URL), and for each raw file staging reads, its path under raw/, size and sha256. A
directory of many files (4,100 CVs) is pinned by one tree digest. A download whose checksum
differs from the manifest is never staged: the staged tables are only reproducible from the
exact bytes that were verified. A file already on disk with the wrong checksum is reported
and left alone, not overwritten, so nothing a user put there is lost.

Downloads are data. Nothing from a download is executed or imported: git checkouts run no
hooks and write symlinks as plain files, LFS smudging is off, zip members are extracted only
if their path stays inside the target directory, and the stagers parse CSV/JSON/text/xlsx/docx
without evaluating them.

Blocked sources. popresume's resumes sit behind a manual-approval Hugging Face gate: its five
public JDs are fetched and staged, then the unblock steps are printed. confit's public release
is placeholder text, so nothing is fetched; only the steps are printed. Both exit with 2.

Exit status: 0 when everything asked for succeeded, 1 when a check or download failed, 2 when
a source is blocked and needs the user (a failure outranks a blocked source).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "external"))

from hiring_bias_mitigation.data import external as X  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "data" / "external_manifests"
SCHEMA_VERSION = 1
#: Required fields of each fetch kind, beyond `id` and `kind`.
FETCH_KINDS = {
    "git": ("url", "commit", "dest"),
    "gdrive": ("file_id", "dest"),
    "kaggle": ("dataset", "dest"),
    "zenodo": ("record", "file", "md5", "dest"),
    "http": ("url", "dest"),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
OK, MISSING, MISMATCH = "ok", "missing", "mismatch"
EXIT_OK, EXIT_FAILED, EXIT_BLOCKED = 0, 1, 2


class FetchError(RuntimeError):
    """A download, checkout or checksum failed; the message says which and what to do."""


# ---------------------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------------------

def manifest_path(slug: str, manifest_dir: Path | None = None) -> Path:
    return (manifest_dir or MANIFEST_DIR) / f"{slug}.json"


def load_manifest(slug: str, manifest_dir: Path | None = None) -> dict:
    path = manifest_path(slug, manifest_dir)
    if not path.exists():
        raise FileNotFoundError(f"no manifest for {slug!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path_error(p: object) -> str | None:
    if not isinstance(p, str) or not p:
        return "must be a non-empty string"
    pp = PurePosixPath(p)
    if pp.is_absolute() or ".." in pp.parts or "\\" in p:
        return "must be a relative POSIX path inside raw/ (no '..', no leading '/')"
    return None


def validate_manifest(m: dict, slug: str | None = None, repo_root: Path = REPO_ROOT) -> list[str]:
    """Every problem with a manifest, as messages; an empty list means it is usable.

    Checked here rather than when a download fails halfway, because a manifest is the record
    of what was verified and a malformed one would make `--verify` vacuous.
    """
    errors: list[str] = []
    for key in ("schema_version", "slug", "name", "upstream", "fetch", "files", "stage",
                "blocked"):
        if key not in m:
            errors.append(f"missing key {key!r}")
    if errors:
        return errors
    if m["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {m['schema_version']} != {SCHEMA_VERSION}")
    if slug is not None and m["slug"] != slug:
        errors.append(f"slug {m['slug']!r} does not match the file name {slug!r}")
    if m["slug"] not in X.SOURCES:
        errors.append(f"slug {m['slug']!r} is not in external.SOURCES")
    elif bool(m["blocked"]) != X.SOURCES[m["slug"]].blocked:
        errors.append("blocked disagrees with external.SOURCES")
    if not isinstance(m["upstream"], list) or not m["upstream"]:
        errors.append("upstream must list at least one URL")

    ids = set()
    for i, item in enumerate(m["fetch"]):
        kind = item.get("kind")
        if kind not in FETCH_KINDS:
            errors.append(f"fetch[{i}]: unknown kind {kind!r}")
            continue
        for key in ("id", *FETCH_KINDS[kind]):
            if not item.get(key):
                errors.append(f"fetch[{i}] ({kind}): missing {key!r}")
        if item.get("id") in ids:
            errors.append(f"fetch[{i}]: duplicate id {item.get('id')!r}")
        ids.add(item.get("id"))
        for key in ("dest", "extract_to"):
            if key in item and (err := _relative_path_error(item[key])):
                errors.append(f"fetch[{i}].{key}: {err}")
        if kind == "git" and not COMMIT_RE.match(str(item.get("commit", ""))):
            errors.append(f"fetch[{i}]: commit must be a full 40-hex sha")
        if kind in ("http", "zenodo") and not str(item.get("url", "https://")).startswith(
                "https://"):
            errors.append(f"fetch[{i}]: only https URLs are fetched")

    seen = set()
    for i, f in enumerate(m["files"] + m.get("references", [])):
        where = f"files[{i}]" if i < len(m["files"]) else f"references[{i - len(m['files'])}]"
        if err := _relative_path_error(f.get("path")):
            errors.append(f"{where}.path: {err}")
        if f.get("path") in seen:
            errors.append(f"{where}: duplicate path {f.get('path')!r}")
        seen.add(f.get("path"))
        if not isinstance(f.get("bytes"), int) or f["bytes"] < 0:
            errors.append(f"{where}: bytes must be a non-negative int")
        if "tree_sha256" in f:
            if not SHA256_RE.match(str(f["tree_sha256"])) or not isinstance(f.get("n_files"), int):
                errors.append(f"{where}: a tree needs a 64-hex tree_sha256 and an int n_files")
        elif not SHA256_RE.match(str(f.get("sha256", ""))):
            errors.append(f"{where}: sha256 must be 64 lowercase hex characters")
        if i < len(m["files"]) and f.get("from") not in ids:
            errors.append(f"{where}: 'from' must name a fetch id, got {f.get('from')!r}")
        if i >= len(m["files"]) and not str(f.get("url", "")).startswith("https://"):
            errors.append(f"{where}: a reference needs an https url")

    stage = m["stage"]
    if stage is not None:
        script = repo_root / str(stage.get("script", ""))
        if not script.is_file():
            errors.append(f"stage.script {stage.get('script')!r} does not exist")
        if not stage.get("tables"):
            errors.append("stage.tables must list the tables the stager writes")
    for dep in m.get("depends_on", {}):
        if dep not in X.SOURCES:
            errors.append(f"depends_on names unknown source {dep!r}")
    if m["blocked"] and not m["blocked"].get("steps"):
        errors.append("a blocked source must list the steps that unblock it")
    return errors


# ---------------------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: Path) -> dict:
    """sha256 over the sorted '<sha256>  <relative path>' lines of every file under `root`.

    One digest pins a directory of thousands of files without listing each one; any added,
    removed, renamed or changed file changes it. `.git` directories are left out, because a
    checkout's own metadata differs between clones of the same commit.
    """
    lines, total = [], 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in
                       p.relative_to(root).parts):
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}\n")
        total += path.stat().st_size
    return {"tree_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
            "n_files": len(lines), "bytes": total}


def check_entry(raw: Path, entry: dict) -> tuple[str, str]:
    """(status, detail) of one manifest file or tree against what is on disk."""
    path = raw / entry["path"]
    if "tree_sha256" in entry:
        if not path.is_dir():
            return MISSING, f"{entry['path']}/ is not on disk"
        got = tree_digest(path)
        if got["tree_sha256"] != entry["tree_sha256"] or got["n_files"] != entry["n_files"]:
            return MISMATCH, (f"{entry['path']}/: {got['n_files']} files, tree "
                              f"{got['tree_sha256'][:12]}, expected {entry['n_files']} files, "
                              f"tree {entry['tree_sha256'][:12]}")
        return OK, f"{entry['path']}/ ({got['n_files']} files)"
    if not path.is_file():
        return MISSING, f"{entry['path']} is not on disk"
    size = path.stat().st_size
    if size != entry["bytes"]:
        return MISMATCH, f"{entry['path']}: {size} bytes, expected {entry['bytes']}"
    got = sha256_file(path)
    if got != entry["sha256"]:
        return MISMATCH, (f"{entry['path']}: sha256 {got[:12]}..., expected "
                          f"{entry['sha256'][:12]}...")
    return OK, entry["path"]


def verify(raw: Path, manifest: dict, references: bool = True) -> list[dict]:
    """One result per manifest file (and per reference document that is on disk)."""
    results = []
    for entry in manifest["files"]:
        status, detail = check_entry(raw, entry)
        results.append({"path": entry["path"], "status": status, "detail": detail,
                        "required": True})
    if references:
        for entry in manifest.get("references", []):
            status, detail = check_entry(raw, entry)
            results.append({"path": entry["path"], "status": status, "detail": detail,
                            "required": False})
    return results


def failures(results: list[dict]) -> list[dict]:
    """Results that should stop staging: any mismatch, and a missing required file."""
    return [r for r in results
            if r["status"] == MISMATCH or (r["status"] == MISSING and r["required"])]


# ---------------------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------------------

def safe_extract(zip_path: Path, dest: Path) -> int:
    """Extracts every member whose path stays inside `dest`; refuses links and escapes."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise FetchError(f"{zip_path}: CRC check failed for member {bad!r}")
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if target != dest and dest not in target.parents:
                raise FetchError(f"{zip_path}: member {info.filename!r} escapes {dest}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise FetchError(f"{zip_path}: member {info.filename!r} is a symlink")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n += 1
    return n


def download(url: str, dest: Path) -> None:
    """HTTPS GET to `dest` via curl (redirects followed, https only); atomic rename at the end."""
    if not url.startswith("https://"):
        raise FetchError(f"refusing a non-https download: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    cmd = ["curl", "-fL", "--retry", "3", "--proto", "=https", "--proto-redir", "=https",
           "-o", str(part), url]
    print(f"  downloading {url}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        part.unlink(missing_ok=True)
        raise FetchError(f"download failed: {url}")
    os.replace(part, dest)


def _git(args: list[str], cwd: Path | None = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    proc = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FetchError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return _git(["rev-parse", "HEAD"], cwd=path)
    except FetchError:
        return None


def fetch_git(item: dict, raw: Path) -> None:
    """A checkout of exactly the pinned commit, with hooks off and symlinks as plain files."""
    dest = raw / item["dest"]
    head = git_head(dest)
    if head == item["commit"]:
        return
    if dest.exists() and any(dest.iterdir()):
        raise FetchError(f"{dest} exists but is not a checkout of {item['commit'][:12]} "
                         f"(HEAD {head or 'none'}); move it away and re-run")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  git {item['url']} @ {item['commit'][:12]}", flush=True)
    try:
        _git(["init", "-q", str(dest)])
        for key, value in (("core.symlinks", "false"), ("core.hooksPath", "/dev/null"),
                           ("protocol.file.allow", "never")):
            _git(["config", key, value], cwd=dest)
        _git(["remote", "add", "origin", item["url"]], cwd=dest)
        # GitHub serves a commit by sha, so a depth-1 fetch gets exactly the pinned tree
        _git(["fetch", "-q", "--depth", "1", "origin", item["commit"]], cwd=dest)
        _git(["-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"], cwd=dest)
        if git_head(dest) != item["commit"]:
            raise FetchError(f"{dest}: checkout is not at {item['commit']}")
    except FetchError:
        # a half-made checkout would make every later run refuse to touch the directory
        shutil.rmtree(dest, ignore_errors=True)
        raise


def fetch_gdrive(item: dict, raw: Path) -> None:
    try:
        import gdown
    except ImportError:
        raise FetchError("Google Drive downloads need gdown; run this script with "
                         "/ritom/scratch/cscs/zhang/xw/data-tools-venv/bin/python -I") from None
    dest = raw / item["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  gdown {item['file_id']}", flush=True)
    try:
        # gdown 6 raises on failure; older versions returned None
        ok = gdown.download(id=item["file_id"], output=str(part), quiet=False) is not None
    except Exception as err:  # FileURLRetrievalError, requests errors: all mean "not fetched"
        ok, reason = False, f"{type(err).__name__}: {err}"
    else:
        reason = "gdown returned nothing"
    if not ok or not part.exists():
        part.unlink(missing_ok=True)
        raise FetchError(f"gdown could not fetch Drive file {item['file_id']} ({reason})")
    os.replace(part, dest)


def item_url(item: dict) -> str:
    if item["kind"] == "kaggle":
        # Kaggle's unauthenticated download endpoint; it redirects to a signed bundle URL
        return f"https://www.kaggle.com/api/v1/datasets/download/{item['dataset']}"
    if item["kind"] == "zenodo":
        return f"https://zenodo.org/api/records/{item['record']}/files/{item['file']}/content"
    return item["url"]


def fetch_item(item: dict, entries: list[dict], raw: Path) -> None:
    """Brings one fetch item's files onto disk, downloading only what is missing."""
    states = {e["path"]: check_entry(raw, e)[0] for e in entries}
    bad = [p for p, s in states.items() if s == MISMATCH]
    if bad:
        raise FetchError(f"{raw}: {bad} differ from the manifest; not overwriting them. Move "
                         "them away to re-fetch, or investigate why they changed.")
    if all(s == OK for s in states.values()):
        return
    if item["kind"] == "git":
        fetch_git(item, raw)
        return
    dest = raw / item["dest"]
    if not dest.exists():
        if item["kind"] == "gdrive":
            fetch_gdrive(item, raw)
        else:
            download(item_url(item), dest)
    own = next((e for e in entries if e["path"] == item["dest"]), None)
    if own is not None and check_entry(raw, own)[0] != OK:
        raise FetchError(f"{dest}: checksum differs from the manifest after download "
                         f"({check_entry(raw, own)[1]})")
    if item["kind"] == "zenodo" and md5_file(dest) != item["md5"]:
        raise FetchError(f"{dest}: md5 differs from the one Zenodo publishes ({item['md5']})")
    if item.get("extract_to") and any(states[e["path"]] != OK for e in entries
                                      if e["path"] != item["dest"]):
        print(f"  extracting {item['dest']} -> {item['extract_to']}/", flush=True)
        safe_extract(dest, raw / item["extract_to"])


def fetch(raw: Path, manifest: dict, references: bool = False) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for item in manifest["fetch"]:
        fetch_item(item, [e for e in manifest["files"] if e["from"] == item["id"]], raw)
    if references:
        for ref in manifest.get("references", []):
            if not (raw / ref["path"]).exists():
                try:
                    download(ref["url"], raw / ref["path"])
                except FetchError as err:
                    print(f"  reference not fetched ({err}); it is optional", flush=True)


# ---------------------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------------------

def stager(slug: str):
    """The stage_<slug> module from scripts/external (our own code, never a download's)."""
    return importlib.import_module(f"stage_{slug}")


def stage(slug: str, manifest: dict, root: Path, out: Path) -> dict:
    module = stager(slug)
    kwargs = {arg: root / dep / "raw" for dep, arg in manifest.get("depends_on", {}).items()}
    return module.stage(root / slug / "raw", out, **kwargs)


def stage_atomically(slug: str, manifest: dict, root: Path, out: Path) -> dict:
    """Stages into a scratch directory beside `out` and moves the tables in only when all of
    them were written, so a stager that fails halfway never leaves old and new tables mixed."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".staging-", dir=out))
    try:
        counts = stage(slug, manifest, root, tmp)
        for table in manifest["stage"]["tables"]:
            os.replace(tmp / f"{table}.parquet", out / f"{table}.parquet")
        return counts
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compare_tables(ref: Path, new: Path) -> dict:
    """Row count, columns, Arrow types and every value of two parquet files."""
    import pyarrow.parquet as pq

    a, b = pq.read_table(ref), pq.read_table(new)
    out = {"rows": [a.num_rows, b.num_rows], "columns_equal": a.column_names == b.column_names,
           "types_differ": {f.name: [str(f.type), str(b.schema.field(f.name).type)]
                            for f in a.schema if f.name in b.column_names
                            and str(f.type) != str(b.schema.field(f.name).type)},
           "values_differ": {}}
    if a.num_rows == b.num_rows:
        for c in a.column_names:
            if c in b.column_names:
                x, y = a.column(c).to_pylist(), b.column(c).to_pylist()
                n = sum(1 for u, v in zip(x, y) if u != v and not (u != u and v != v))
                if n:
                    out["values_differ"][c] = n
    out["values_equal"] = (a.num_rows == b.num_rows and out["columns_equal"]
                           and not out["values_differ"])
    return out


def reproduce(slug: str, manifest: dict, root: Path, work: Path) -> dict:
    """Stages into a fresh temp dir and compares each table with the staged one on disk."""
    tmp = Path(tempfile.mkdtemp(prefix=f"restage-{slug}-", dir=work))
    try:
        stage(slug, manifest, root, tmp)
        staged = root / slug / "staged"
        return {t: (compare_tables(staged / f"{t}.parquet", tmp / f"{t}.parquet")
                    if (staged / f"{t}.parquet").exists() else {"values_equal": False,
                                                               "error": "no staged table"})
                for t in manifest["stage"]["tables"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def blocker_message(slug: str, manifest: dict) -> str:
    b = manifest["blocked"]
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(b["steps"], 1))
    return f"{slug} is BLOCKED: {b['reason']}\nWhat you need to do:\n{steps}"


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------

def list_sources(root: Path) -> None:
    print(f"external root: {root}")
    print(f"{'slug':22} {'status':9} {'raw':4} {'staged':7} {'roles'}")
    for slug, spec in X.SOURCES.items():
        has_manifest = manifest_path(slug).exists()
        status = "blocked" if spec.blocked else ("ready" if has_manifest else "no-manif")
        raw = "yes" if (root / slug / "raw").is_dir() else "-"
        staged_dir = root / slug / "staged"
        n_staged = len(list(staged_dir.glob("*.parquet"))) if staged_dir.is_dir() else 0
        print(f"{slug:22} {status:9} {raw:4} {n_staged or '-'!s:7} {','.join(spec.roles)}")


def print_results(slug: str, results: list[dict]) -> None:
    for r in results:
        if r["status"] != OK or not r["required"]:
            tag = r["status"].upper() + ("" if r["required"] else " (optional reference)")
            print(f"  {tag:30} {r['detail']}")
    n_ok = sum(r["status"] == OK for r in results if r["required"])
    n_req = sum(r["required"] for r in results)
    print(f"  {slug}: {n_ok}/{n_req} required files verified")


def run_source(slug: str, args, root: Path) -> int:
    manifest = load_manifest(slug)
    errors = validate_manifest(manifest, slug)
    if errors:
        print(f"{slug}: invalid manifest: {errors}")
        return EXIT_FAILED
    raw = root / slug / "raw"
    print(f"== {slug}", flush=True)

    if args.verify:
        results = verify(raw, manifest)
        print_results(slug, results)
        return EXIT_FAILED if failures(results) else EXIT_OK
    if not manifest["blocked"]:
        return fetch_and_stage(slug, manifest, args, root)
    # A blocked source's public part is fetched only when something stages from it (popresume's
    # five JDs); confit's public release is placeholders, so there is nothing worth fetching.
    code = (fetch_and_stage(slug, manifest, args, root) if manifest["stage"] is not None
            else EXIT_OK)
    print(blocker_message(slug, manifest))
    return overall_exit([code, EXIT_BLOCKED])


def fetch_and_stage(slug: str, manifest: dict, args, root: Path) -> int:
    raw = root / slug / "raw"
    for dep in manifest.get("depends_on", {}):
        dep_manifest = load_manifest(dep)
        if not args.stage_only and not args.reproduce:
            fetch(root / dep / "raw", dep_manifest)
        if failures(verify(root / dep / "raw", dep_manifest, references=False)):
            print(f"  {slug} stages against {dep}'s raw files, which are missing or differ; "
                  f"fetch {dep} first (--source {dep})")
            return EXIT_FAILED

    if not args.stage_only and not args.reproduce:
        try:
            fetch(raw, manifest, references=args.with_references)
        except FetchError as err:
            print(f"  FAILED: {err}")
            return EXIT_FAILED
    results = verify(raw, manifest, references=args.with_references)
    print_results(slug, results)
    # Only the files staging reads can stop it; a paper that changed upstream is reported
    # above (and fails --verify) but has no bearing on the staged tables.
    if failures([r for r in results if r["required"]]):
        print(f"  {slug}: not staged, because the raw files do not match the manifest")
        return EXIT_FAILED

    if manifest["stage"] is None:
        return EXIT_OK
    if args.reproduce:
        report = reproduce(slug, manifest, root, args.work_dir)
        for table, r in report.items():
            print(f"  {table}: {json.dumps(r)}")
        return EXIT_OK if all(r["values_equal"] for r in report.values()) else EXIT_FAILED
    out = (args.staged_into / slug) if args.staged_into else root / slug / "staged"
    counts = stage_atomically(slug, manifest, root, out)
    print(f"  staged into {out}: {json.dumps(counts)}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list sources and what is on disk")
    parser.add_argument("--source", action="append", choices=sorted(X.SOURCES),
                        help="a source to process; repeat for several")
    parser.add_argument("--all", action="store_true", help="every source with a manifest")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="checksums only, no network (all sources unless --source is given)")
    mode.add_argument("--stage-only", action="store_true",
                      help="stage from the raw files on disk; no network")
    mode.add_argument("--reproduce", action="store_true",
                      help="stage into a temp dir and compare with the staged tables; no network")
    parser.add_argument("--with-references", action="store_true",
                        help="also fetch/verify the optional reference documents (papers)")
    parser.add_argument("--root", type=Path, default=None,
                        help="external data root (default: external.external_root())")
    parser.add_argument("--staged-into", type=Path, default=None,
                        help="stage into DIR/<slug> instead of <root>/<slug>/staged")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()),
                        help="where --reproduce stages its temporary copy")
    args = parser.parse_args(argv)
    root = (args.root or X.external_root()).resolve()
    if args.list:
        list_sources(root)
        return EXIT_OK
    if not args.source and not args.all:
        if not args.verify:
            parser.error("name a source (--source SLUG), --all, --list or --verify")
        args.all = True
    slugs = (sorted(p.stem for p in MANIFEST_DIR.glob("*.json")) if args.all
             else list(dict.fromkeys(args.source)))
    codes = {}
    for slug in slugs:
        try:
            codes[slug] = run_source(slug, args, root)
        except (FetchError, FileNotFoundError, ModuleNotFoundError, ValueError) as err:
            print(f"  {slug}: FAILED: {err}")
            codes[slug] = EXIT_FAILED
    if len(codes) > 1:
        names = {EXIT_OK: "ok", EXIT_BLOCKED: "blocked", EXIT_FAILED: "FAILED"}
        print("summary: " + ", ".join(f"{s}={names[c]}" for s, c in codes.items()))
    return overall_exit(codes.values())


def overall_exit(codes) -> int:
    """A failure outranks a blocked source: blocked is expected, a failure is not."""
    codes = set(codes)
    if EXIT_FAILED in codes:
        return EXIT_FAILED
    return EXIT_BLOCKED if EXIT_BLOCKED in codes else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

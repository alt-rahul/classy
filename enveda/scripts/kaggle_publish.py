"""Publish local files as a PRIVATE Kaggle dataset: create it, or add a new version (WP14).

    python scripts/kaggle_publish.py --name casmi-code \\
        --src src/casmi=casmi --src configs=configs --src pyproject.toml \\
        --message "code at <commit>"

``--src SRC[=DEST]`` (repeatable): a directory uploads its contents (recursively) under
the dataset directory DEST (default: the dataset root); a file goes to DEST/<name>.
Relative SRC paths are resolved against the enveda checkout. ``--exclude`` globs
(matched against file names and every path component) skip files; ``__pycache__``,
``*.pyc`` and dot-files are always skipped.

Every version also carries a generated ``MANIFEST.json`` (dataset path, size, sha256,
git commit) at its root. The script uploads each file through the blob API (chunked,
resumable; tokens are cached in ``dist/kaggle/<name>.tokens.json`` for 12 hours so a
rerun after a failure skips files already sent), creates the dataset (private) or a new
version, then waits until Kaggle reports it READY. Subdirectories are sent as the
``directories`` tree of the create request, so the mounted dataset keeps the layout.

Credentials and owner: see scripts/kaggle_common.py.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_common as kc  # noqa: E402

ALWAYS_EXCLUDE = ("__pycache__", "*.pyc", ".*")
TOKEN_TTL_S = 12 * 3600


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="dataset slug, e.g. casmi-code")
    p.add_argument("--src", action="append", required=True, metavar="SRC[=DEST]",
                   help="file or directory to include (repeatable)")
    p.add_argument("--message", required=True, help="version notes")
    p.add_argument("--title", default=None, help="dataset title on creation (default: the slug)")
    p.add_argument("--owner", default=None, help="owner slug (default: discovered)")
    p.add_argument("--exclude", action="append", default=[], help="glob to skip (repeatable)")
    p.add_argument("--jobs", type=int, default=3, help="parallel file uploads")
    p.add_argument("--license", default="unknown", help="license name on creation")
    p.add_argument("--timeout", type=float, default=3 * 3600, help="seconds to wait for READY")
    p.add_argument("--no-wait", action="store_true", help="do not wait for READY")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded and stop")
    return p.parse_args(argv)


def _skip(rel: Path, patterns: list[str]) -> bool:
    for part in rel.parts:
        if any(fnmatch.fnmatch(part, pat) for pat in patterns):
            return True
    return False


def collect(srcs: list[str], excludes: list[str]) -> list[tuple[str, Path]]:
    """[(dataset path, local path)] sorted by dataset path; error on duplicates."""
    patterns = list(ALWAYS_EXCLUDE) + list(excludes)
    out: dict[str, Path] = {}
    for spec in srcs:
        src, _, dest = spec.partition("=")
        local = Path(src).expanduser()
        if not local.is_absolute():
            local = kc.ENV_ROOT / local
        dest = dest.strip("/")
        if local.is_dir():
            items = [(f.relative_to(local), f) for f in sorted(local.rglob("*")) if f.is_file()]
        elif local.is_file():
            items = [(Path(local.name), local)]
        else:
            kc.die(f"--src {spec}: {local} does not exist")
        for rel, f in items:
            if _skip(rel, patterns):
                continue
            key = str(Path(dest) / rel) if dest else str(rel)
            if key in out and out[key] != f:
                kc.die(f"two sources map to {key}: {out[key]} and {f}")
            out[key] = f
    if "MANIFEST.json" in out:
        kc.die("MANIFEST.json is generated; do not include one")
    return sorted(out.items())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(kc.ENV_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(name: str, files: list[tuple[str, Path]], message: str, stage: Path) -> Path:
    entries = [{"path": k, "size": f.stat().st_size, "sha256": _sha256(f)} for k, f in files]
    man = {"dataset": name, "message": message, "git_commit": _git_commit(),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_files": len(entries), "total_bytes": sum(e["size"] for e in entries), "files": entries}
    stage.mkdir(parents=True, exist_ok=True)
    p = stage / "MANIFEST.json"
    p.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return p


def _tree(tokens: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Dataset paths -> (root files, directories) in the CreateDataset request shape."""
    root: dict = {"files": [], "dirs": {}}
    for key in sorted(tokens):
        parts = key.split("/")
        node = root
        for d in parts[:-1]:
            node = node["dirs"].setdefault(d, {"files": [], "dirs": {}})
        node["files"].append({"token": tokens[key]})

    def conv(node: dict) -> list[dict]:
        return [{"name": n, "files": c["files"], "directories": conv(c)}
                for n, c in sorted(node["dirs"].items())]

    return root["files"], conv(root)


def _load_tokens(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {k: v for k, v in d.items() if now - v.get("t", 0) < TOKEN_TTL_S}


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    files = collect(args.src, args.exclude)
    if not files:
        kc.die("nothing to upload")
    total = sum(f.stat().st_size for _, f in files)
    kc.say(f"dataset {args.name}: {len(files)} files, {total / 1e6:.1f} MB")
    for k, f in files[:20]:
        kc.say(f"  {k}  ({f.stat().st_size / 1e6:.2f} MB)")
    if len(files) > 20:
        kc.say(f"  ... {len(files) - 20} more")
    if args.dry_run:
        return 0

    own = kc.owner(args.owner)
    stage = kc.ENV_ROOT / "dist" / "kaggle" / args.name
    manifest = write_manifest(args.name, files, args.message, stage)
    files = files + [("MANIFEST.json", manifest)]
    tok_path = stage.parent / f"{args.name}.tokens.json"
    cache = _load_tokens(tok_path)

    def sig(f: Path) -> str:
        st = f.stat()
        return f"{f.resolve()}|{st.st_size}|{int(st.st_mtime)}"

    tokens: dict[str, str] = {}
    todo = []
    for k, f in files:
        c = cache.get(sig(f))
        if c and k != "MANIFEST.json":
            tokens[k] = c["token"]
        else:
            todo.append((k, f))
    kc.say(f"uploading {len(todo)} files ({len(files) - len(todo)} cached) as {own}/{args.name}")
    t0 = time.time()

    def up(item: tuple[str, Path]) -> tuple[str, Path, str]:
        k, f = item
        return k, f, kc.upload_blob(f, "DATASET")

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for k, f, tok in ex.map(up, todo):
            tokens[k] = tok
            cache[sig(f)] = {"token": tok, "t": time.time()}
            tok_path.write_text(json.dumps(cache, indent=0))
    kc.say(f"uploads done in {time.time() - t0:.0f} s")

    root_files, dirs = _tree(tokens)
    info = kc.dataset_info(own, args.name)
    if info is None:
        body = {"ownerSlug": own, "slug": args.name, "title": args.title or args.name,
                "licenseName": args.license, "isPrivate": True, "files": root_files,
                "directories": dirs, "description": args.message, "categoryIds": []}
        res = kc.rpc("datasets.DatasetApiService", "CreateDataset", body, timeout=600)
        action = "created"
    else:
        if not info.get("isPrivate", False):
            kc.say(f"WARNING: {own}/{args.name} is PUBLIC on Kaggle; make it private in the UI")
        body = {"ownerSlug": own, "datasetSlug": args.name,
                "body": {"versionNotes": args.message, "deleteOldVersions": False,
                         "files": root_files, "directories": dirs, "categoryIds": []}}
        res = kc.rpc("datasets.DatasetApiService", "CreateDatasetVersion", body, timeout=600)
        action = "new version"
    if res.get("error"):
        kc.die(f"{action} {own}/{args.name}: {res.get('error')}")
    kc.say(f"{action}: {res.get('ref') or own + '/' + args.name} {res.get('url', '')} status {res.get('status')}")
    for k, f in files:
        cache.pop(sig(f), None)  # tokens are single use
    tok_path.write_text(json.dumps(cache, indent=0))

    if args.no_wait:
        return 0
    st = kc.wait_dataset_ready(own, args.name, timeout_s=args.timeout)
    info = kc.dataset_info(own, args.name) or {}
    kc.say(f"{own}/{args.name}: {st}, version {info.get('currentVersionNumber')}, "
           f"private {info.get('isPrivate')}, {info.get('totalBytes')} bytes")
    return 0 if st == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())

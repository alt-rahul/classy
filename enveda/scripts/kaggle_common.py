"""Minimal Kaggle API client shared by the kaggle_* scripts (WP14).

Talks to the RPC endpoints the official ``kagglesdk`` uses
(``POST https://www.kaggle.com/api/v1/<service>/<method>`` with a camelCase JSON body)
using only ``requests``, so nothing beyond the repo requirements has to be installed.

Credentials, in order (never printed, logged or written):
1. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables.
2. ``~/.kaggle/kaggle.json`` (``{"username": ..., "key": ...}``).
3. A dummy Basic header. In the cloud sessions of this project a network proxy replaces
   the Authorization header of every request to www.kaggle.com with the user's real key.

The owner slug (Kaggle username) comes from ``--owner``, else ``KAGGLE_OWNER`` or
``KAGGLE_USERNAME``, else it is discovered from the API (the owner of the caller's own
datasets, then of the caller's own kernels).

File uploads use the blob flow of the official client: ``StartBlobUpload`` returns a
Google Cloud Storage resumable-upload URL and a token; the file is sent in chunks with
``Content-Range`` so a dropped connection costs one chunk, then the token is passed to
``CreateDataset`` / ``CreateDatasetVersion``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests

API = "https://www.kaggle.com/api/v1"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
COMPETITION_ID = 143318
RDKIT_WHEEL_DATASET = "metric/rdkit-2026-3-3-wheel"
CHUNK = 64 * 1024 * 1024  # upload chunk, a multiple of 256 KiB as GCS requires

ENV_ROOT = Path(__file__).resolve().parents[1]  # the enveda checkout


class KaggleError(RuntimeError):
    pass


def say(*args: Any) -> None:
    print(*args, flush=True)


def _auth() -> tuple[str, str]:
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if user and key:
        return user, key
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cfg.is_file():
        try:
            d = json.loads(cfg.read_text())
            if d.get("username") and d.get("key"):
                return str(d["username"]), str(d["key"])
        except (OSError, ValueError):
            pass
    return "dummy", "dummy"  # replaced by the credential-injecting proxy


_SESSION: requests.Session | None = None


def session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.auth = _auth()
        s.headers["User-Agent"] = "casmi-kaggle-scripts/1"
        _SESSION = s
    return _SESSION


def _retrying(fn: Callable[[], requests.Response], what: str, retries: int = 4) -> requests.Response:
    delay = 5.0
    for attempt in range(retries + 1):
        try:
            r = fn()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == retries:
                raise KaggleError(f"{what}: {type(exc).__name__}: {exc}") from None
            say(f"  {what}: {type(exc).__name__}, retry in {delay:.0f} s")
        else:
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
                return r
            say(f"  {what}: HTTP {r.status_code}, retry in {delay:.0f} s")
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise AssertionError("unreachable")


def rpc(service: str, method: str, body: dict | None = None, timeout: float = 120) -> dict:
    """POST ``body`` to ``<API>/<service>/<method>``; the JSON reply, KaggleError on error."""
    url = f"{API}/{service}/{method}"
    r = _retrying(lambda: session().post(url, json=body or {}, timeout=timeout), method)
    ctype = r.headers.get("Content-Type", "")
    data: Any = None
    if "application/json" in ctype:
        try:
            data = r.json()
        except ValueError:
            data = None
    if r.status_code >= 400 or (isinstance(data, dict) and int(data.get("code", 0) or 0) >= 400):
        msg = data.get("message") if isinstance(data, dict) else r.text[:300]
        raise KaggleError(f"{method}: HTTP {r.status_code}: {msg}")
    if data is None:
        raise KaggleError(f"{method}: HTTP {r.status_code}, non-JSON reply ({ctype})")
    return data


def get_v1(path: str, params: dict | None = None, timeout: float = 120) -> Any:
    """GET the legacy REST API (``<API>/<path>``)."""
    url = f"{API}/{path.lstrip('/')}"
    r = _retrying(lambda: session().get(url, params=params, timeout=timeout), path)
    if r.status_code >= 400:
        raise KaggleError(f"GET {path}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


# --- identity ---------------------------------------------------------------------------

_OWNER: str | None = None


def owner(explicit: str | None = None) -> str:
    """Kaggle username used as owner slug (see the module docstring)."""
    global _OWNER
    if explicit:
        return explicit
    for var in ("KAGGLE_OWNER", "KAGGLE_USERNAME"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    if _OWNER:
        return _OWNER
    try:
        mine = get_v1("datasets/list", {"group": "my"})
        refs = sorted({d["ownerRef"] for d in mine if d.get("ownerRef")})
        if len(refs) == 1:
            _OWNER = refs[0]
            return _OWNER
    except KaggleError:
        pass
    try:
        ks = rpc("kernels.KernelsApiService", "ListKernels", {"group": "PROFILE",
                                                               "pageSize": 20}).get("kernels", [])
        refs = sorted({k["ref"].split("/")[0] for k in ks if "/" in k.get("ref", "")})
        if len(refs) == 1:
            _OWNER = refs[0]
            return _OWNER
    except KaggleError:
        pass
    raise KaggleError("cannot discover the Kaggle username; pass --owner or set KAGGLE_OWNER")


# --- uploads ----------------------------------------------------------------------------

def _gcs_offset(url: str, size: int) -> int | None:
    """Bytes the resumable session already holds, or None if it is complete."""
    r = requests.put(url, headers={"Content-Range": f"bytes */{size}", "Content-Length": "0"},
                     timeout=120)
    if r.status_code in (200, 201):
        return None
    if r.status_code == 308:
        rng = r.headers.get("Range")
        return int(rng.split("-")[1]) + 1 if rng else 0
    raise KaggleError(f"upload status query: HTTP {r.status_code}: {r.text[:200]}")


def upload_blob(path: Path, blob_type: str = "DATASET", quiet: bool = False) -> str:
    """Upload one file; returns the blob token for CreateDataset(Version)."""
    path = Path(path)
    size = path.stat().st_size
    start = rpc("blobs.BlobApiService", "StartBlobUpload", {
        "type": blob_type, "name": path.name, "contentLength": size,
        "lastModifiedEpochSeconds": int(path.stat().st_mtime)})
    url, token = start["createUrl"], start["token"]
    t0 = time.time()
    if size == 0:
        r = requests.put(url, data=b"", timeout=120)
        if r.status_code not in (200, 201):
            raise KaggleError(f"upload {path.name}: HTTP {r.status_code}: {r.text[:200]}")
        return token
    offset, failures = 0, 0
    with path.open("rb") as fh:
        while offset is not None and offset < size:
            n = min(CHUNK, size - offset)
            fh.seek(offset)
            data = fh.read(n)
            hdr = {"Content-Range": f"bytes {offset}-{offset + n - 1}/{size}",
                   "Content-Length": str(n)}
            try:
                r = requests.put(url, data=data, headers=hdr, timeout=600)
                code = r.status_code
            except (requests.ConnectionError, requests.Timeout) as exc:
                code, r = -1, None
                say(f"  upload {path.name}: {type(exc).__name__} at byte {offset}")
            if code in (200, 201):
                offset = None
                break
            if code == 308:
                rng = r.headers.get("Range")
                offset = int(rng.split("-")[1]) + 1 if rng else 0
                failures = 0
                continue
            failures += 1
            if failures > 8:
                raise KaggleError(f"upload {path.name}: giving up at byte {offset} (HTTP {code})")
            time.sleep(min(10 * failures, 120))
            offset = _gcs_offset(url, size)
    if not quiet:
        dt = max(time.time() - t0, 1e-6)
        say(f"  uploaded {path.name}: {size / 1e6:.1f} MB in {dt:.0f} s ({size / 1e6 / dt:.1f} MB/s)")
    return token


# --- datasets ---------------------------------------------------------------------------

def dataset_status(own: str, slug: str) -> str | None:
    """Status name of the dataset's current version (e.g. READY), None if it does not exist."""
    try:
        d = rpc("datasets.DatasetApiService", "GetDatasetStatus",
                {"ownerSlug": own, "datasetSlug": slug})
    except KaggleError as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            return None
        raise
    return str(d.get("status", "")) or "UNKNOWN"


def dataset_info(own: str, slug: str) -> dict | None:
    try:
        return rpc("datasets.DatasetApiService", "GetDataset", {"ownerSlug": own, "datasetSlug": slug})
    except KaggleError as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            return None
        raise


def wait_dataset_ready(own: str, slug: str, timeout_s: float = 3600, poll_s: float = 20) -> str:
    t0 = time.time()
    last = None
    while True:
        st = dataset_status(own, slug)
        if st != last:
            say(f"  dataset {own}/{slug}: {st} ({time.time() - t0:.0f} s)")
            last = st
        if st in ("READY", "FAILED", "DELETED"):
            return st
        if time.time() - t0 > timeout_s:
            raise KaggleError(f"dataset {own}/{slug} not ready after {timeout_s:.0f} s (status {st})")
        time.sleep(poll_s)


# --- kernels ----------------------------------------------------------------------------

def save_kernel(own: str, slug: str, title: str, notebook: dict, *, datasets: list[str],
                competitions: list[str], kernels: list[str] | None = None, private: bool = True,
                gpu: bool = False, internet: bool = False,
                timeout_s: int | None = None) -> dict:
    """Push a notebook (creates the kernel or a new version) and start its run."""
    nb = json.loads(json.dumps(notebook))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    body: dict[str, Any] = {
        "slug": f"{own}/{slug}", "newTitle": title, "text": json.dumps(nb),
        "language": "python", "kernelType": "notebook", "isPrivate": bool(private),
        "enableGpu": bool(gpu), "enableTpu": False, "enableInternet": bool(internet),
        "datasetDataSources": list(datasets), "competitionDataSources": list(competitions),
        "kernelDataSources": list(kernels or []), "modelDataSources": [], "categoryIds": [],
    }
    if timeout_s:
        body["sessionTimeoutSeconds"] = int(timeout_s)
    d = rpc("kernels.KernelsApiService", "SaveKernel", body, timeout=300)
    problems = {k: d.get(k) for k in ("error", "invalidTags", "invalidDatasetSources",
                                       "invalidCompetitionSources", "invalidKernelSources")
                if d.get(k)}
    if problems:
        raise KaggleError(f"SaveKernel {own}/{slug}: {problems}")
    return d


def kernel_status(own: str, slug: str) -> tuple[str, str]:
    d = rpc("kernels.KernelsApiService", "GetKernelSessionStatus",
            {"userName": own, "kernelSlug": slug})
    return str(d.get("status", "")), str(d.get("failureMessage", "") or "")


def wait_kernel(own: str, slug: str, timeout_s: float = 4 * 3600, poll_s: float = 30) -> str:
    t0 = time.time()
    last = None
    while True:
        try:
            st, msg = kernel_status(own, slug)
        except KaggleError as exc:  # transient errors while the session starts
            st, msg = "UNKNOWN", str(exc)
        if st != last:
            say(f"  kernel {own}/{slug}: {st} after {time.time() - t0:.0f} s {msg}".rstrip())
            last = st
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            return st
        if time.time() - t0 > timeout_s:
            raise KaggleError(f"kernel {own}/{slug} still {st} after {timeout_s:.0f} s")
        time.sleep(poll_s)


def kernel_output(own: str, slug: str, dest: Path) -> list[Path]:
    """Download the latest run's output files and log into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    token = ""
    log_text = None
    while True:
        body = {"userName": own, "kernelSlug": slug, "pageSize": 100}
        if token:
            body["pageToken"] = token
        d = rpc("kernels.KernelsApiService", "ListKernelSessionOutput", body)
        if log_text is None:
            log_text = d.get("log") or ""
        for f in d.get("files", []) or []:
            name = f.get("fileName") or ""
            if not name or ".." in Path(name).parts or name.startswith("/"):
                continue
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            r = _retrying(lambda: requests.get(f["url"], timeout=600), f"download {name}")
            if r.status_code >= 400:
                raise KaggleError(f"download {name}: HTTP {r.status_code}")
            out.write_bytes(r.content)
            got.append(out)
        token = d.get("nextPageToken") or ""
        if not token:
            break
    logp = dest / f"{slug}.log"
    logp.write_text(log_text or "")
    got.append(logp)
    return got


def log_text_lines(log_json: str) -> list[str]:
    """Kaggle run logs are a JSON list of {stream_name, time, data}; return the text lines."""
    try:
        items = json.loads(log_json)
    except ValueError:
        return log_json.splitlines()
    out: list[str] = []
    for it in items if isinstance(items, list) else []:
        out.extend(str(it.get("data", "")).splitlines())
    return out


# --- competitions -----------------------------------------------------------------------

def list_submissions(competition: str = COMPETITION) -> list[dict]:
    subs: list[dict] = []
    token = ""
    for _ in range(50):
        body: dict[str, Any] = {"competitionName": competition, "pageSize": 100}
        if token:
            body["pageToken"] = token
        d = rpc("competitions.CompetitionApiService", "ListSubmissions", body)
        subs.extend(d.get("submissions", []) or [])
        token = d.get("nextPageToken") or ""
        if not token:
            break
    return subs


def create_code_submission(kernel_owner: str, kernel_slug: str, version: int, message: str,
                           file_name: str = "submission.csv",
                           competition: str = COMPETITION) -> dict:
    return rpc("competitions.CompetitionApiService", "CreateCodeSubmission", {
        "competitionName": competition, "kernelOwner": kernel_owner, "kernelSlug": kernel_slug,
        "kernelVersion": int(version), "fileName": file_name,
        "submissionDescription": message}, timeout=300)


def die(msg: str, code: int = 1) -> None:
    say(f"ERROR: {msg}")
    sys.exit(code)

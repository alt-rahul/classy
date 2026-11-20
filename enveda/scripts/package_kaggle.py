"""Kaggle packaging (WP14): generate, push and fetch the offline inference notebook.

Subcommands
-----------
notebook  write ``notebooks/kaggle_inference.ipynb`` for one config::

              python scripts/package_kaggle.py notebook --config b0_npdb

push      regenerate the notebook and push it as a PRIVATE kernel (internet off, CPU,
          competition data + code + artefact + wheel datasets attached), optionally
          wait for the run and download its output::

              python scripts/package_kaggle.py push --config b0_npdb --wait --fetch

          Prints the kernel version number that ``scripts/kaggle_submit.py`` needs.
output    download the latest run's output and log of a kernel::

              python scripts/package_kaggle.py output --slug casmi-inference

probe     push the environment probe kernel (mount paths, Python, CPUs, RAM, GPU,
          library versions, offline RDKit install) and fetch its log::

              python scripts/package_kaggle.py probe

The datasets the notebook mounts are listed in ``ARTEFACTS`` below (artefact subdir under
``CASMI_ARTEFACTS`` -> dataset slug). Later WPs add theirs there (or pass ``--artefact
SUBDIR=SLUG``) after publishing them with ``scripts/kaggle_publish.py``. The notebook
finds every dataset by its slug under ``/kaggle/input``, so it does not depend on
Kaggle's mount layout; it builds a symlink tree ``/kaggle/working/casmi_run/artefacts``
that ``casmi.paths`` reads through ``CASMI_ARTEFACTS`` and removes it at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_common as kc  # noqa: E402

ROOT = kc.ENV_ROOT
NOTEBOOK = ROOT / "notebooks" / "kaggle_inference.ipynb"
PROBE_NOTEBOOK = ROOT / "dist" / "kaggle" / "probe.ipynb"
KERNEL_SLUG = "casmi-inference"
KERNEL_TITLE = "casmi inference"
PROBE_SLUG = "casmi-probe"
CODE_DATASET = "casmi-code"
# Artefact subdirectory (relative to CASMI_ARTEFACTS) -> dataset slug of the owner.
ARTEFACTS: dict[str, str] = {
    "processed/db/np": "casmi-db-np",
    "processed/db/pubchem": "casmi-db-pubchem",
}
# Extra wheel datasets ("owner/slug"); the RDKit wheel is always attached.
WHEEL_DATASETS: list[str] = []


# --- notebook text ----------------------------------------------------------------------

def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n")}


def _nb(cells: list[dict]) -> dict:
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


FIND_INPUT = r'''
import os, sys, time, json, glob, shutil, hashlib, subprocess, traceback
from pathlib import Path

INPUT = Path("/kaggle/input")


def find_dataset(slug, owner=None):
    """Mount directory of an attached dataset, found by slug (any Kaggle mount layout)."""
    for cand in ([INPUT / "datasets" / owner / slug] if owner else []) + [INPUT / slug]:
        if cand.is_dir():
            return cand
    hits = []
    for dirpath, dirnames, _ in os.walk(INPUT):
        depth = len(Path(dirpath).relative_to(INPUT).parts)
        if depth >= 4:
            dirnames[:] = []
            continue
        for d in sorted(dirnames):
            if d.lower() == slug.lower():
                hits.append(Path(dirpath) / d)
    if owner:
        hits.sort(key=lambda p: (owner not in p.parts, len(p.parts), str(p)))
    return hits[0] if hits else None


def find_test(competition):
    """The competition's test.parquet (replaced by the hidden test on a scoring rerun)."""
    comp = find_dataset(competition)
    if comp is not None:
        hits = sorted(comp.rglob("test.parquet"))
        if hits:
            return hits[0]
    hits = sorted(INPUT.rglob("test.parquet"))
    if not hits:
        raise FileNotFoundError("no test.parquet under /kaggle/input")
    return hits[0]
'''


def inference_notebook(config: str, owner: str, artefacts: dict[str, str], wheels: list[str],
                       commit: str) -> dict:
    cfg_cell = f'''
# ---- the only settings ---------------------------------------------------------------
from pathlib import Path
CONFIG = {config!r}                     # configs/<CONFIG>.yaml in the code dataset
OWNER = {owner!r}
COMPETITION = {kc.COMPETITION!r}
CODE_DATASET = {CODE_DATASET!r}
ARTEFACTS = {json.dumps(artefacts, indent=4, sort_keys=True)}  # subdir under CASMI_ARTEFACTS -> dataset
RDKIT_WHEEL_DATASET = {kc.RDKIT_WHEEL_DATASET!r}
WHEEL_DATASETS = {wheels!r}             # extra wheel datasets, installed with --no-index
WORK = Path("/kaggle/working")
RUN = WORK / "casmi_run"               # CASMI_ROOT: logs and scratch
SUBMISSION = WORK / "submission.csv"
FALLBACK_SMILES = "Oc1ccccc1"          # only if the whole pipeline cannot run
'''
    setup = r'''
T0 = time.time()
print("python", sys.version.split()[0], "| cpus", os.cpu_count())
TEST = find_test(COMPETITION)  # without it no submission is possible at all
print("test:", TEST)
CODE = None
SETUP_ERROR = None


def pip_install(wheel_dir, *reqs):
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--quiet",
           f"--find-links={wheel_dir}", *reqs]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(" ".join(cmd[3:]), "->", r.returncode, r.stderr[-800:])
    return r.returncode == 0


def setup():
    global CODE
    # RDKit from the metric's wheel dataset (internet is off).
    try:
        import rdkit  # noqa: F401
    except ImportError:
        owner, slug = RDKIT_WHEEL_DATASET.split("/")
        wd = find_dataset(slug, owner)
        print("rdkit wheel dir:", wd)
        pip_install(wd, "rdkit==2026.3.3")
    for ref in WHEEL_DATASETS:
        owner, slug = ref.split("/")
        wd = find_dataset(slug, owner)
        wheels = sorted(str(p) for p in wd.rglob("*.whl")) if wd else []
        if wheels:
            pip_install(wd, *wheels)
    import importlib
    importlib.invalidate_caches()
    import rdkit
    print("rdkit", rdkit.__version__)

    # Code dataset on sys.path (and PYTHONPATH for worker processes).
    CODE = find_dataset(CODE_DATASET, OWNER)
    if CODE is None or not (CODE / "casmi" / "__init__.py").is_file():
        raise FileNotFoundError(f"code dataset {CODE_DATASET} not found ({CODE})")
    sys.path.insert(0, str(CODE))
    os.environ["PYTHONPATH"] = str(CODE) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Artefact tree: symlinks to the mounted datasets, plus writable dirs (mkey cache).
    art = RUN / "artefacts"
    if art.exists():
        shutil.rmtree(art)
    for sub, slug in sorted(ARTEFACTS.items()):
        src = find_dataset(slug, OWNER)
        if src is None:
            print(f"WARNING: dataset {slug} not attached; {sub} missing")
            continue
        dst = art / sub
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=True)
        print(f"{sub:24s} -> {src}")
    (art / "processed" / "cache").mkdir(parents=True, exist_ok=True)
    (art / "models").mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "CASMI_ROOT": str(RUN),
        "CASMI_ARTEFACTS": str(art),
        "CASMI_RAW": str(TEST.parent),
        "CASMI_CONFIGS": str(CODE / "configs"),
        "NUMBA_CACHE_DIR": str(WORK / "numba_cache"),
    })
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


try:
    setup()
except Exception:
    SETUP_ERROR = traceback.format_exc()
    print(SETUP_ERROR)
'''
    run = r'''
def write_fallback(reason):
    """A valid file no matter what: one guess per test molecule."""
    import pandas as pd
    ids = sorted(set(pd.read_parquet(TEST, columns=["molecule_id"])["molecule_id"].astype(str)))
    pd.DataFrame({"molecule_id": ids, "smiles": [FALLBACK_SMILES] * len(ids)}).to_csv(
        SUBMISSION, index=False, lineterminator="\n")
    print(f"FALLBACK submission written ({reason}): {len(ids)} molecules")


test_df = None
cfg = None
try:
    if SETUP_ERROR:
        raise RuntimeError("setup failed")
    from casmi import io, log, paths, pipeline
    from casmi.config import load_config
    lg = log.get_logger(f"kaggle_{CONFIG}")
    print("paths:", json.dumps(paths.as_dict(), indent=1))
    cfg = load_config(CODE / "configs" / f"{CONFIG}.yaml")
    print("config overrides:", cfg.overrides())
    pipeline.limit_threads(cfg.n_jobs)
    test_df = io.read_test(TEST)
    res = pipeline.run(test_df, cfg)
    io.write_submission(res.preds, SUBMISSION, test_df=test_df, max_guesses=cfg.top_k)
    rep = RUN / "logs" / f"kaggle_{CONFIG}_molecules.csv"
    rep.parent.mkdir(parents=True, exist_ok=True)
    res.molecules.to_csv(rep, index=False, lineterminator="\n")
    print(f"pipeline done in {res.wall_s:.1f} s; peak RSS {log.peak_rss_mb():.0f} MB")
except Exception:
    traceback.print_exc()
    write_fallback("pipeline failed")
'''
    validate = r'''
try:
    from casmi import io
    report = io.validate_submission(SUBMISSION, test_df=test_df,
                                    max_guesses=cfg.top_k if cfg is not None else 25)
    print("validate_submission:", report)
except Exception:
    traceback.print_exc()
    write_fallback("validation failed")
    import pandas as pd
    chk = pd.read_csv(SUBMISSION)
    assert chk["molecule_id"].is_unique and chk["smiles"].notna().all()

shutil.rmtree(RUN / "artefacts", ignore_errors=True)  # symlinks into /kaggle/input
data = SUBMISSION.read_bytes()
print("submission sha256", hashlib.sha256(data).hexdigest(), f"{len(data)} bytes")
print(data.decode()[:600])
print(f"total runtime {time.time() - T0:.1f} s")
'''
    head = (f"# CASMI 2026 inference ({config})\n\nGenerated by `scripts/package_kaggle.py` from "
            f"git commit `{commit}`. Offline: RDKit from `{kc.RDKIT_WHEEL_DATASET}`, code from "
            f"`{owner}/{CODE_DATASET}`, artefacts from the datasets in `ARTEFACTS`. Writes "
            "`/kaggle/working/submission.csv`. Edit only the settings cell.")
    return _nb([_md(head), _code(cfg_cell), _code(FIND_INPUT), _code(setup), _code(run),
                _code(validate)])


def probe_notebook(owner: str) -> dict:
    src = FIND_INPUT + f'''
OWNER = {owner!r}
RDKIT_WHEEL_DATASET = {kc.RDKIT_WHEEL_DATASET!r}
CODE_DATASET = {CODE_DATASET!r}
COMPETITION = {kc.COMPETITION!r}
''' + r'''
import platform, importlib, multiprocessing as mp
out = {}
print("== python", sys.version, platform.platform())
out["python"] = sys.version

print("== /kaggle/input tree")
for dirpath, dirnames, filenames in os.walk(INPUT, followlinks=False):
    rel = Path(dirpath).relative_to(INPUT)
    depth = len(rel.parts)
    dirnames.sort()
    size = sum((Path(dirpath) / f).stat().st_size for f in filenames)
    print(f"{'  ' * depth}{dirpath}/  [{len(filenames)} files, {size / 1e6:.1f} MB]")
    for f in sorted(filenames)[:6]:
        print(f"{'  ' * depth}  {f}")
    if len(filenames) > 6:
        print(f"{'  ' * depth}  ... {len(filenames) - 6} more")
    if depth >= 6:
        dirnames[:] = []

print("== cpu")
out["cpu_count"] = os.cpu_count()
out["affinity"] = len(os.sched_getaffinity(0))
for f in ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
          "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
    if os.path.exists(f):
        out[f] = open(f).read().strip()
print(subprocess.run("lscpu | head -20", shell=True, capture_output=True, text=True).stdout)
mem = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo") if ":" in l}
out["MemTotal"], out["MemAvailable"] = mem.get("MemTotal"), mem.get("MemAvailable")
print("== gpu", shutil.which("nvidia-smi"))
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
for d in ["/kaggle/working", "/kaggle/tmp", "/kaggle/temp", "/tmp"]:
    if os.path.isdir(d):
        u = shutil.disk_usage(d)
        out[f"disk {d}"] = f"total {u.total / 1e9:.1f} GB, free {u.free / 1e9:.1f} GB, writable {os.access(d, os.W_OK)}"
out["start_methods"] = mp.get_all_start_methods()
out["env_names"] = sorted(k for k in os.environ if k.startswith("KAGGLE"))
out["KAGGLE_KERNEL_RUN_TYPE"] = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")

print("== modules")
mods = {}
for m in ["numpy", "pandas", "pyarrow", "scipy", "numba", "llvmlite", "lightgbm", "polars", "yaml",
          "sklearn", "duckdb", "torch", "requests", "PIL", "rdkit"]:
    try:
        mods[m] = getattr(importlib.import_module(m), "__version__", "?")
    except Exception as e:
        mods[m] = f"MISSING ({type(e).__name__})"
out["modules_before"] = mods

print("== rdkit install")
owner, slug = RDKIT_WHEEL_DATASET.split("/")
wd = find_dataset(slug, owner)
out["rdkit_wheel_dir"] = str(wd)
out["rdkit_wheel_files"] = sorted(os.listdir(wd)) if wd else None
t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    f"--find-links={wd}", "rdkit==2026.3.3"], capture_output=True, text=True)
out["rdkit_pip_rc"] = r.returncode
out["rdkit_pip_s"] = round(time.time() - t, 1)
print(r.stdout[-1500:], r.stderr[-1500:])
importlib.invalidate_caches()
try:
    import rdkit
    from rdkit import Chem
    out["rdkit"] = rdkit.__version__
    out["rdkit_test"] = Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))
except Exception as e:
    out["rdkit"] = f"FAILED {e}"

print("== code dataset")
code = find_dataset(CODE_DATASET, OWNER)
out["code_dir"] = str(code)
if code is not None:
    sys.path.insert(0, str(code))
    os.environ["NUMBA_CACHE_DIR"] = "/kaggle/working/numba_cache"
    try:
        import casmi, casmi.paths
        out["casmi"] = casmi.__file__
        from casmi import pipeline  # noqa: F401  (imports numba kernels lazily)
        out["casmi_pipeline_import"] = "ok"
    except Exception as e:
        out["casmi"] = f"FAILED {type(e).__name__}: {e}"
for slug in ["casmi-db-np", "casmi-db-pubchem"]:
    out[f"dataset {slug}"] = str(find_dataset(slug, OWNER))
try:
    out["test_parquet"] = str(find_test(COMPETITION))
    import pandas as pd
    t = pd.read_parquet(out["test_parquet"])
    out["test_shape"] = list(t.shape)
    out["test_molecules"] = int(t["molecule_id"].nunique())
except Exception as e:
    out["test_parquet"] = f"FAILED {e}"

print("== numba")
try:
    import numba
    @numba.njit(cache=True)
    def f(n):
        s = 0.0
        for i in range(n):
            s += i
        return s
    t = time.time(); f(10); out["numba_compile_s"] = round(time.time() - t, 2)
except Exception as e:
    out["numba"] = f"FAILED {e}"

print("== summary")
print(json.dumps(out, indent=1, default=str))
Path("/kaggle/working/probe.json").write_text(json.dumps(out, indent=1, default=str))
'''
    return _nb([_md("# casmi environment probe (WP14)"), _code(src)])


# --- commands ---------------------------------------------------------------------------

def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _artefacts(extra: list[str]) -> dict[str, str]:
    a = dict(ARTEFACTS)
    for spec in extra:
        sub, _, slug = spec.partition("=")
        if not sub or not slug:
            kc.die(f"--artefact {spec}: expected SUBDIR=SLUG")
        a[sub.strip("/")] = slug
    return a


def write_notebook(config: str, owner: str, artefacts: dict[str, str], wheels: list[str],
                   out: Path = NOTEBOOK) -> Path:
    if not (ROOT / "configs" / f"{config}.yaml").is_file():
        kc.die(f"configs/{config}.yaml does not exist")
    nb = inference_notebook(config, owner, artefacts, wheels, _commit())
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    for bad in ("/home/", "/root/", "/tmp/"):
        if bad in text:
            kc.die(f"notebook would contain an absolute path outside /kaggle/ ({bad})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    kc.say(f"wrote {out.relative_to(ROOT)} (config {config}, owner {owner}, commit {_commit()})")
    return out


def push(nb: dict, slug: str, title: str, owner: str, datasets: list[str], wait: bool,
         fetch: bool, timeout_s: float) -> int:
    res = kc.save_kernel(owner, slug, title, nb, datasets=datasets, competitions=[kc.COMPETITION],
                         private=True, gpu=False, internet=False)
    ver = res.get("versionNumber")
    kc.say(f"pushed {res.get('ref')} version {ver} {res.get('url', '')}")
    kc.say(f"  datasets: {datasets}; competition: {kc.COMPETITION}; private, internet off, CPU")
    if not wait:
        return 0
    t0 = time.time()
    st = kc.wait_kernel(owner, slug, timeout_s=timeout_s)
    kc.say(f"kernel {owner}/{slug} v{ver}: {st} after {time.time() - t0:.0f} s")
    if fetch:
        dest = ROOT / "dist" / "kaggle" / "output" / f"{slug}_v{ver}"
        got = kc.kernel_output(owner, slug, dest)
        kc.say(f"output ({len(got)} files) in {dest.relative_to(ROOT)}: {[p.name for p in got]}")
    return 0 if st == "COMPLETE" else 1


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("notebook", "push"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True, help="config name in configs/ (no .yaml)")
        s.add_argument("--owner", default=None)
        s.add_argument("--artefact", action="append", default=[], metavar="SUBDIR=SLUG",
                       help="extra artefact dataset (repeatable)")
        s.add_argument("--wheels", action="append", default=[], metavar="OWNER/SLUG",
                       help="extra wheel dataset (repeatable)")
        s.add_argument("--out", type=Path, default=NOTEBOOK)
        if name == "push":
            s.add_argument("--slug", default=KERNEL_SLUG)
            s.add_argument("--title", default=KERNEL_TITLE)
            s.add_argument("--wait", action="store_true")
            s.add_argument("--fetch", action="store_true", help="download the output after --wait")
            s.add_argument("--timeout", type=float, default=6 * 3600)
    s = sub.add_parser("output")
    s.add_argument("--slug", default=KERNEL_SLUG)
    s.add_argument("--owner", default=None)
    s.add_argument("--dest", type=Path, default=None)
    s = sub.add_parser("probe")
    s.add_argument("--owner", default=None)
    s.add_argument("--no-wait", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse(argv)
    own = kc.owner(a.owner)
    if a.cmd in ("notebook", "push"):
        arts = _artefacts(a.artefact)
        wheels = list(WHEEL_DATASETS) + list(a.wheels)
        path = write_notebook(a.config, own, arts, wheels, a.out)
        if a.cmd == "notebook":
            return 0
        datasets = sorted({f"{own}/{CODE_DATASET}", kc.RDKIT_WHEEL_DATASET,
                           *(f"{own}/{s}" for s in arts.values()), *wheels})
        nb = json.loads(path.read_text(encoding="utf-8"))
        return push(nb, a.slug, a.title, own, datasets, a.wait, a.fetch, a.timeout)
    if a.cmd == "output":
        dest = a.dest or ROOT / "dist" / "kaggle" / "output" / a.slug
        got = kc.kernel_output(own, a.slug, dest)
        kc.say(f"{len(got)} files in {dest}: {[p.name for p in got]}")
        return 0
    if a.cmd == "probe":
        nb = probe_notebook(own)
        PROBE_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
        PROBE_NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")
        datasets = sorted({f"{own}/{CODE_DATASET}", f"{own}/casmi-db-np", kc.RDKIT_WHEEL_DATASET})
        rc = push(nb, PROBE_SLUG, PROBE_SLUG, own, datasets, not a.no_wait, not a.no_wait, 3600)
        log = ROOT / "dist" / "kaggle" / "output"
        for p in sorted(log.glob(f"{PROBE_SLUG}_v*/{PROBE_SLUG}.log"))[-1:]:
            lines = kc.log_text_lines(p.read_text())
            p.with_suffix(".txt").write_text("\n".join(lines) + "\n")
            kc.say(f"log text: {p.with_suffix('.txt').relative_to(ROOT)} ({len(lines)} lines)")
        return rc
    return 2


if __name__ == "__main__":
    sys.exit(main())

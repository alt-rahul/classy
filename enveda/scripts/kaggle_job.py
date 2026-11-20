"""Generic Kaggle job runner: push a PRIVATE kernel (CPU by default, internet off) that mounts
a code dataset plus listed datasets and kernel outputs, runs shell commands, and keeps their
output in /kaggle/working; then wait, fetch the output, or publish part of it as a private
dataset version (ROADMAP R3 / R13; conventions of scripts/kaggle_common.py and
scripts/package_kaggle.py).

Subcommands::

    # publish the code dataset (src/casmi -> casmi/, scripts/, configs/, pyproject.toml)
    python scripts/kaggle_job.py code --code casmi-job-code --message "va build"

    # push a job (writes dist/kaggle/jobs/kaggle_job_<slug>.ipynb, then SaveKernel)
    python scripts/kaggle_job.py push --slug casmi-va-build-0 --code casmi-job-code \\
        --dataset casmi-va-inputs --dataset casmi-db-pubchem \\
        --cmd 'python "$CODE/scripts/build_va.py" build --shard 0 --n-shards 4 ...' \\
        --timeout-hours 12 [--wait --fetch] [--publish casmi-db-va --publish-subdir va]

    python scripts/kaggle_job.py status --slug S [--slug S2 ...]
    python scripts/kaggle_job.py wait --slug S [--slug S2 ...] [--fetch] [--publish NAME ...]
    python scripts/kaggle_job.py fetch --slug S [--dest DIR] [--include GLOB ...]
    python scripts/kaggle_job.py publish --slug S --name DATASET [--publish-subdir DIR] [--message M]

Inside the kernel each command runs under ``bash -c`` in /kaggle/working with these
environment variables: ``CODE`` (code dataset mount; also on PYTHONPATH, so ``import casmi``
works), ``WORK`` (/kaggle/working), ``INPUT`` (/kaggle/input), ``DS_<SLUG>`` for every
dataset and ``KS_<SLUG>`` for every kernel-output source (slug upper-cased, ``-`` -> ``_``),
``CASMI_ROOT`` (/kaggle/working/casmi_run), ``NUMBA_CACHE_DIR``, plus ``--env`` pairs.
RDKit is installed from the metric wheel dataset when missing. stdout and stderr go to the
notebook log and to ``/kaggle/working/job.log``; ``/kaggle/working/job_status.json`` holds
each command's exit code and seconds (the notebook itself never raises, so the output is
always saved; ``wait`` and ``fetch`` report failed commands).

A push that Kaggle refuses because the concurrent-session limit is reached is retried every
``--retry-min`` minutes for up to ``--retry-hours`` hours. Downloads are streamed to disk
(large outputs never sit in memory) and files already present with the same size are
skipped. Credentials are never printed (see kaggle_common.py).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_common as kc  # noqa: E402
from package_kaggle import FIND_INPUT, _code, _md, _nb  # noqa: E402

ROOT = kc.ENV_ROOT
NB_DIR = ROOT / "dist" / "kaggle" / "jobs"   # generated job notebooks (untracked)
OUT_DIR = ROOT / "dist" / "kaggle" / "output"
DEFAULT_CODE = "casmi-code"
CODE_SOURCES = ("src/casmi=casmi", "scripts=scripts", "configs=configs", "pyproject.toml")
BUSY_WORDS = ("concurrent", "maximum", "limit", "quota", "too many", "exceeded")


def env_name(prefix: str, slug: str) -> str:
    """``DS_CASMI_DB_NP`` style variable name of a dataset or kernel slug."""
    s = slug.split("/")[-1].upper().replace("-", "_").replace(".", "_")
    return f"{prefix}_{s}"


def _full(own: str, ref: str) -> str:
    return ref if "/" in ref else f"{own}/{ref}"


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


RUN_CELL = r'''
T0 = time.time()
WORK = Path("/kaggle/working")
print("python", sys.version.split()[0], "| cpus", os.cpu_count(), "| job", JOB["slug"], "| commit", JOB["commit"])
try:
    import rdkit  # noqa: F401
except ImportError:
    owner, slug = JOB["rdkit_wheels"].split("/")
    wd = find_dataset(slug, owner)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--quiet",
                        f"--find-links={wd}", "rdkit==2026.3.3"], capture_output=True, text=True)
    print("rdkit install", r.returncode, r.stderr[-500:])
own_code, code_slug = JOB["code"].split("/")
CODE = find_dataset(code_slug, own_code)
print("code", CODE)
env = dict(os.environ)
env.update({"CODE": str(CODE or ""), "WORK": str(WORK), "INPUT": str(INPUT),
            "PYTHONPATH": os.pathsep.join([str(CODE or "")] + [p for p in [os.environ.get("PYTHONPATH", "")] if p]),
            "CASMI_ROOT": str(WORK / "casmi_run"), "NUMBA_CACHE_DIR": str(WORK / "numba_cache"),
            "PYTHONUNBUFFERED": "1"})
for ref in JOB["datasets"]:
    o, s = ref.split("/")
    d = find_dataset(s, o)
    env[JOB["ds_env"][ref]] = str(d or "")
    print("dataset", ref, "->", d)
for ref in JOB["kernels"]:
    o, s = ref.split("/")
    d = find_dataset(s, o)
    env[JOB["ks_env"][ref]] = str(d or "")
    print("kernel output", ref, "->", d)
env.update(JOB["env"])
status = {"slug": JOB["slug"], "commit": JOB["commit"], "commands": []}
logf = open(WORK / "job.log", "a", buffering=1)
for cmd in JOB["cmds"]:
    t = time.time()
    print(f"[{(t - T0) / 60:.1f} min] $ {cmd}", flush=True)
    logf.write(f"$ {cmd}\n")
    p = subprocess.Popen(["bash", "-c", cmd], env=env, cwd=str(WORK), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in p.stdout:
        logf.write(line)
        print(line, end="", flush=True)
    rc = p.wait()
    status["commands"].append({"cmd": cmd, "rc": rc, "seconds": round(time.time() - t, 1)})
    (WORK / "job_status.json").write_text(json.dumps(status, indent=1))
    print(f"[{(time.time() - T0) / 60:.1f} min] exit code {rc}", flush=True)
    if rc != 0 and JOB["stop_on_error"]:
        break
logf.close()
status["seconds"] = round(time.time() - T0, 1)
status["ok"] = all(c["rc"] == 0 for c in status["commands"]) and len(status["commands"]) == len(JOB["cmds"])
(WORK / "job_status.json").write_text(json.dumps(status, indent=1))
for p in sorted(WORK.rglob("*")):
    if p.is_file() and "numba_cache" not in p.parts:
        print(f"{p.stat().st_size / 1e6:10.2f} MB  {p.relative_to(WORK)}")
print("job", "OK" if status["ok"] else "FAILED", f"in {(time.time() - T0) / 60:.1f} min")
'''


def job_notebook(job: dict[str, Any]) -> dict:
    settings = (f"# ---- settings (generated by scripts/kaggle_job.py at {job['commit']}) ----\n"
                f"import json\nJOB = json.loads({json.dumps(json.dumps(job, sort_keys=True))})\n"
                f"OWNER = JOB['owner']\n")
    return _nb([_md(f"# casmi job `{job['slug']}`\nGenerated by `scripts/kaggle_job.py`; commands:\n\n"
                    + "\n".join(f"    {c}" for c in job["cmds"])),
                _code(settings), _code(FIND_INPUT), _code(RUN_CELL)])


def make_job(own: str, args: argparse.Namespace) -> dict[str, Any]:
    datasets = [_full(own, d) for d in args.dataset]
    kernels = [_full(own, k) for k in args.kernel_source]
    code = _full(own, args.code)
    env = {}
    for it in args.env:
        k, _, v = it.partition("=")
        if not k:
            kc.die(f"--env {it}: expected KEY=VALUE")
        env[k] = v
    return {"slug": args.slug, "owner": own, "code": code, "datasets": datasets, "kernels": kernels,
            "ds_env": {d: env_name("DS", d) for d in datasets},
            "ks_env": {k: env_name("KS", k) for k in kernels},
            "cmds": list(args.cmd), "env": env, "stop_on_error": not args.keep_going,
            "rdkit_wheels": kc.RDKIT_WHEEL_DATASET, "commit": _commit()}


def write_notebook(job: dict[str, Any]) -> tuple[dict, Path]:
    nb = job_notebook(job)
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    for bad in ("/home/", "/root/"):
        if bad in text:
            kc.die(f"notebook would contain a local absolute path ({bad}); use $CODE, $WORK, $DS_*")
    path = NB_DIR / f"kaggle_job_{job['slug']}.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return nb, path


def push(own: str, job: dict[str, Any], args: argparse.Namespace) -> int | None:
    nb, path = write_notebook(job)
    kc.say(f"wrote {path.relative_to(ROOT)}")
    datasets = job["datasets"] + [job["code"]]
    if kc.RDKIT_WHEEL_DATASET not in datasets:
        datasets.append(kc.RDKIT_WHEEL_DATASET)
    datasets = list(dict.fromkeys(datasets))
    deadline = time.time() + args.retry_hours * 3600
    while True:
        try:
            res = kc.save_kernel(own, job["slug"], args.title or job["slug"].replace("-", " "), nb,
                                 datasets=datasets, competitions=[kc.COMPETITION] if args.competition else [],
                                 kernels=job["kernels"], private=True, gpu=args.gpu, internet=False,
                                 timeout_s=int(args.timeout_hours * 3600))
            break
        except kc.KaggleError as exc:
            busy = any(w in str(exc).lower() for w in BUSY_WORDS)
            if not busy or time.time() > deadline:
                raise
            kc.say(f"  push refused ({str(exc)[:160]}); retry in {args.retry_min:.0f} min")
            time.sleep(args.retry_min * 60)
    ver = res.get("versionNumber")
    kc.say(f"pushed {res.get('ref') or own + '/' + job['slug']} version {ver} {res.get('url', '')}")
    kc.say(f"  datasets {datasets}; kernels {job['kernels']}; private, internet off, "
           f"{'GPU' if args.gpu else 'CPU'}, timeout {args.timeout_hours} h")
    return ver


# --- output -----------------------------------------------------------------------------

def list_output(own: str, slug: str) -> tuple[list[dict], str]:
    files: list[dict] = []
    token, log_text = "", None
    while True:
        body: dict[str, Any] = {"userName": own, "kernelSlug": slug, "pageSize": 200}
        if token:
            body["pageToken"] = token
        d = kc.rpc("kernels.KernelsApiService", "ListKernelSessionOutput", body)
        if log_text is None:
            log_text = d.get("log") or ""
        files.extend(d.get("files", []) or [])
        token = d.get("nextPageToken") or ""
        if not token:
            break
    return files, log_text or ""


def _download(url: str, out: Path, size: int | None) -> int:
    tmp = out.with_name(out.name + ".part")
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                if r.status_code >= 400:
                    raise kc.KaggleError(f"download {out.name}: HTTP {r.status_code}")
                n = 0
                with tmp.open("wb") as fh:
                    for block in r.iter_content(chunk_size=1 << 22):
                        fh.write(block)
                        n += len(block)
            if size is not None and n != size:
                raise kc.KaggleError(f"download {out.name}: {n} bytes, expected {size}")
            tmp.replace(out)
            return n
        except (requests.ConnectionError, requests.Timeout, kc.KaggleError) as exc:
            if attempt == 5:
                raise kc.KaggleError(f"download {out.name}: {exc}") from None
            kc.say(f"  {out.name}: {type(exc).__name__}, retry")
            time.sleep(min(10 * (attempt + 1), 60))
    return 0


def fetch(own: str, slug: str, dest: Path | None = None, include: list[str] | None = None) -> Path:
    """Stream the latest run's output files (matching ``include`` globs, default all) and the
    log into ``dest`` (default dist/kaggle/output/<slug>/)."""
    dest = Path(dest) if dest is not None else OUT_DIR / slug
    dest.mkdir(parents=True, exist_ok=True)
    files, log_text = list_output(own, slug)
    t0, tot, n_new = time.time(), 0, 0
    for f in files:
        name = f.get("fileName") or ""
        if not name or ".." in Path(name).parts or name.startswith("/"):
            continue
        if include and not any(fnmatch.fnmatch(name, pat) for pat in include):
            continue
        out = dest / name
        out.parent.mkdir(parents=True, exist_ok=True)
        size = f.get("totalBytes") or f.get("size")
        size = int(size) if size not in (None, "") else None
        if out.exists() and size is not None and out.stat().st_size == size:
            continue
        tot += _download(f["url"], out, size)
        n_new += 1
    (dest / f"{slug}.log").write_text(log_text)
    lines = kc.log_text_lines(log_text)
    (dest / f"{slug}.txt").write_text("\n".join(lines) + "\n")
    dt = max(time.time() - t0, 1e-6)
    kc.say(f"fetched {n_new} files ({tot / 1e6:.1f} MB, {tot / 1e6 / dt:.1f} MB/s) of {len(files)} into {dest}")
    st = dest / "job_status.json"
    if st.exists():
        s = json.loads(st.read_text())
        bad = [c for c in s.get("commands", []) if c.get("rc") != 0]
        kc.say(f"  job status: {'OK' if s.get('ok') else 'FAILED'}; "
               + "; ".join(f"rc {c['rc']} after {c['seconds']} s: {c['cmd'][:80]}" for c in s.get("commands", [])))
        if bad or not s.get("ok"):
            tail = "\n".join(lines[-30:])
            kc.say(f"  last log lines:\n{tail}")
    return dest


def publish_output(own: str, slug: str, name: str, subdir: str | None, message: str,
                   dest: Path | None = None) -> int:
    import kaggle_publish

    d = fetch(own, slug, dest)
    src = d / subdir if subdir else d
    if not src.is_dir():
        kc.die(f"{src} does not exist in the output of {slug}")
    return kaggle_publish.main(["--name", name, "--src", str(src), "--message", message,
                                "--exclude", "*.log", "--exclude", f"{slug}.txt", "--exclude", "*.part"])


def wait_all(own: str, slugs: list[str], timeout_h: float, poll_s: float) -> dict[str, str]:
    out = {}
    for s in slugs:
        out[s] = kc.wait_kernel(own, s, timeout_s=timeout_h * 3600, poll_s=poll_s)
        kc.say(f"kernel {s}: {out[s]}")
    return out


# --- CLI --------------------------------------------------------------------------------

def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd_name", choices=("code", "notebook", "push", "status", "wait", "fetch", "publish"))
    p.add_argument("--slug", action="append", default=[], help="kernel slug (repeatable for status/wait)")
    p.add_argument("--owner", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--code", default=DEFAULT_CODE, help=f"code dataset slug (default {DEFAULT_CODE})")
    p.add_argument("--publish-code", action="store_true", help="push: publish the code dataset first")
    p.add_argument("--dataset", action="append", default=[], help="attached dataset [owner/]slug")
    p.add_argument("--kernel-source", action="append", default=[], help="attached kernel output [owner/]slug")
    p.add_argument("--competition", action="store_true", help="also attach the competition data")
    p.add_argument("--cmd", action="append", default=[], help="shell command (repeatable, run in order)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--keep-going", action="store_true", help="run later commands after a failure")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--timeout-hours", type=float, default=12.0)
    p.add_argument("--retry-hours", type=float, default=6.0, help="keep retrying a refused push this long")
    p.add_argument("--retry-min", type=float, default=5.0)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll", type=float, default=60.0, help="seconds between status polls")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--dest", type=Path, default=None)
    p.add_argument("--include", action="append", default=[], help="fetch: only files matching the glob")
    p.add_argument("--publish", default=None, metavar="DATASET", help="publish the output as this dataset")
    p.add_argument("--name", default=None, help="publish: dataset slug")
    p.add_argument("--publish-subdir", default=None, help="publish only this output subdirectory")
    p.add_argument("--message", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    own = kc.owner(args.owner)
    if args.cmd_name == "code" or (args.cmd_name == "push" and args.publish_code):
        import kaggle_publish

        argv_pub = ["--name", args.code.split("/")[-1], "--message",
                    args.message or f"code at {_commit()}", "--exclude", "*.ipynb"]
        for s in CODE_SOURCES:
            argv_pub += ["--src", s]
        rc = kaggle_publish.main(argv_pub)
        if args.cmd_name == "code" or rc != 0:
            return rc
    if args.cmd_name in ("notebook", "push"):
        if len(args.slug) != 1 or not args.cmd:
            kc.die("push/notebook need exactly one --slug and at least one --cmd")
        args.slug = args.slug[0]
        job = make_job(own, args)
        if args.cmd_name == "notebook":
            _, path = write_notebook(job)
            kc.say(f"wrote {path.relative_to(ROOT)}")
            return 0
        push(own, job, args)
        if not (args.wait or args.fetch or args.publish):
            return 0
        st = wait_all(own, [args.slug], args.timeout_hours + 1.0, args.poll)[args.slug]
        if args.publish:
            return publish_output(own, args.slug, args.publish, args.publish_subdir,
                                  args.message or f"output of {args.slug}", args.dest)
        if args.fetch:
            fetch(own, args.slug, args.dest, args.include)
        return 0 if st == "COMPLETE" else 1
    if not args.slug:
        kc.die("--slug is required")
    if args.cmd_name == "status":
        for s in args.slug:
            st, msg = kc.kernel_status(own, s)
            kc.say(f"{s}: {st} {msg}".rstrip())
        return 0
    if args.cmd_name == "wait":
        res = wait_all(own, args.slug, args.timeout_hours + 1.0, args.poll)
        for s in args.slug:
            if args.publish:
                publish_output(own, s, args.publish, args.publish_subdir,
                               args.message or f"output of {s}", args.dest)
            elif args.fetch:
                fetch(own, s, (args.dest / s) if args.dest and len(args.slug) > 1 else args.dest, args.include)
        return 0 if all(v == "COMPLETE" for v in res.values()) else 1
    if args.cmd_name == "fetch":
        for s in args.slug:
            fetch(own, s, (args.dest / s) if args.dest and len(args.slug) > 1 else args.dest, args.include)
        return 0
    if args.cmd_name == "publish":
        name = args.name or args.publish
        if not name:
            kc.die("publish needs --name")
        return publish_output(own, args.slug[0], name, args.publish_subdir,
                              args.message or f"output of {args.slug[0]}", args.dest)
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Submit a Kaggle kernel version to the competition, with the user's pacing rules (WP14).

    python scripts/kaggle_submit.py --check
    python scripts/kaggle_submit.py --kernel casmi-inference --version 3 \\
        --config b0_npdb --local-score n/a --message "b0 NP-tier prior"
    python scripts/kaggle_submit.py --status [--wait]

``--check``   lists the competition submissions (Kaggle API) and prints the time of the most
              recent one (UTC), its age, the count in the current UTC day, and whether a
              submission is allowed now.
submit       refuses (exit 3) and prints the seconds to wait if the most recent submission
              is less than 55 minutes old or 5 were already made in the current UTC day
              (every listed submission counts, errored ones included). Otherwise submits the
              kernel version as a code submission (``CompetitionApiService/CreateCodeSubmission``,
              the call behind ``kaggle competitions submit -k OWNER/KERNEL -v VERSION``) and
              appends a row to ``docs/SUBMISSIONS.md``.
``--status``  reads the submissions list and fills status and public score into
              ``docs/SUBMISSIONS.md`` (matched by submission ref); ``--wait`` polls until
              every logged submission is scored or failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_common as kc  # noqa: E402

MIN_GAP_S = 55 * 60
MAX_PER_DAY = 5
LOG = kc.ENV_ROOT / "docs" / "SUBMISSIONS.md"
HEADER = """# Submissions

Every leaderboard submission, appended by `scripts/kaggle_submit.py` (public score filled in by
`scripts/kaggle_submit.py --status`). Rules (docs/RUNBOOK.md, standing orders): at least 55
minutes between submissions, at most 5 per UTC day, always check with `--check` first.
Times are UTC. "kernel" is `owner/slug vN` of the private inference notebook that was run.

| # | submitted (UTC) | kernel | config | local validation | message | status | public score | ref |
|---|---|---|---|---|---|---|---|---|
"""


def _parse_time(s: str) -> dt.datetime:
    s = s.strip().replace("Z", "+00:00")
    m = re.match(r"^(.*T\d\d:\d\d:\d\d)(\.\d+)?(.*)$", s)
    if m:  # trim fractions to microseconds for fromisoformat
        frac = (m.group(2) or "")[:7]
        s = m.group(1) + frac + (m.group(3) or "")
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _subs() -> list[dict]:
    subs = kc.list_submissions()
    for s in subs:
        s["_t"] = _parse_time(s["date"]) if s.get("date") else None
    return sorted([s for s in subs if s["_t"] is not None], key=lambda s: s["_t"], reverse=True)


def pacing(subs: list[dict], now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today = [s for s in subs if s["_t"] >= day0]
    last = subs[0]["_t"] if subs else None
    wait_gap = max(0.0, MIN_GAP_S - (now - last).total_seconds()) if last else 0.0
    wait_day = (day0 + dt.timedelta(days=1) - now).total_seconds() if len(today) >= MAX_PER_DAY else 0.0
    return {"now": now, "last": last, "n_today": len(today), "n_total": len(subs),
            "wait_gap_s": wait_gap, "wait_day_s": wait_day, "wait_s": max(wait_gap, wait_day)}


def _fmt(t: dt.datetime | None) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S UTC") if t else "none"


def check() -> dict:
    subs = _subs()
    p = pacing(subs)
    kc.say(f"now:                  {_fmt(p['now'])}")
    if p["last"]:
        age = (p["now"] - p["last"]).total_seconds()
        kc.say(f"most recent:          {_fmt(p['last'])} ({age / 60:.1f} min ago)")
    else:
        kc.say("most recent:          none")
    kc.say(f"submissions today:    {p['n_today']} of {MAX_PER_DAY} (UTC day); {p['n_total']} in total")
    for s in subs[:5]:
        kc.say(f"  {_fmt(s['_t'])}  ref {s.get('ref')}  {s.get('status') or 'PENDING'}  public "
               f"{s.get('publicScore', '') or '-'}  {s.get('description', '')[:60]}")
    if p["wait_s"] > 0:
        why = "55-minute gap" if p["wait_gap_s"] >= p["wait_day_s"] else "5 per UTC day reached"
        kc.say(f"submit allowed:       NO, wait {int(p['wait_s']) + 1} s ({why})")
    else:
        kc.say("submit allowed:       yes")
    return p


def _cell(s: object) -> str:
    return str(s).replace("|", "/").replace("\n", " ").strip()


def _append_row(t: dt.datetime, kernel: str, config: str, local: str, message: str, ref: str) -> None:
    if not LOG.exists():
        LOG.write_text(HEADER)
    rows = [ln for ln in LOG.read_text().splitlines() if re.match(r"^\| \d+ \|", ln)]
    n = len(rows) + 1
    line = (f"| {n} | {t.strftime('%Y-%m-%d %H:%M:%S')} | {_cell(kernel)} | {_cell(config)} | "
            f"{_cell(local)} | {_cell(message)} | PENDING |  | {_cell(ref)} |")
    text = LOG.read_text()
    if not text.endswith("\n"):
        text += "\n"
    LOG.write_text(text + line + "\n")
    kc.say(f"logged row {n} in {LOG.relative_to(kc.ENV_ROOT)}")


def submit(a: argparse.Namespace) -> int:
    p = check()
    if p["wait_s"] > 0:
        kc.say(f"REFUSED: wait {int(p['wait_s']) + 1} seconds before submitting")
        return 3
    own = kc.owner(a.owner)
    kernel = f"{own}/{a.kernel} v{a.version}"
    if a.dry_run:
        kc.say(f"dry run: would submit {kernel} ({a.config}): {a.message}")
        return 0
    res = kc.create_code_submission(own, a.kernel, a.version, a.message, file_name=a.file)
    ref = str(res.get("ref", ""))
    kc.say(f"submitted {kernel}: ref {ref} {res.get('message', '')}")
    t = dt.datetime.now(dt.timezone.utc)
    for s in _subs():  # prefer Kaggle's own timestamp for the row
        if str(s.get("ref")) == ref:
            t = s["_t"]
            break
    _append_row(t, kernel, a.config, a.local_score, a.message, ref)
    return 0


def status(wait: bool, poll_s: float, timeout_s: float) -> int:
    t0 = time.time()
    while True:
        subs = {str(s.get("ref")): s for s in _subs()}
        if not LOG.exists():
            kc.say("no docs/SUBMISSIONS.md yet")
            return 0
        lines = LOG.read_text().splitlines()
        pending = 0
        for i, ln in enumerate(lines):
            m = re.match(r"^\| (\d+) \|", ln)
            if not m:
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 9:
                continue
            s = subs.get(cells[8])
            if s is None:
                continue
            st = str(s.get("status") or "PENDING")  # PENDING is the omitted enum default
            score = str(s.get("publicScore", "") or "")
            if st == "ERROR" and s.get("errorDescription"):
                st = f"ERROR: {_cell(s['errorDescription'])[:120]}"
            cells[6], cells[7] = st or cells[6], score or cells[7]
            lines[i] = "| " + " | ".join(cells) + " |"
            if not (st.startswith("COMPLETE") or st.startswith("ERROR")):
                pending += 1
            kc.say(f"#{cells[0]} ref {cells[8]}: {st or '?'} public {score or '-'}")
        LOG.write_text("\n".join(lines) + "\n")
        if not wait or pending == 0:
            return 0
        if time.time() - t0 > timeout_s:
            kc.say(f"still {pending} pending after {timeout_s:.0f} s")
            return 1
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="print pacing state and exit")
    p.add_argument("--status", action="store_true", help="fill scores into docs/SUBMISSIONS.md")
    p.add_argument("--wait", action="store_true", help="with --status: poll until scored")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--timeout", type=float, default=6 * 3600)
    p.add_argument("--kernel", help="kernel slug, e.g. casmi-inference")
    p.add_argument("--version", type=int, help="kernel version number (from package_kaggle.py push)")
    p.add_argument("--config", help="config name the kernel version ran")
    p.add_argument("--local-score", default="n/a", help="local validation score to log")
    p.add_argument("--message", help="submission description")
    p.add_argument("--file", default="submission.csv", help="output file of the kernel")
    p.add_argument("--owner", default=None)
    p.add_argument("--dry-run", action="store_true", help="check the rules, do not submit")
    a = p.parse_args(argv)
    if a.check:
        check()
        return 0
    if a.status:
        return status(a.wait, a.poll, a.timeout)
    missing = [n for n in ("kernel", "version", "config", "message") if getattr(a, n) in (None, "")]
    if missing:
        p.error(f"submit mode needs --{', --'.join(missing)}")
    return submit(a)


if __name__ == "__main__":
    sys.exit(main())

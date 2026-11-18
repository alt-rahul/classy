# Rules for implementer agents

Several agents work in this repo at the same time on one machine. Follow these rules exactly.

## Read first
- `docs/PLAN.md`: your work package (WP) section, plus every section it references, plus section 7 (code layout and conventions).
- `docs/COMPETITION.md`: competition background.
- The existing `src/casmi/` modules you depend on (at least `paths.py`, `config.py`, `log.py`).

## Ownership
- Only create or edit the files your WP owns (PLAN.md section 7 lists the owner of every module and script).
- If you need something from a module another WP owns, add a clearly named function at the END of that file without changing existing code, and mention it in your report. Never refactor another WP's code.
- New tunables go into `src/casmi/config.py` as new fields appended to the section that owns them. Never rename or remove existing fields.
- Paths always come from `casmi.paths`. No hard-coded absolute paths in code.

## Shared machine
- 4 CPUs, 15 GB RAM, about 29 GB free disk, no GPU, shared with other agents. Stay within the cores and RAM your WP lists; run heavy jobs under `nice -n 10`.
- A PubChem build (WP3) runs in the background for hours using 2 cores and up to 7 GB RAM. Do not touch `scripts/build_pubchem.py`, `data/external/pubchem/`, `data/external/tmp/` or `data/processed/db/pubchem/`.
- Long jobs: run them in the background with nohup, log to `logs/`, and check on them with the Monitor tool or periodic checks (foreground sleep is blocked). Do not end your task while your own job is still running unless you hit a real blocker.

## Git
- Do not commit, push, or run git commands that change the index or working tree (no add, commit, stash, checkout, reset, rebase). The coordinator commits after reviewing your report. Read-only git (status, diff, log) is fine.

## Code
- Python 3.10 to 3.13 compatible.
- Inference code path (anything `casmi.pipeline` imports) may use only numpy, pandas, pyarrow, scipy, numba, rdkit, lightgbm. Build and training code may also use polars, duckdb, torch, scikit-learn.
- Deterministic: seeds from `Config.seed`, sorted iteration, stable sorts with explicit tie-breakers.
- `pytest -q` must pass (whole suite, not just your tests) before you report done. Tests that need large data use `@pytest.mark.data`.

## Writing
- Never use the em-dash character in any file. Use " - ".
- Record findings that later WPs need in your report; artefact sizes and checksums go in `docs/ARTEFACTS.md` (append your own section).

## Report
End with a report under 350 words: files created or modified, acceptance results with real numbers, deviations from the plan and why, and anything the next WPs must know.

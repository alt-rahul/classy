# Third-party programs

From round 4 on, `store/best.json` holds public programs merged per puzzle (shortest valid program wins; every program is re-validated by `src/common.check` before it enters the store, tag `pub:<source>`). Sources, all public Kaggle datasets for Santa 2026 - The Elf Calculator:

| Tag | Dataset | File | License | Total |
|---|---|---|---|---|
| pub:gp_frontier | goodpjw2008/santa-2026-dag-lns-programs | public_frontier.csv | Apache 2.0 | 133,990 |
| pub:gp_ours | goodpjw2008/santa-2026-dag-lns-programs | our_programs.csv | Apache 2.0 | 133,986 |
| pub:asterios_ds | asteriospanoras/santa-2026-elf-programs-and-experiments | best_programs.csv | Apache 2.0 | 133,985 |
| pub:rudolph | alexchilton/santa-2026-rudolph-programs | rudolph_programs.csv | CC0 | 133,977 |

| pub:asteriospanoras/<file> | asteriospanoras datasets (newer versions, via `src/refresh_public.py`) | various | Apache 2.0 | - |
| pubnb:<owner>/<notebook>/<file> | output files of public competition notebooks (via `src/refresh_public.py`) | various | Apache 2.0 (Kaggle's license for public notebooks) | - |

`src/refresh_public.py` only merges datasets whose Kaggle license is Apache 2.0, CC0, CC BY, CC BY-SA or MIT; datasets licensed "Other" or unlicensed are skipped. Public engine sources used on Kaggle compute are listed in `kaggle/engines/NOTICE_engines.md`.

Re-fetch with `bash src/fetch_public.sh` (needs Kaggle API access), then merge with `python3 src/merge_public.py name=path.csv ...`.
Our own constructor alone scored 162,003 (`submissions/submission_own_162003.csv`).

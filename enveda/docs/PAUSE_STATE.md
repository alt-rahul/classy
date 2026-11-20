# Pause state (2026-11-20 22:06 UTC, paused at the user's request)

All local agents and local processes were stopped. Work in progress is committed (WIP snapshots on the branch). Resume each package from the point below; the binding plan is `docs/ROADMAP.md`.

| Package | State when stopped | How to resume |
|---|---|---|
| R1 validation v2 | Cells, w_obs and fast panel built; b1 fast panel P(A) = 0.204 [0.186, 0.221] (588 s). Was computing probe-submission local MRRs from b1's stored pools. | Re-run the probe MRR step; compare b1's P(A) with its board score once #3 is scored (acceptance: within 0.03). |
| R2 external libraries | Download, conversion and finalize done (peak 1.86 GB). Stopped in the keys stage (mkeys for new structures, NP/PubChem tier membership). | `nice -n 10 python -I scripts/build_extlib.py --stage keys --n-jobs 2`, then remaining stages and `casmi-extlib` publish. |
| R4 encoder | Kaggle GPU kernels `casmi-encoder-train` (since 19:47) and `casmi-encoder-e1` (since 21:40) were still running on Kaggle at pause time. | Fetch their outputs from Kaggle; evaluate against the R4 gates. |
| R6 analog features | Strong signal: same-formula AUC of `an_tan_max` 0.889 (NP-TIMS) / 0.847 (NP-OTHER) / 0.979 (SYN-TIMS), above R7's frag_score. Was running the MRR studies (`scripts/eval_analogs.py --stage analyse ... --variants`). | Re-run that stage. |
| R9 Kaggle kernel | b1 packaged (casmi-library dataset, casmi-inference v3, byte-identical to local) and submitted as #3 at 22:02:22 UTC. Robustness job (extreme-input test, time guards) not started. | Poll #3; then the extreme-input test. |
| R13 virtual-analog tier | Was about to publish the job code dataset and push the 4 build kernels. | Resume from the publish + push step. |
| Other Kaggle CPU kernels | `elf-c01` to `elf-c05` (dataset `elf-compute`) were running on Kaggle at pause time, launched by one of the stopped agents. | Check their purpose and outputs before reuse. |

Submissions at pause time: #1 b0 0.154 (complete); #2 public 0.495 fork (pending); #3 b1 (pending). Next submission allowed 55 minutes after 22:02:22 UTC.

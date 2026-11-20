# Experiments

One row per run, appended by `scripts/evaluate.py` (WP7) or by hand; newest at the bottom.
Rules (PLAN.md sections 5.5, 7 and WP15): one change at a time, one row each; every submission
in `submissions/` gets a row; numbers are local validation (out-of-fold where models are fitted),
plus the public leaderboard score in the notes once the user reports it.

Columns:
- date: YYYY-MM-DD.
- tag: as in the submission name (`b0_`, `b1_`, `b2_`, `r1_`, `final_` ...).
- config: `configs/<name>.yaml` plus any overrides.
- git commit: from `casmi.log.git_commit()` (`-dirty` if uncommitted changes).
- cells: MRR@25 per (set, condition) cell, NP-TIMS / SYN-TIMS / NP-OTHER x C1 / C2 / C3,
  the worst-case blend on NP-TIMS with its 90 percent bootstrap interval, and candidate recall.
- notes: what changed, public LB score, anything surprising.

| date | tag | config | git commit | cells | notes |
|---|---|---|---|---|---|
| 2026-11-18 | b0_npdb_prior | `configs/b0_npdb.yaml` | 42be57fd1474-dirty | none: no local validation exists yet (train not available, WP7 not built) | WP4 Baseline 0, NP tier only, prior-only fusion (PLAN 4.6) + assembly (4.8). `submissions/20261118_b0_npdb_prior.csv`, sha256 `5355da5c...3fcb8a`, 400 rows x 25 guesses. Public LB not meaningful (public test is drawn from train). Pool at 10 ppm: median 44.5, p05 10, p95 138, max 702; empty NP pool 1/400 (0.0025, m_2c2340); 94 molecules padded (89 from the 20 ppm window, 22 from nearest NP rows); 27 mkey duplicates skipped. 46 s cold mkey cache, 4.6 s warm, peak RSS 0.77 GB, 2 mkey workers. Cold and warm runs byte-identical. |
| 2026-11-18 | b0b_pubchem_prior | `configs/b0b_pubchem.yaml` | 42be57fd1474-dirty | none: no local validation exists yet (train not available, WP7 not built) | WP4 Baseline 0b, NP + PubChem tiers, same fusion: every NP-tier candidate ranks above every PubChem-only one; NP rows also use PubChem pmid_count in the literature prior. `submissions/20261118_b0b_pubchem_prior.csv`, sha256 `b29a3a54...ad515f`, 400 x 25. Public LB not meaningful. Pool at 10 ppm: median 10,470, p05 3,289, p95 21,180, max 33,523; no empty pool; no padding; first guess equals b0's on 84.8 percent of molecules. 78 s with mkey cache off (50 s warm), peak RSS 1.66 GB, 8 batches of 50. Cache-off and cached runs byte-identical. |
| 2026-11-20 | b0_npdb_prior | configs/b0_npdb.yaml | 42a4c2b641a5-dirty | NP-TIMS C1/C2/C3 0.822/0.822/0.000 (n 250/250/250, recall 1.00/1.00/0.00), worst 0.493 [0.475, 0.511]; SYN-TIMS C1/C2/C3 0.009/0.000/0.000 (n 44/3000/3000, recall 0.05/0.00/0.00), worst 0.002 [0.000, 0.004]; NP-OTHER C1/C2/C3 0.486/0.414/0.000 (n 1871/3000/3000, recall 0.99/0.99/0.00), worst 0.263 [0.255, 0.270] | (eval 1237 s) |
| 2026-11-20 | b1_library_np | configs/b1_library.yaml + db_tiers [np] | 2e3b74a37eb4-dirty | NP-TIMS C1/C2/C3 0.941/0.679/0.000 (n 250/250/250, recall 1.00/1.00/0.00), worst 0.460 [0.444, 0.476]; SYN-TIMS C1/C2/C3 0.277/0.000/0.000 (n 44/3000/3000, recall 1.00/0.00/0.00), worst 0.055 [0.041, 0.070]; NP-OTHER C1/C2/C3 0.859/0.426/0.000 (n 1871/3000/3000, recall 0.98/0.98/0.00), worst 0.342 [0.336, 0.348] | WP8 Baseline 1, NP tier only (one change vs b0_npdb_prior: the casmi.libmatch lib stage, graded evidence, rule 1 at >= 0.95 / 4 peaks, additive rule 1b, demotion below 0.2; tuning in the WP8 note below). Public test plumbing (NP tier): 400/400 at rank 1 (eval 690 s wall, sets run in parallel) |
| 2026-11-20 | b0b_pubchem_prior | configs/b0b_pubchem.yaml | 2e3b74a37eb4-dirty | NP-TIMS C1/C2/C3 0.827/0.827/0.000 (n 250/250/250, recall 1.00/1.00/0.00), worst 0.496 [0.478, 0.514]; SYN-TIMS C1/C2/C3 0.008/0.001/0.000 (n 44/3000/3000, recall 0.93/0.96/0.00), worst 0.002 [0.000, 0.004]; NP-OTHER C1/C2/C3 0.491/0.418/0.000 (n 1871/3000/3000, recall 0.97/0.97/0.00), worst 0.265 [0.257, 0.272] | (eval 2164 s) |
| 2026-11-20 | b1_library_prior | configs/b1_library.yaml | 41a661bfc91c-dirty | NP-TIMS C1/C2/C3 0.942/0.674/0.000 (n 250/250/250, recall 1.00/1.00/0.00), worst 0.458 [0.442, 0.474]; NP-OTHER C1/C2/C3 0.859/0.428/0.000 (n 1871/3000/3000, recall 0.97/0.97/0.00), worst 0.343 [0.337, 0.349] | WP8 Baseline 1 = the submission config (LIB stage + NP and PubChem tiers; settings and tuning in the WP8 note below). NP-TIMS and NP-OTHER only (--batch 25): the all-sets run was OOM-killed by the shared memory cgroup during SYN-TIMS C3; its log had SYN-TIMS C1/C2 0.239/0.001 (n 44/3000, recall 1.00/0.96), C3 is 0 without a generator. submissions/20261120_b1_library_prior.csv sha256 406b36df...239c760; public test local MRR 1.000 (400/400 at rank 1) against public_test_truth (eval 2424 s wall, sets run in parallel) |

## Notes and spikes

Free-text results that do not fit a row (WP9 tautomer-consistency report, WP13 DreaMS spike,
feature importance, mix-sensitivity tables) go below, each under a dated heading.

### 2026-11-18 WP9 code phase: fingerprint-model throughput (synthetic data, no train yet)

`python scripts/train_fpmodel.py --benchmark --bench-seconds 30 --threads 3 [--bench-peaks 32]`
(results in `logs/fp_benchmark_{100,32}peaks.json`). Full training steps (augmentation,
featurisation, forward, backward, AdamW) on synthetic batches of 512, 3 torch threads, machine
shared with other agents (load average 1 to 3 during the runs). "Overlapped" = data preparation
hidden by the prefetch thread (`fpm_prefetch`, the default in `train`).

| variant | params | peaks / entries per spectrum | spectra/s (sync) | spectra/s (overlapped) | epoch of 2.3M |
|---|---|---|---|---|---|
| plan model (1024 wide, 2 blocks, 0.1 Da), fused AdamW | 27.0M | 100 / 181 | 3,656 | 3,934 | 10.5 min |
| same, torch default (foreach) AdamW | 27.0M | 100 / 181 | 2,388 | 2,505 | 16.1 min |
| plan model + bf16 autocast | 27.0M | 100 / 181 | 3,711 | 4,001 | 10.3 min |
| plan model + bf16 + batch 1024 | 27.0M | 100 / 181 | 5,805 | 6,480 | 6.6 min |
| width 512 (risk 8 fallback) | 13.0M | 100 / 181 | 6,957 | 8,050 | 5.5 min |
| 0.2 Da bins (risk 8 fallback) | 16.8M | 100 / 181 | 4,216 | 4,597 | 9.1 min |
| plan model, fused AdamW | 27.0M | 46 / 86 | 4,200 | 4,411 | 9.1 min |
| plan model + bf16 + batch 1024 | 27.0M | 46 / 86 | 6,453 | 6,947 | 5.9 min |
| width 512 | 13.0M | 46 / 86 | 6,300 | 6,850 | 6.1 min |

Findings: the cost is dominated by the 20,000 x 1024 embedding (dense gradient fill, embedding-bag
backward, AdamW update of 20.5M rows), not by the matmuls, so bf16 alone gains little. torch's
fused AdamW kernel (`fpm_fused_adamw`, default True) makes the optimiser step 6x faster (94 ms to
14 ms per step) and brings the plan model to about 4k spectra/s. Sparse embedding gradients with
SparseAdam were slower (1,300 spectra/s). After cleaning, public test spectra have a mean of 32
peaks (median 24): 27 fragment and 28 loss bins, 65 embedding entries per spectrum, so the
100-peak benchmark is pessimistic. With the plan recipe a 4 h budget gives about 22 to 26 epochs
of 2.3M spectra; no fallback is needed. Numpy inference of the full-size model on all 1,213
public spectra: 1.4 s; `candidate_scores` on 30k packed candidates: 0.24 s.

### 2026-11-18 WP9 tautomer consistency, proxy before train (NP tier vs PubChem tier)

PLAN 4.4 asks for train SMILES vs PubChem SMILES of the same ik14 (needs train; rerun then with
`casmi.fpmodel.fingerprint_consistency`). Proxy measured now: 10,000 random NP-tier structures
whose SMILES come from COCONUT or LOTUS (not PubChemLite), matched by ik14 to the PubChem tier,
Morgan 4096 on `chem.std_mol` of each SMILES (script kept in the WP9 scratchpad, 45 s).

| comparison | mean Tanimoto | identical fingerprints | pairs with Tanimoto < 0.9 |
|---|---|---|---|
| `std_mol` (current) | 0.9925 | 97.96 % | 2.01 % |
| `std_mol` + InChI round trip (`MolFromInchi(MolToInchi(m))`) | 1.0000 | 100.00 % | 0.00 % |

The SMILES strings differ for 98 % of pairs, yet the fingerprints agree for 98 %. The failures are
tautomers: COCONUT writes many amides as imidic acids (`N=C(O)`), plus lactam/lactim and
keto/enol forms; their Tanimoto to the PubChem form is often 0.23 to 0.30, so the true candidate
would get a wrecked fingerprint score in about 2 % of cases. The mean is above the plan's 0.95
trigger, but the round trip removes the problem entirely here and costs 0.98 ms instead of
0.44 ms per molecule (std_mol + Morgan). Recommendation for WP1/WP10: add the InChI round trip to
the shared standardisation for both targets and candidates (then rebuild `np_fp.npy` and
retrain), unless the train-vs-PubChem measurement after WP6 says otherwise.

### 2026-11-18 InChI round trip adopted for fingerprints (tautomer proxy re-measured)

Follow-up to the WP9 entry above. `chem.fp_mol` = `std_mol` + `MolFromInchi(MolToInchi(m))`
(stereo removed; the `std_mol` molecule is kept if the round trip fails or changes the heavy-atom
formula), behind `Config.fp_inchi_roundtrip` (default True). It feeds every fingerprint path:
`chem.morgan_fps` / `maccs_fps` / `featurize` (morgan, maccs), `fpmodel.build_targets` (target
cache now records the flag) and `build_npdb.py --stage fp` (`np_fp.npy` rebuilt, sha256
`a526ef20...20f9a9`). Emitted SMILES, `std_smiles`, `canonical_smiles`, `metric.mkey`, formula,
mass and NP-likeness are unchanged (Baseline 0 still reproduces `20261118_b0_npdb_prior.csv`
byte for byte). Same 10,000 NP-tier (COCONUT/LOTUS SMILES) vs PubChem-tier pairs as WP9:

| fingerprints on | mean Morgan Tanimoto | identical Morgan | Tanimoto < 0.9 | identical MACCS | ms per molecule (parse, standardise, Morgan; 1 process) |
|---|---|---|---|---|---|
| `std_mol` (before) | 0.9925 | 97.96 % | 2.01 % | 98.49 % | 0.41 |
| `fp_mol` (after) | 1.0000 | 100.00 % | 0.00 % | 100.00 % | 1.09 |

Round-trip fallbacks: 0 of 20,000 SMILES. The rebuilt `np_fp.npy` rows of these 10,000 structures
equal `morgan_fps` of the NP SMILES (100 %) and of the PubChem SMILES (100 %), so a target built
from a PubChem-style train SMILES now matches the stored candidate fingerprint exactly. Cost: the
NP-tier fingerprint stage takes 11.3 min instead of 5.5 min on 2 cores; targets for about 275k
train structures add about 3 min on one process. Standard InChI does not cover plain keto/enol
(`CC(=O)C` vs `CC(O)=C` still differ); every pair of this sample became identical, so its
differences were all of the heteroatom mobile-H kind InChI covers. Rerun with
`casmi.fpmodel.fingerprint_consistency` on train SMILES vs PubChem SMILES once train exists.

### 2026-11-18 WP13 DreaMS embeddings on CPU: not adopted (throughput just below 150/s, no gain over plain cosine where mass is controlled)

`DreaMS_embedding_model_torchscript.pt` (HF `roman-bushuiev/DreaMS`, 468,475,544 bytes, sha256
`df69866990b194b27d5497e440f87f6ceca568ee512e7c22cb1f278721376f0c`, in `data/external/dreams/`)
loads with plain `torch.jit.load` on torch 2.15.0+cpu without the `dreams` package (0.4 s, 117.1M
parameters, output 1024-d, finite for all 1,213 public test spectra). Input format (read from
`dreams.utils.data.SpectrumPreprocessor` and the scripted graph): float32 `(batch, 101, 2)`; row 0
is `(precursor_mz, 1.1)`; rows 1-100 are the 100 most intense peaks of the raw spectrum in
ascending m/z with intensity divided by the base peak, raw m/z (the model divides m/z by 1000
itself and builds Fourier features from raw m/z), zero rows (m/z 0) as padding at the end. Always
pad to exactly 100 peaks: the model masks padded query rows but not padded keys, so trimming the
padding changes the embedding (cosine 0.76 to 0.99 to the padded one, measured). As shipped it
runs at 5.8 spectra/s on 1 thread with 2.4 GB peak RSS (it materialises a batch x 101 x 101 x 980
graphormer tensor that is only ever summed over its last axis). A probe wrapper around the scripted
submodules (pre-reduced bias, exact; bf16 weights with AMX; Fourier features kept in fp32; first
Fourier linear split into cos and sin halves) reaches 34 to 39 spectra/s per thread at 0.84 GB
with cosine >= 0.99993 to fp32. Plain bf16 autocast is wrong: the Fourier features (frequencies up
to 5000 per Da) lose precision and embeddings end up at median cosine 0.31 to fp32. Scaling: 2
torch threads in one process give only 43/s; 2 single-thread processes give 72.8/s together while
other agents occupied the other 2 cores. 4-core extrapolation, assuming 4 independent
single-thread processes with the per-process rate measured with all 4 cores busy (about 36/s):
about 145/s (range 135 to 155), so 600k consensus spectra would take about 70 minutes and the
fast path needs AMX-bf16 (fp32 gives about 6/s per core, about 24/s on 4 cores). Sanity
check (fp32 embeddings; bf16 agrees to 3 decimals): embeddings separate molecules, but the gain
over plain cosine comes from the precursor token. Precursor m/z alone finds a same-molecule
nearest neighbour for 0.972 of spectra (same polarity). Inside a precursor window, where library
search operates, plain cosine is as good or better: 0.02 Da window, 491 spectra that have both a
same-molecule and an isobaric other-molecule candidate: plain 1.000, DreaMS 0.994, precursor-only
0.965 (5 Da window, 1,073 spectra: 0.990 vs 0.993). Isobaric other-molecule pairs score a median
0.396 with DreaMS against 0.016 with plain cosine. Feeding `casmi.spectra.clean`ed peaks instead
of raw top 100 changes nothing material. Decision (PLAN WP13 rule, 150/s on 4 cores): drop. Other
costs if revisited: 600k x 1024 fp16 embeddings are a 1.26 GB artefact on top of the 469 MB
model. The open question (does DreaMS retrieve structurally closer analogs for classes 2 and 3,
by hit Tanimoto) needs train and is untested. Probe scripts live only in the WP13 scratchpad.

| measure (1,213 public test spectra, `nice -n 10`, shared machine) | DreaMS | plain cosine 0.01 Da on cleaned spectra |
|---|---|---|
| spectra/s, 1 thread, TorchScript as shipped, fp32 | 5.8 (2.36 GB RSS) | - |
| spectra/s, 1 thread, bf16 probe wrapper, batch 8-16 | 34-39 (0.84 GB RSS) | - |
| spectra/s, 4 cores (extrapolated from 1 process and 2 processes) | about 145 (135-155) | - |
| median similarity, same vs different molecule, all pairs | 0.756 vs 0.195 | 0.695 vs 0.000 |
| same, same-polarity pairs only | 0.820 vs 0.207 | 0.788 vs 0.006 |
| NN is same molecule, any polarity (1,158 spectra with a partner) | 0.919 | 0.863 |
| NN same molecule, same polarity (1,090; pos 916 / neg 174) | 0.975 (0.976 / 0.971) | 0.925 (0.921 / 0.943) |
| control, no peaks used: NN by closest precursor m/z, same polarity (1,090) = 0.972 | - | - |
| NN same molecule within 0.02 Da precursor window (491) | 0.994 | 1.000 |

### 2026-11-20 WP8 tuning: library match rules (Baseline 1)

Method: `casmi.libmatch` "lib" stage run once per cell through the WP7 harness (exclusions,
library guard, checks; NP tier only, so pools are small), every scored pool stored, then an
offline grid re-scores the stored pools with `fusion.fuse` under each setting (rank = first
truth-key row after `fusion.sort_pool` order, dedupe by ik14; the harness reproduces the offline
numbers exactly for the chosen setting). Tuned on NP-TIMS (C1 n 250, C2 n 250) and NP-OTHER (C1
n 1,871, C2 n 3,000); C3 is 0 for every setting (no generator). "w" = worst-case blend with C3 =
0; "p0" = C2 restricted to pubmed_count = 0 (NP-TIMS n 14, NP-OTHER n 1,350), the least
popularity-biased class 2 proxy (PLAN 5.6). Scripts kept in the WP8 scratchpad only.

| setting (lib_best_cos from `lib_sim`) | NP-TIMS C1 | C2 | p0 | w | NP-OTHER C1 | C2 | p0 | w |
|---|---|---|---|---|---|---|---|---|
| no LIB rule (LIB rows ranked by prior only, about Baseline 0) | 0.820 | 0.820 | 0.462 | 0.492 | 0.483 | 0.412 | 0.145 | 0.261 |
| plan rule 1 (0.7, 4 peaks), cos_max | 0.916 | 0.679 | 0.460 | 0.455 | 0.855 | 0.393 | 0.140 | 0.328 |
| plan rule 1 (0.7, 4 peaks), graded | 0.929 | 0.673 | 0.460 | 0.455 | 0.878 | 0.390 | 0.140 | 0.331 |
| + rule 1b band (0.5, 2 peaks: 2.3 to 3.0, above all NP) | 0.938 | 0.582 | 0.388 | 0.420 | 0.891 | 0.367 | 0.137 | 0.325 |
| additive 1b (weight 1.0, from 0.5, 2 peaks) + rule 1 only at >= 0.95 | 0.951 | 0.668 | 0.424 | 0.457 | 0.890 | 0.389 | 0.140 | 0.334 |
| **same + demotion below 0.2 (chosen, `configs/b1_library.yaml`)** | **0.941** | **0.679** | **0.464** | **0.460** | **0.858** | **0.424** | **0.156** | **0.341** |
| additive (0.5, from 0.7, 4 peaks) + rule 1 >= 0.95 + demotion below 0.4 | 0.926 | 0.765 | 0.505 | 0.491 | 0.836 | 0.448 | 0.164 | 0.347 |

Findings:
- The search's single "best pair" can be a 2-peak spectrum at cosine 1.0 that hides a 20-peak
  match (the only public-test miss of the first version, m_e80818). `lib_sim: graded` (best
  pair over ALL spectrum and consensus pairs by (passes rule 1, passes 1b, cosine)) fixes it and
  is the best similarity source in every setting (+0.013 C1 NP-TIMS, +0.022 NP-OTHER at the plan
  thresholds over cos_max; consensus cosine alone is worse on C1).
- On NP-TIMS the NP-tier prior alone already ranks the truth first for 72 percent of molecules
  (common natural products), so any rule that hard-overrides the prior lifts C1 and costs C2.
  The C2 loss sits on popular truths: on the pubmed_count = 0 subset rule 1 costs nothing
  (0.462 vs 0.460). This is the popularity bias of simulated class 2 (PLAN 5.6): a real class 2
  molecule has no public spectra and is less popular than its library isomers, which the prior
  already ranks above it.
- Plan rule 1 misses on NP-TIMS C1 (0.7, 4 peaks): 20 of 29 are an isomer that also passes
  rule 1 with a higher cosine, 8 a truth with cosine below 0.7 (cross-instrument references), 1
  without a same-polarity reference. A prior tie-break inside rule 1 adds only +0.004; adding
  the similarity to the prior score instead (additive 1b) gains +0.022 C1 and loses no C2.
- Demotion (measured best cosine below a threshold: LIB candidate below every NP-tier one) is
  real evidence, not a popularity effect: it raises C2 on the pubmed_count = 0 subset of
  NP-OTHER (n 1,350) from 0.140 to 0.156 (0.2) and 0.164 (0.4), at a C1 cost (true references
  across instruments can score below 0.2: NP-OTHER C1 -0.032 at 0.2). 0.2 was chosen: on the
  proxy blend 0.2 x NP-TIMS C1 + 0.4 x NP-OTHER p0 every demotion setting ties (0.2506 to
  0.2508), and 0.2 keeps the most C1.
- Under the plan's primary number (NP-TIMS worst-case blend) no LIB setting beats "prior only"
  (0.492) by more than noise; the strongest-demotion row ties it (0.491) and wins clearly on
  NP-OTHER (0.347 vs 0.261). The ranker (WP11) should learn this trade-off from the LIB
  features rather than from fixed rules.

### 2026-11-20 R7 MetFrag-lite fragment features and fragment mass calibration

Code: `src/casmi/fragments.py` (stage `fragments`, plugin `casmi.fragments`),
`scripts/eval_fragments.py`, `scripts/fragment_calibration.py`; settings are the public ones
(1-2 bond cuts, two acyclic cuts credited 0.6, linear intensity, 0.005 Da, H shifts -2..+3),
ring cleavage (two bonds of one ring system) credited 0.6 as well. Commands (1 core, `nice`):
`python scripts/eval_fragments.py --sets NP-TIMS NP-OTHER --tag v1` (864 s, 1.37 GB) and
`--sets SYN-TIMS` (900 s, 1.6 GB); `python scripts/fragment_calibration.py --per-lib 40000`
(405 s). Tables and summaries in `data/processed/fragments/` (ARTEFACTS.md, R7).

Protocol: v1 holdout, condition C2 (all library spectra of the molecule hidden, database
complete), pools from the NP + PubChem tiers at 10 ppm through `validation.ExcludingDB`, current
ranking = prior-only fusion (`pipeline` stages mass, pool, features, score; b0 rules). The top
100 by that score plus every truth row get fragment features. AUC: per molecule, truth versus
same-formula decoys among those rows (ties one half), mean over molecules with at least one
such decoy. MRR@25 with ik14 de-duplication (the WP8 offline convention); "+ w x f": w x f added
to the score of the top-100 rows, w from a 15-value grid tuned on one hash half of the
molecules and tested on the other, both ways (cross-fit). Re-run on v2 cells with `--val-dir` /
`--condition` once R1 lands; `--rescore-from` re-scores stored rows under other settings.

AUC, truth versus same-formula decoys (higher value = truth; below 0.5 means lower is better):

| feature | NP-TIMS (n 249) | NP-OTHER (n 2,788) | NP-OTHER pubmed 0 (n 1,281) | SYN-TIMS (n 2,697) |
|---|---|---|---|---|
| `frag_score` (credited explained intensity) | **0.759** | **0.794** | 0.783 | **0.958** |
| `frag_expl_lin` | 0.748 | 0.776 | 0.767 | 0.944 |
| `frag_expl_sqrt` | 0.736 | 0.773 | | 0.946 |
| `frag_n_expl` / `frag_frac_peaks` | 0.698 | 0.753 | | 0.931 |
| `frag_base_w` (credit of the base peaks) | 0.713 | 0.736 | | 0.910 |
| `frag_base_expl` | 0.676 | 0.692 | | 0.867 |
| `frag_base_best` | 0.628 | 0.670 | | 0.815 |
| `frag_mean_cuts` (lower = truth) | 0.440 | 0.346 | | 0.221 |
| `frag_err_mda` (lower = truth) | 0.467 | 0.410 | | 0.441 |
| `frag_log_n` (structure only, no spectrum) | 0.437 | 0.498 | | 0.476 |
| formula-level `formula.explain` | 0.500 by construction | 0.500 | 0.500 | 0.500 |
| `frag_score`, truth versus all decoys in the top 100 | 0.803 | 0.834 | | |

The within-molecule `_z`, `_rank`, `_fz`, `_gap` columns are monotone in the raw value inside a
molecule, so their per-molecule AUC equals the raw one. Decoy source: NP-OTHER 0.778 against
NP-tier isomers, 0.827 against PubChem-only isomers (NP-TIMS 0.750 / 0.801). Explained
intensity: the true formula's sub-formulas explain 0.95 (NP-TIMS) / 0.83 (NP-OTHER) / 0.99
(SYN-TIMS) of the intensity, the true structure's fragments 0.59 / 0.56 / 0.74, same-formula
decoys 0.37 / 0.31 / 0.27. SYN-TIMS caveat: the truth is in the prior top 100 for only 3.4
percent of molecules (synthetic, pubmed 0), so its decoys are the most popular PubChem isomers;
on the 101 molecules with the truth in the top 100 the AUC is 0.927.

MRR@25 of the prior-only ranking and of prior + w x `frag_score_z` (primary, fixed in advance):

| tuning | set | base | + fragments | rank-1 wins / losses | w (halves) |
|---|---|---|---|---|---|
| one weight, both sets pooled | NP-TIMS + NP-OTHER (3,250) | 0.449 | **0.578** | 708 / 218 | 0.3, 0.3 |
| same weight | NP-OTHER (3,000) | 0.417 | **0.570** | 683 / 155 (+17.6 %) | |
| same weight | NP-OTHER pubmed 0 (1,350) | 0.188 | **0.404** | | |
| same weight | NP-TIMS (250) | 0.827 | 0.684 | 25 / 63 (-15.2 %) | |
| tuned within NP-TIMS | NP-TIMS | 0.827 | 0.837 | 18 / 12 (+2.4 %) | 0.01, 0.02 |
| tuned within SYN-TIMS | SYN-TIMS (3,000) | 0.001 | 0.020 | 45 / 0 | 1.5, 1.5 |

Other bonus columns (pooled weight): `frag_score_fz` 0.519, `frag_expl_lin_z` 0.553,
`frag_expl_sqrt_z` 0.556 (all below `frag_score_z`). NP-TIMS grid (whole set, not cross-fit):
w 0.02 0.839, 0.05 0.838, 0.1 0.802, 0.3 0.684.

Variants (re-scored rows, same pools):

| variant | NP-TIMS AUC | NP-OTHER AUC | NP-OTHER MRR (cross-fit within NP-OTHER) |
|---|---|---|---|
| default (above) | 0.759 | 0.794 | 0.570 |
| ring cleavage credited 1.0 (MetFrag "one step") | 0.761 | 0.789 | 0.549 |
| no down-weighting (two cuts and rings 1.0) | 0.748 | | |
| tolerance 0.01 Da | 0.753 | | |
| positive-mode offset 0.51 mDa removed (NP-TIMS queries) | 0.759 | | |

Findings:
- Structure-level fragment evidence separates isomers where formula evidence cannot (public
  0.757 reproduced: 0.759 on NP-TIMS, 0.794 on NP-OTHER); the R7 AUC acceptance (>= 0.72 on
  the C2 cells) holds on all three sets. The structure-only confounder `frag_log_n` has no
  signal (0.44 to 0.50), so the separation comes from the spectra, not from candidate size.
- As one fusion term it is worth +0.15 MRR on NP-OTHER (+0.22 on its pubmed-0 half), but it
  costs NP-TIMS 0.14 at the same weight: NP-TIMS truths are famous (prior top-1 73 percent) and
  their NP-tier isomers often explain the timsTOF spectra better (56 of the 63 NP-TIMS losses
  are same-formula NP-tier isomers, median frag_score margin 0.20). Tuned on NP-TIMS alone the
  best weight is 0.01 to 0.02 (+0.010). The right weight depends on how strong the prior is;
  this is for the ranker (R8) with `w_obs` group weights and prior-gap context, not for a
  fixed fusion weight. Since the board's class 2 is obscure (ROADMAP 2.1), the pubmed-0 numbers
  are the closer proxy.
- C2-PC (SYN-TIMS): fragment features are strongest there (0.958), but a prior-only top 100
  holds the truth for 3.4 percent of molecules; the stage must rank by a score that includes
  the encoder (R4/R5) for the PubChem route, else most of this signal is unreachable.
- Ring cleavages as one step (credit 1.0) and no down-weighting both lose: keep 0.6.

Fragment mass calibration (`scripts/fragment_calibration.py`): residual = observed m/z minus
theoretical ion m/z of the true structure's fragments, peaks whose explanations within 10 mDa
all share one theoretical mass; offset = median over |residual| <= 3 mDa and intensity >= 5
percent (the window tails are chance assignments and not symmetric).

| library, polarity | spectra | core peaks | fragment offset (mDa, se) | precursor offset (mDa) | spectra with median > +0.25 mDa |
|---|---|---|---|---|---|
| enveda-180 + | 29,650 | 154,756 | **+0.548** (0.002) | +0.716 | 83 % |
| enveda-180 - | 10,251 | 37,597 | +0.035 (0.005) | +0.244 | 20 % |
| enveda-np-examples + | 776 | 4,103 | **+0.508** (0.018) | +0.686 | 71 % |
| enveda-np-examples - | 386 | 2,075 | +0.068 (0.028) | +0.155 | 32 % |
| gnps + / - | 31,211 / 6,799 | | -0.009 / -0.239 | +0.037 / -0.018 | |
| massbank + / - | 21,979 / 9,840 | | -0.006 / -0.041 | +0.005 / -0.008 | |
| mona + / - | 24,229 / 7,902 | | +0.044 / **-0.527** | -0.003 / -0.012 | |
| riken + / - | 21,810 / 3,463 | | -0.003 / **-0.621** | -0.005 / -0.013 | |
| pluskal_ms2 + / - | 23,257 / 14,645 | | -0.094 / **-0.545** | +0.003 / +0.005 | |

- The host's report is confirmed: positive-mode fragments of both Enveda libraries read one
  electron mass high (+0.548 mDa against 0.549), flat over m/z in enveda-180 (+0.43 to +0.75 from
  m/z < 100 to 600), on most spectra (per-spectrum median p25/p50/p75 +0.32/+0.51/+0.70) and for
  every positive adduct; negative mode is clean. The precursor m/z carries the same +0.47 mDa
  difference between polarities on top of a common +0.2 mDa. The visible test is a copy of these
  train spectra (identical peaks), the rerun test was corrected, so test and train differ by
  0.55 mDa in positive mode. Several external-style libraries (MoNA, RIKEN, MSnLib/pluskal, part
  of GNPS) have the mirror error in negative mode (-0.5 to -0.6 mDa).
- Correction: `Config.frag_mz_offset_pos_da` / `frag_mz_offset_neg_da` (subtracted before
  fragment matching), default 0.0, which is right for the corrected hidden test. Removing the
  offset from NP-TIMS queries changes nothing at the 5 mDa fragment tolerance (AUC 0.759 ->
  0.759), so validation features need no correction either.
- Elsewhere: library cosine between two enveda-180 [M+H]+ spectra of the same structure (3,007
  pairs) with one side shifted by 0.55 mDa: tolerance 0.01 Da mean delta -0.00001 (0.1 percent of
  pairs move by more than 0.01), 0.005 Da +0.00007 (1.3 percent), 0.002 Da -0.0019 with 10.5
  percent of pairs moving by more than 0.01 and 0.18 matched peaks lost per pair. So the 0.01 and
  0.02 Da library tolerances are unaffected, while 2 mDa features (`formula_strict_tol_da`,
  `expl_frac_strict`) computed on train-derived validation queries are biased against the
  corrected test: subtract 0.55 mDa from positive-mode Enveda spectra before any feature with a
  tolerance at or below 2 mDa, or do not use such features in the ranker. The encoder's 3 ppm
  m/z jitter (about 1 mDa at 350) covers the shift, so no retraining (ROADMAP risk 14).

Speed (one core, cold cache, RDKit parse + enumeration + matching): 400 random NP-OTHER
molecules x 100 candidates in 59.6 s (median 1.23 ms, mean 1.49 ms, p95 3.5 ms per candidate;
per molecule median 0.12 s, p95 0.33 s, max 0.82 s); molecules above 1,000 Da: median 4.97 ms per
candidate, worst 9.7 ms (0.91 s per molecule). Single structures from 3,000 random NP-tier
SMILES: median 0.82 ms, p99 6.0 ms; 200 NP-tier structures above 1,000 Da: median 4.8 ms, p99
22.7 ms, worst 30 ms (a C80 polyprenol with about 90 acyclic bonds). From the disk cache a
candidate costs about 0.07 ms (300k rows re-scored in 20 s).

### 2026-11-20 R3 PubChem documentation priors: prior-only MRR on the class-2 proxies

Method: `scripts/eval_priors.py` (spectrum-free; logs/r3_eval_priors.json and .parquet). One query per holdout molecule of `splits/holdout_v1.csv` (6,250 mkeys); pool = all-tier lookup (NP + PubChem, NP rows win) at 10 ppm around the primary ik14's mass (median 5,267 candidates for C2-NP, 10,346 for C2-PC); correct = any ik14 of the mkey; MRR@25 with ties scored by their expectation under a random order; priors declared before the run, nothing tuned: `pmid` = log1p(pmid_count); `b0` = Baseline 0b order (NP-tier rows first, then `fusion.prior_score`); `sid` = log1p(sid_count); `sid+pmid`; `doc` = log1p(sid) + log1p(pmid) + 1.0 in_np + 0.5 log1p(n_citations) + 0.5 log1p(synonym_count) (the public documentation prior without its depositor-source term). holdout_v2 (R1) is not built yet, so C2-PC here is SYN-TIMS whose truth is not in the NP tier and has no spectra in another train library, and w_obs = (1 + pmid_count)^-alpha is shown for two alphas instead of the calibrated one. Brackets: paired bootstrap 90 percent interval of `doc` minus `pmid` (2,000 resamples of molecules).

| cell | n | truth in pool | pmid | b0 | sid | sid+pmid | doc | doc - pmid |
|---|---|---|---|---|---|---|---|---|
| C2-NP (NP-TIMS + NP-OTHER), unweighted | 3,250 | 0.999 | 0.369 | 0.440 | 0.428 | 0.435 | 0.472 | +0.103 [+0.097, +0.110] |
| C2-NP, w_obs alpha 0.5 | 3,250 | 0.999 | 0.121 | 0.250 | 0.237 | 0.205 | 0.257 | +0.136 [+0.127, +0.145] |
| C2-NP, w_obs alpha 1.0 | 3,250 | 0.999 | 0.049 | 0.198 | 0.182 | 0.131 | 0.186 | +0.137 [+0.127, +0.147] |
| C2-NP, pmid_count = 0 | 760 | 0.996 | 0.001 | 0.156 | 0.136 | 0.070 | 0.124 | +0.122 [+0.108, +0.136] |
| NP-TIMS only, unweighted | 250 | 1.000 | 0.756 | 0.832 | 0.776 | 0.784 | 0.822 | +0.066 [+0.048, +0.086] |
| C2-PC (SYN-TIMS, not NP tier, no other-library spectra) | 2,948 | 0.964 | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 | +0.000 [-0.000, +0.000] |
| C2-PC, pmid_count = 0 | 2,886 | 0.963 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |

Same per-molecule results rescored on `splits/holdout_v2.csv` (R1: `c2_cell`, calibrated `w_obs`, pmid = max(pmid_count, pubmed_count); `python scripts/eval_priors.py --reuse logs/r3_eval_priors_v1_results.parquet --out logs/r3_eval_priors_v2.json`); C2-PC shown to 4 decimals:

| cell | n | pmid | b0 | sid | sid+pmid | doc | doc - pmid |
|---|---|---|---|---|---|---|---|
| v2 C2-NP, w_obs (alpha A 0.796) | 3,250 | 0.069 | 0.210 | 0.194 | 0.151 | 0.205 | +0.136 [+0.126, +0.146] |
| v2 C2-NP, unweighted | 3,250 | 0.369 | 0.440 | 0.428 | 0.435 | 0.472 | +0.103 [+0.096, +0.110] |
| v2 C2-NP, pmid_count = 0 | 746 | 0.001 | 0.153 | 0.132 | 0.068 | 0.121 | +0.120 [+0.107, +0.133] |
| v2 C2-PC | 2,850 | 0.0008 | 0.0006 | 0.0005 | 0.0010 | 0.0009 | +0.0001 [-0.0002, +0.0005] |
| v2 C2-PC, pmid_count = 0 | 2,784 | 0.0000 | 0.0003 | 0.0003 | 0.0002 | 0.0002 | +0.0002 [+0.0001, +0.0004] |

Findings:
- C2-NP: every new prior beats the pmid-only prior, with intervals well above 0: `doc` +0.103 unweighted, +0.136 with the calibrated v2 `w_obs` (0.069 -> 0.205). On the pmid_count = 0 subset (760 molecules, where the pmid-only prior is blind) `sid` alone reaches 0.136 and `doc` 0.124. The R3 acceptance on C2-NP holds.
- `b0` (NP tier first + the WP4 NP prior) is as strong as `doc` once fame is down-weighted (0.198 vs 0.186 at alpha 1.0) and stronger on pmid_count = 0 (0.156 vs 0.124): its NP-tier features (annotation level, organisms, citations, source count) carry information the PubChem counts do not. The two should be combined by the R8 ranker on `w_obs`-weighted groups, not hand-weighted from this table.
- C2-PC: every prior-only ranking is at the floor (MRR about 0.001; the truth, an obscure enveda-180 synthetic, is 1 of about 10,000 PubChem decoys with similar or larger counts). The new priors are not above the pmid-only prior there except negligibly on pmid_count = 0 (+0.0002 [+0.0001, +0.0004]). That part of the R3 acceptance does not hold for a prior alone; on this route the priors can only help as ranker features next to spectral evidence (R4, R5, R7).
- Data checks: `sid_count` equals PubChem PUG REST (`compound/cid/<cid>/sids?sids_type=all`, union over the group's CIDs) on 50 of 50 random tier rows (40 single-CID, 10 multi-CID groups); Baseline 0 and 0b reproduce `submissions/20261118_b0_npdb_prior.csv` and `..._b0b_pubchem_prior.csv` byte for byte on the rewritten tiers.

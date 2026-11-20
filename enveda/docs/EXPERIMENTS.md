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

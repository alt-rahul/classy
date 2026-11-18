"""WP9: fingerprint model - featurisation, numpy inference, scoring, training (PLAN.md 4.4)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi import fpmodel as fm
from casmi.config import Config
from casmi.spectra import SpectrumStore

ROOT = Path(__file__).resolve().parents[1]

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "OC(=O)c1ccccc1O",
    "OC1C(O)C(Oc2ccc(C=CC(=O)O)cc2)OC(CO)C1O", "COc1cc(C=CC(=O)O)ccc1O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "O=C(O)CC(O)(CC(=O)O)C(=O)O", "c1ccc2[nH]ccc2c1",
]


def tiny_cfg(**kw) -> Config:
    base = dict(fpm_hidden=64, fpm_meta_hidden=16, fpm_n_blocks=1, fpm_batch_size=64,
                fpm_threads=2, fpm_warmup_steps=20, fpm_log_every_steps=1000,
                fpm_prefetch=False)
    base.update(kw)
    return Config().replace(**base)


@pytest.fixture(scope="module")
def synth() -> fm.TrainingData:
    return fm.synthetic_training_data(64, 4, seed=1)


# --------------------------------------------------------------------------------------
# Featurisation and metadata
# --------------------------------------------------------------------------------------


def test_instrument_families_match_library():
    from casmi.library import INSTRUMENT_FAMILIES

    assert fm.INSTRUMENT_FAMILIES == INSTRUMENT_FAMILIES


def test_spec_dimensions():
    spec = fm.ModelSpec.from_config(Config())
    assert (spec.n_frag, spec.n_nl, spec.n_in, spec.n_out) == (13000, 7000, 20000, 4263)
    assert spec.n_meta == len(fm.META_FIELDS) == 28
    s2 = fm.ModelSpec.from_config(Config().replace(fpm_mz_bin_da=0.2))
    assert (s2.n_frag, s2.n_nl) == (6500, 3500)
    back, extra = fm.ModelSpec.from_json(spec.to_json({"a": 1}))
    assert back == spec and extra == {"a": 1}


def _naive_bins(mz, inten, prec, spec):
    out = []
    for m, x in zip(mz, inten):
        c = math.floor(m / spec.bin_da)
        if 0 <= m and c < spec.n_frag:
            out.append((c, math.sqrt(x)))
    for m, x in zip(mz, inten):
        loss = prec - m
        if loss >= 0 and math.floor(loss / spec.bin_da) < spec.n_nl:
            out.append((spec.n_frag + math.floor(loss / spec.bin_da), math.sqrt(x)))
    return out


def test_featurize_sparse_matches_naive_loop():
    rng = np.random.default_rng(0)
    spec = fm.ModelSpec.from_config(Config())
    spectra, precs = [], []
    for i in range(40):
        k = int(rng.integers(0, 30))
        prec = float(rng.uniform(100, 1250))
        mz = np.sort(rng.uniform(1, 1400, k)).astype(np.float32)
        spectra.append((mz, rng.uniform(0.01, 1, k).astype(np.float32)))
        precs.append(prec if i % 7 else np.nan)
    store = SpectrumStore.from_spectra(spectra)
    sp = fm.featurize_sparse(store, np.array(precs), spec)
    assert len(sp) == 40 and sp.offsets[0] == 0 and np.all(np.diff(sp.offsets) >= 0)
    for i, (mz, it) in enumerate(spectra):
        a, b = sp.offsets[i], sp.offsets[i + 1]
        got = list(zip(sp.indices[a:b].tolist(), sp.weights[a:b].tolist()))
        exp = _naive_bins(mz.astype(np.float64), it, precs[i], spec)
        assert [g[0] for g in got] == [e[0] for e in exp], i
        np.testing.assert_allclose([g[1] for g in got], [e[1] for e in exp], rtol=1e-6)
        if np.isnan(precs[i]):
            assert (sp.indices[a:b] < spec.n_frag).all()


def test_featurize_public_test(public_test_parquet):
    df = pd.read_parquet(public_test_parquet)
    cfg = Config()
    spec = fm.ModelSpec.from_config(cfg)
    store, meta = fm.spectra_from_frame(df, cfg)
    assert len(store) == len(df) == 1213
    sp = fm.featurize_sparse(store, meta.precursor_mz, spec)
    frag, nl = fm.active_bins(sp, spec)
    assert (frag > 0).all() and (nl > 0).all()
    assert sp.indices.min() >= 0 and sp.indices.max() < spec.n_in
    assert np.isfinite(sp.weights).all() and (sp.weights > 0).all()
    assert np.median(frag) <= cfg.top_n_peaks
    print(f"\npublic test active bins per spectrum: fragment median {np.median(frag):.0f} "
          f"mean {frag.mean():.1f}, loss median {np.median(nl):.0f} mean {nl.mean():.1f}, "
          f"entries mean {np.diff(sp.offsets).mean():.1f}")
    x = fm.meta_matrix(meta, store.n_peaks, spec)
    assert x.shape == (1213, 28) and np.isfinite(x).all()
    assert (x[:, :11].sum(1) == 1).all() and x[:, 10].sum() == 0      # all test adducts
    assert (x[:, fm.META_FIELDS.index("inst=timstof")] == 1).all()
    pos = (df["ionization_mode"] == "positive").to_numpy()
    assert (x[:, fm.META_FIELDS.index("positive")] == pos).all()
    merged = df["collision_energy_ev"].map(len).to_numpy() > 1
    assert (x[:, fm.META_FIELDS.index("ce_merged")] == merged).all()
    assert (meta.n_peaks_raw == df["ms2_mzs"].map(len).to_numpy()).all()
    assert np.isfinite(meta.mass).all()


def test_meta_from_frame_aliases_and_missing():
    df = pd.DataFrame({
        "precursor_mz": [181.07, 179.05, 400.0, 300.0],
        "adduct": ["[M+H]+", "[M-H]-", "[2M+H]+", "nonsense"],
        "ionization_mode": ["positive", "negative", "positive", None],
        "collision_energy_ev": [[20.0, 40.0, 60.0], [30.0], None, []],
        "instrument_type": ["Orbitrap Exploris", "maXis QTOF", None, "Q Exactive"],
        "ingest_lib": ["gnps", "gnps", "enveda-180", "mona"],
        "base_peak_intensity": [1e5, np.nan, 0.0, 1e3],
    })
    m = fm.SpectrumMeta.from_frame(df)
    assert m.adduct_code.tolist() == [0, 6, 10, 10]
    assert m.adduct_known.tolist() == [True, True, True, False]
    assert m.polarity.tolist() == [1, -1, 1, 0]
    assert m.n_energies.tolist() == [3, 1, 0, 0]
    assert m.collision_energy[0] == 40.0 and np.isnan(m.collision_energy[2])
    assert [fm.INSTRUMENT_FAMILIES[c] for c in m.instrument_code] == [
        "orbitrap", "qtof", "timstof", "orbitrap"]
    assert abs(m.mass[0] - (181.07 - 1.007276)) < 1e-4
    assert abs(m.mass[2] - (400.0 - 1.007276) / 2) < 1e-4
    x = fm.meta_matrix(m, np.array([10, 5, 0, 3]))
    assert x[0, fm.META_FIELDS.index("ce_merged")] == 1
    assert x[1, fm.META_FIELDS.index("bpi_missing")] == 1
    assert x[2, fm.META_FIELDS.index("bpi_missing")] == 1        # bpi 0 counts as missing
    assert x[2, fm.META_FIELDS.index("ce_missing")] == 1


# --------------------------------------------------------------------------------------
# Training data, filters, sampling, augmentation
# --------------------------------------------------------------------------------------


def test_eligibility_filter(synth):
    d = synth.subset(np.arange(len(synth)))
    d.label_mass[0] = 90.0
    d.precursor_error_ppm[1] = 25.0
    d.meta.adduct_known[2] = False
    d.target_row[3] = -1
    mask, drops = d.eligible(Config())
    assert not mask[:4].any() and mask[4:].sum() == len(d) - 4
    assert drops == {"no_target": 1, "mass_range": 1, "min_peaks": 0, "adduct_unparsed": 1,
                     "precursor_error": 1}


def test_sampling_weights_and_epochs(synth):
    cfg = Config()
    d = synth.subset(np.arange(len(synth)))
    d.target_row[:] = np.repeat(np.arange(len(d) // 4), 4)
    d.target_row[:9] = 0                                  # structure 0: 9 spectra
    w = fm.sampling_weights(d, cfg)
    tl = (d.meta.adduct_code < 10) & (d.meta.instrument_code == 0)
    exp = 1 / np.sqrt(np.bincount(d.target_row)[d.target_row]) * np.where(tl, 2.0, 1.0)
    np.testing.assert_allclose(w, exp)
    a = fm.epoch_indices(w, 200000, 5, 0)
    assert (a == fm.epoch_indices(w, 200000, 5, 0)).all()
    assert not (a == fm.epoch_indices(w, 200000, 5, 1)).all()
    freq = np.bincount(a, minlength=len(d)) / len(a)
    np.testing.assert_allclose(freq, w / w.sum(), atol=0.002)


def test_pos_weights_clipped(synth):
    cfg = Config()
    w = np.ones(len(synth))
    pw = fm.pos_weights(synth, w, cfg)
    y = synth.targets.dense(synth.target_row)
    f = y.mean(0)
    exp = np.clip(np.where(f > 0, (1 - f) / np.maximum(f, 1e-12), np.inf), 1, 10)
    np.testing.assert_allclose(pw, exp, rtol=1e-5)
    assert pw.min() >= 1 and pw.max() <= 10


def _store_ok(st: SpectrumStore, n: int):
    assert len(st) == n and st.offsets[0] == 0 and st.offsets[-1] == len(st.mz)
    assert st.mz.dtype == np.float32 and st.intensity.dtype == np.float32
    for i in range(n):
        m, x = st[i]
        assert len(m) > 0 and abs(float(x.max()) - 1.0) < 1e-6


def test_augment_identity_when_disabled(synth):
    cfg = Config().replace(fpm_aug_consensus_prob=0.0, fpm_aug_peak_drop=0.0,
                           fpm_aug_intensity_sigma=0.0, fpm_aug_noise_peaks_max=0)
    idx = np.arange(0, 200, 3)
    st, meta = fm.augment_batch(synth, idx, np.random.default_rng(0), cfg)
    ref = synth.store.take(idx)
    assert (st.offsets == ref.offsets).all() and (st.mz == ref.mz).all()
    np.testing.assert_allclose(st.intensity, ref.intensity, rtol=1e-6)


def test_augment_dropout_jitter_noise(synth):
    cfg = Config().replace(fpm_aug_consensus_prob=0.0, fpm_aug_peak_drop=0.5,
                           fpm_aug_noise_peaks_max=10)
    idx = np.arange(len(synth))
    st, meta = fm.augment_batch(synth, idx, np.random.default_rng(1), cfg)
    _store_ok(st, len(idx))
    tims = synth.meta.instrument_code[idx] == 0
    n0, n1 = synth.store.n_peaks[idx], st.n_peaks
    assert (n1[tims] <= n0[tims]).all()           # timsTOF: dropout only, no noise
    assert n1.sum() < n0.sum()
    for i in range(len(idx)):                     # base peak always survives
        m0, x0 = synth.store[int(idx[i])]
        assert float(m0[np.argmax(x0)]) in set(st[i][0].tolist())
    same = fm.augment_batch(synth, idx, np.random.default_rng(1), cfg)[0]
    assert (same.mz == st.mz).all()               # deterministic in the RNG


def test_augment_consensus_partners():
    d = fm.synthetic_training_data(16, 6, seed=2)
    d.meta.adduct_code[:] = 0
    d.meta.instrument_code[:] = 0
    d.meta.collision_energy[:] = np.tile([10.0, 20.0, 40.0, 60.0, 60.0, np.nan], 16)
    d._groups = None
    cfg = Config().replace(fpm_aug_consensus_prob=1.0, fpm_aug_peak_drop=0.0,
                           fpm_aug_intensity_sigma=0.0, fpm_aug_noise_peaks_max=0)
    idx = np.arange(len(d))
    st, meta = fm.augment_batch(d, idx, np.random.default_rng(0), cfg)
    _store_ok(st, len(idx))
    no_ce = np.isnan(d.meta.collision_energy)
    assert (meta.n_energies[no_ce] == 1).all()            # no partner: unchanged
    assert (meta.n_energies[~no_ce] >= 2).all() and (meta.n_energies <= 3).all()
    assert (st.n_peaks[~no_ce] >= d.store.n_peaks[~no_ce] * 0.9).all()
    # partners have other energies, so the mean CE of a 60 eV spectrum drops
    sixty = d.meta.collision_energy == 60.0
    assert (meta.collision_energy[sixty] < 60.0).all()


def test_make_batch_shapes(synth):
    cfg = Config()
    spec = fm.ModelSpec.from_config(cfg)
    sp, x, y = fm.make_batch(synth, np.arange(32), spec, cfg, np.random.default_rng(0))
    assert len(sp) == 32 and x.shape == (32, 28) and y.shape == (32, 4263)
    assert x.dtype == np.float32 and y.dtype == np.float32
    np.testing.assert_array_equal(y, synth.targets.dense(synth.target_row[:32]))


def test_holdout_exclusion_by_ik14_and_mkey():
    train_ik = np.array(["AAA", "BBB", "CCC", "DDD", "AAA"], dtype=object)
    ho = pd.DataFrame({"ik14": ["AAA", "ZZZ"], "mkey": ["MK1", "MK2"]})
    st = pd.DataFrame({"ik14": ["AAA", "BBB", "CCC", "DDD"], "mkey": ["MK1", "MK9", "MK2", "MK3"]})
    mask, info = fm.holdout_exclusion(train_ik, ho, st)
    assert mask.tolist() == [True, False, True, False, True]
    assert info["by_ik14"] == 2 and info["by_mkey"] == 1 and info["mkey_checked"]
    mask2, info2 = fm.holdout_exclusion(train_ik, ho, None)
    assert mask2.tolist() == [True, False, False, False, True] and not info2["mkey_checked"]


def test_split_training_data_synthetic():
    cfg = Config()
    d = fm.synthetic_training_data(200, 4, seed=3)
    tr, val, cal, info = fm.split_training_data(d, cfg, exclude_holdout=True, holdout=None,
                                                max_spectra=500)
    assert len(tr) == 500 and val is not None and cal is not None
    assert not set(tr.ik14.tolist()) & set(val.ik14.tolist())
    assert (cal.meta.adduct_code < 10).all()
    tr2, val2, cal2, _ = fm.split_training_data(d, cfg, exclude_holdout=False)
    assert len(tr2) == len(d) and val2 is None and cal2 is None


def test_build_targets_and_cache(tmp_path):
    from casmi import chem

    cfg = Config()
    ik = [f"K{i:013d}" for i in range(len(SMILES))]
    cache = tmp_path / "t.npz"
    t1 = fm.build_targets(ik[:5], SMILES[:5], cfg, cache=cache)
    t2 = fm.build_targets(ik[::-1], SMILES[::-1] + [], cfg, cache=cache)  # 5 cached + 3 new
    res = chem.featurize(SMILES, fields=("morgan", "maccs", "mono_mass"))
    r = t2.rows(ik)
    assert (t2.morgan[r] == res["morgan"]).all() and (t2.maccs[r] == res["maccs"]).all()
    assert t2.ok.all() and np.allclose(t2.mono_mass[r], res["mono_mass"])
    r1 = t1.rows(ik[:5])
    assert (t1.morgan[r1] == res["morgan"][:5]).all()
    dense = t2.dense(r)
    assert dense.shape == (len(SMILES), 4263)
    assert (dense[:, :4096] == chem.unpack_fp(res["morgan"], 4096)).all()
    assert (dense[:, 4096:] == chem.unpack_fp(res["maccs"], 167)).all()
    t3 = fm.build_targets(["BAD0000000000X"], ["not a smiles"], cfg, cache=cache)
    assert not t3.ok[0]


def _train_frame(n_per: int = 3, seed: int = 0) -> pd.DataFrame:
    from casmi import chem
    from casmi.adducts import TEST_ADDUCTS

    rng = np.random.default_rng(seed)
    rows = []
    for s, smi in enumerate(SMILES):
        M = chem.mono_mass(smi)
        for r in range(n_per):
            a = "[M+H]+" if r % 2 == 0 else "[M-H]-"
            prec = TEST_ADDUCTS[a].precursor_mz(M)
            k = int(rng.integers(8, 40))
            mz = np.sort(rng.uniform(30, prec, k))
            it = rng.uniform(0.01, 1, k)
            rows.append(dict(
                spectrum_id=f"sp{s}_{r}", ms2_mzs=mz.tolist(),
                ms2_normalized_intensities=(it / it.max()).tolist(),
                base_peak_intensity=1e5, adduct=a,
                ionization_mode="positive" if a.endswith("+") else "negative",
                instrument_type="Q Exactive" if r == 2 else None, precursor_mz=prec,
                collision_energy_ev=[10.0 * (r + 1)], normalized_smiles=smi,
                inchikey14=f"IK{s:012d}", molecular_formula=chem.formula(smi),
                ingest_lib="enveda-180" if r < 2 else "gnps", precursor_error_ppm=1.0,
                num_peaks=k))
    return pd.DataFrame(rows)


def test_from_parquet(tmp_path):
    df = _train_frame()
    p = tmp_path / "train.parquet"
    df.to_parquet(p, index=False)
    cfg = Config()
    d = fm.TrainingData.from_parquet(p, cfg, targets_cache=None, batch_rows=7)
    assert len(d) == len(df) and len(d.targets) == len(SMILES)
    assert (d.target_row >= 0).all() and d.targets.ok.all()
    assert d.meta.collision_energy.tolist() == df["collision_energy_ev"].map(lambda v: v[0]).tolist()
    assert [fm.INSTRUMENT_FAMILIES[c] for c in d.meta.instrument_code[:3]] == [
        "timstof", "timstof", "orbitrap"]
    np.testing.assert_array_equal(d.meta.n_peaks_raw, df["num_peaks"].to_numpy())
    assert np.isfinite(d.label_mass).all()
    d2 = fm.TrainingData.from_parquet(p, cfg, targets_cache=None, max_rows=10)
    assert len(d2) == 10
    # already-cleaned input (WP6 spectra files) keeps its raw counts and precursor intensity
    df3 = df.assign(n_peaks_raw=500, prec_intensity=0.25)
    df3.to_parquet(tmp_path / "cleaned.parquet", index=False)
    d3 = fm.TrainingData.from_parquet(tmp_path / "cleaned.parquet", cfg, targets_cache=None)
    assert (d3.meta.n_peaks_raw == 500).all() and (d3.meta.prec_intensity == 0.25).all()


def test_from_library_matches_parquet(tmp_path):
    from casmi.library import Library

    df = _train_frame(seed=4)
    p = tmp_path / "train.parquet"
    df.to_parquet(p, index=False)
    cfg = Config()
    Library.build(p, cfg, out_dir=tmp_path / "lib")
    a = fm.TrainingData.from_library(tmp_path / "lib", cfg, targets_cache=None)
    b = fm.TrainingData.from_parquet(p, cfg, targets_cache=None)
    assert len(a) == len(b) and len(a.targets) == len(b.targets)
    # same spectra (library reorders rows): compare sorted peak multisets per structure
    for ik in sorted(set(b.ik14.tolist())):
        ra, rb = np.flatnonzero(a.ik14 == ik), np.flatnonzero(b.ik14 == ik)
        ma = np.sort(np.concatenate([a.store[i][0] for i in ra]))
        mb = np.sort(np.concatenate([b.store[i][0] for i in rb]))
        np.testing.assert_array_equal(ma, mb)
        assert (a.targets.morgan[a.target_row[ra[0]]] == b.targets.morgan[b.target_row[rb[0]]]).all()
    xa = fm.meta_matrix(a.meta, a.store.n_peaks)
    xb = fm.meta_matrix(b.meta, b.store.n_peaks)
    np.testing.assert_allclose(np.sort(xa, axis=0), np.sort(xb, axis=0), atol=1e-6)
    c = fm.load_training_source(tmp_path / "lib", cfg, max_rows=5, targets_cache=None)
    assert len(c) == 5 and set(c.ik14.tolist()) <= set(a.ik14.tolist())


# --------------------------------------------------------------------------------------
# numpy inference, aggregation, calibration, candidate scores
# --------------------------------------------------------------------------------------


def test_numpy_model_roundtrip(tmp_path, synth):
    spec = fm.ModelSpec.from_config(tiny_cfg())
    params = fm.random_params(spec, seed=3)
    p32 = fm.save_params(params, spec, tmp_path / "m.npz", {"note": "x"})
    p16 = fm.save_params(params, spec, tmp_path / "m16.npz", float16=True)
    m32, m16 = fm.FpModel.load(p32), fm.FpModel.load(p16)
    assert m32.spec == spec and m32.extra == {"note": "x"}
    z = m32.predict_logits(synth.store, synth.meta)
    z16 = m16.predict_logits(synth.store, synth.meta)
    assert z.shape == (len(synth), 4263) and np.isfinite(z).all()
    assert np.abs(z - z16).max() < 0.05 * np.abs(z).max()
    sp = fm.featurize_sparse(synth.store, synth.meta.precursor_mz, spec)
    x = fm.meta_matrix(synth.meta, synth.store.n_peaks, spec)
    # chunking does not change the result
    np.testing.assert_allclose(fm.predict_numpy(params, spec, sp, x, chunk=7), z, atol=1e-5)
    with np.load(p32, allow_pickle=False) as f:   # no pickled objects in the file
        assert "spec_json" in f.files


def test_molecule_logits_weights():
    z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    mid = ["b", "a", "b", "b"]
    bpi = np.array([1e6, 1e4, np.nan, 1e9])
    pol = np.array([1, -1, -1, 1])
    mp = fm.molecule_logits(z, mid, bpi, pol)
    assert mp.molecule_id.tolist() == ["a", "b"]
    w = np.array([2.0, 4 / 3, 1.0, 2.0])               # clip(log10 bpi, 3, 6) / 3
    np.testing.assert_allclose(mp.logits[0], z[1])
    np.testing.assert_allclose(mp.logits[1], (w[[0, 2, 3], None] * z[[0, 2, 3]]).sum(0) / w[[0, 2, 3]].sum(),
                               rtol=1e-6)
    np.testing.assert_allclose(mp.logits_pos[1], (2 * z[0] + 2 * z[3]) / 4, rtol=1e-6)
    np.testing.assert_allclose(mp.logits_neg[1], z[2])
    assert np.isnan(mp.logits_pos[0]).all() and mp.n_pos.tolist() == [0, 2]
    pol_str = np.array(["positive", "negative", "negative", "positive"], dtype=object)
    mp2 = fm.molecule_logits(z, mid, bpi, pol_str)
    np.testing.assert_allclose(mp2.logits_pos, mp.logits_pos)
    assert mp.row("b") == 1


def test_fit_calibration_recovers_platt():
    rng = np.random.default_rng(0)
    n, K = 4000, 6
    z = rng.normal(0, 2, (n, K))
    a_true = np.array([0.5, 1.0, 1.5, 2.0, 0.8, 1.2])
    b_true = np.array([-1.0, 0.0, 0.5, -0.5, 1.0, -2.0])
    y = (rng.random((n, K)) < fm.sigmoid(a_true * z + b_true)).astype(float)
    cal = fm.fit_calibration(z, y, l2=1e-3)
    np.testing.assert_allclose(cal.a, a_true, atol=0.15)
    np.testing.assert_allclose(cal.b, b_true, atol=0.15)
    ident = fm.BitCalibration.identity(3)
    np.testing.assert_allclose(ident.transform(np.array([0.1, 0.5, 0.9])), [0.1, 0.5, 0.9])


def test_calibration_save_load(tmp_path):
    cal = fm.BitCalibration(np.array([1.0, 2.0]), np.array([0.5, -0.5]))
    p = cal.save(tmp_path / "c.npz", {"n": 3})
    back = fm.BitCalibration.load(p)
    np.testing.assert_array_equal(back.a, cal.a)
    np.testing.assert_array_equal(back.b, cal.b)


def test_bit_dot_packed_equals_unpacked():
    rng = np.random.default_rng(0)
    bits = rng.random((50, 167)) < 0.3
    packed = np.packbits(bits, axis=1)
    w = rng.normal(size=(167, 3))
    ref = bits.astype(float) @ w
    np.testing.assert_allclose(fm.bit_dot(packed, w), ref, atol=1e-10)
    np.testing.assert_allclose(fm.bit_dot(bits, w), ref, atol=1e-10)
    np.testing.assert_allclose(fm.bit_dot(packed, w[:, 0], chunk=7), ref[:, 0], atol=1e-10)
    with pytest.raises(ValueError):
        fm.bit_dot(packed[:, :20], w)


def test_candidate_scores_reference():
    from casmi import chem

    cfg = Config()
    rng = np.random.default_rng(1)
    p = rng.beta(0.3, 3.0, 4263)
    mf, ok = chem.morgan_fps(SMILES)
    kf, _ = chem.maccs_fps(SMILES)
    assert ok.all()
    s = fm.candidate_scores(p, mf, kf, cfg)
    sb = fm.candidate_scores(p, chem.unpack_fp(mf, 4096), chem.unpack_fp(kf, 167), cfg)
    for k in s:
        np.testing.assert_allclose(s[k], sb[k], rtol=1e-12, atol=1e-9)
    c = np.concatenate([chem.unpack_fp(mf, 4096), chem.unpack_fp(kf, 167)], axis=1).astype(float)
    pc = np.clip(p, 0.01, 0.99)
    ll = (c * np.log(pc) + (1 - c) * np.log(1 - pc)).sum(1)
    cos = c @ p / (np.linalg.norm(p) * np.linalg.norm(c, axis=1))
    tan = c @ p / (p.sum() + c.sum(1) - c @ p)
    np.testing.assert_allclose(s["ll"], ll, rtol=1e-10)
    np.testing.assert_allclose(s["cos"], cos, rtol=1e-10)
    np.testing.assert_allclose(s["tanimoto"], tan, rtol=1e-10)
    ck, pk = c[:, 4096:], p[4096:]
    np.testing.assert_allclose(s["maccs_cos"], ck @ pk / (np.linalg.norm(pk) * np.linalg.norm(ck, axis=1)),
                               rtol=1e-10)
    pkc = np.clip(pk, 0.01, 0.99)
    np.testing.assert_allclose(s["maccs_ll"], (ck * np.log(pkc) + (1 - ck) * np.log(1 - pkc)).sum(1),
                               rtol=1e-10)
    order = np.argsort(-ll, kind="stable")
    assert (s["ll_rank"][order] == np.arange(1, len(SMILES) + 1)).all()
    # the true fingerprint as prediction scores itself best
    s2 = fm.candidate_scores(np.clip(c[3], 0.02, 0.98), mf, kf, cfg)
    assert s2["ll_rank"][3] == 1 and np.argmax(s2["cos"]) == 3 and np.argmax(s2["tanimoto"]) == 3
    # identity calibration changes nothing; no MACCS -> NaN MACCS scores
    s3 = fm.candidate_scores(p, mf, kf, cfg, fm.BitCalibration.identity(4263))
    np.testing.assert_allclose(s3["ll"], s["ll"], rtol=1e-9)
    s4 = fm.candidate_scores(p, mf, None, cfg)
    assert np.isnan(s4["maccs_ll"]).all()
    cm = c[:, :4096]
    np.testing.assert_allclose(s4["ll"], (cm * np.log(pc[:4096]) + (1 - cm) * np.log(1 - pc[:4096])).sum(1),
                               rtol=1e-10)


def test_predict_molecules_on_synthetic_frame(tmp_path):
    spec = fm.ModelSpec.from_config(tiny_cfg())
    path = fm.save_params(fm.random_params(spec, seed=1), spec, tmp_path / "m.npz")
    model = fm.FpModel.load(path)
    df = fm.synthetic_test_frame(30, seed=2, n_molecules=8)
    mp = fm.predict_molecules(model, df)
    assert mp.molecule_id.tolist() == sorted(df["molecule_id"].unique())
    assert mp.logits.shape == (len(mp.molecule_id), 4263)
    z = model.predict_frame(df)
    i = mp.row(mp.molecule_id[0])
    sel = (df["molecule_id"] == mp.molecule_id[0]).to_numpy()
    w = fm.spectrum_weights(df["base_peak_intensity"].to_numpy()[sel])
    np.testing.assert_allclose(mp.logits[i], (w[:, None] * z[sel]).sum(0) / w.sum(), rtol=1e-5,
                               atol=1e-5)


# --------------------------------------------------------------------------------------
# No torch on the inference path
# --------------------------------------------------------------------------------------


def _run_py(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASMI_")}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                          capture_output=True, text=True, timeout=300, cwd=str(ROOT))


def test_import_and_inference_without_torch(tmp_path):
    code = f"""
        import sys
        sys.modules["torch"] = None          # any `import torch` now raises ImportError
        import numpy as np
        import casmi.fpmodel as fm
        from casmi import chem
        spec = fm.ModelSpec.from_config(hidden=32, meta_hidden=8, n_blocks=1)
        path = fm.save_params(fm.random_params(spec), spec, {str(tmp_path / 'nt.npz')!r})
        model = fm.FpModel.load(path)
        df = fm.synthetic_test_frame(20, seed=0)
        mp = fm.predict_molecules(model, df)
        p = mp.proba()[0]
        mf, _ = chem.morgan_fps(["CCO", "c1ccccc1O"])
        kf, _ = chem.maccs_fps(["CCO", "c1ccccc1O"])
        s = fm.candidate_scores(p, mf, kf)
        cal = fm.fit_calibration(mp.logits, (mp.logits > 0).astype(float))
        assert np.isfinite(s["ll"]).all() and len(s["cos"]) == 2
        try:
            import torch  # noqa: F401
            raise SystemExit("torch import was not blocked")
        except ImportError:
            pass
        print("ok")
    """
    out = _run_py(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_import_does_not_load_torch():
    out = _run_py("import sys, casmi.fpmodel; print('torch' in sys.modules)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------------------
# torch: equality, overfitting, checkpoints, script
# --------------------------------------------------------------------------------------


def test_numpy_equals_torch(tmp_path, synth, public_test_parquet):
    pytest.importorskip("torch")
    cfg = tiny_cfg(fpm_hidden=96, fpm_n_blocks=2, fpm_dropout=0.1)
    res, net = fm.train(synth, cfg, steps=15, return_net=True, out_npz=tmp_path / "eq.npz")
    spec = fm.ModelSpec.from_config(cfg)
    model = fm.FpModel.load(res.npz_path)
    for store, meta in [(synth.store, synth.meta),
                        fm.spectra_from_frame(pd.read_parquet(public_test_parquet), cfg)]:
        sp = fm.featurize_sparse(store, meta.precursor_mz, spec)
        x = fm.meta_matrix(meta, store.n_peaks, spec)
        zt = fm.torch_predict(net, sp, x)
        zn = model.predict_logits(store, meta)
        assert np.abs(zt - zn).max() < 1e-4, np.abs(zt - zn).max()


def test_tiny_model_overfits_256_examples(synth):
    pytest.importorskip("torch")
    assert len(synth) == 256
    cfg = tiny_cfg(fpm_hidden=128, fpm_dropout=0.0, fpm_lr=3e-3,
                   fpm_aug_consensus_prob=0.0, fpm_aug_peak_drop=0.0,
                   fpm_aug_intensity_sigma=0.0, fpm_aug_noise_peaks_max=0)
    res, net = fm.train(synth, cfg, steps=300, augment=False, return_net=True)
    spec = fm.ModelSpec.from_config(cfg)
    sp, x, y = fm.make_batch(synth, np.arange(256), spec, cfg, augment=False)
    p = fm.sigmoid(fm.torch_predict(net, sp, x))
    cos = (p * y).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1))
    print(f"\noverfit 256 synthetic examples: final train loss {res.final_loss:.4f}, "
          f"mean cosine {cos.mean():.4f}")
    assert res.final_loss < 0.05 and cos.mean() > 0.97


def test_train_checkpoint_resume(tmp_path, synth):
    pytest.importorskip("torch")
    import torch

    cfg = tiny_cfg(fpm_dropout=0.0, fpm_prefetch=True, fpm_threads=1, fpm_log_every_steps=5)
    a, net_a = fm.train(synth, cfg, steps=20, return_net=True, checkpoint=tmp_path / "a.pt",
                        curve_csv=tmp_path / "a.csv")
    fm.train(synth, cfg, steps=10, checkpoint=tmp_path / "b.pt")   # stops at 10 of 10
    st = torch.load(tmp_path / "b.pt", weights_only=False)
    assert st["step"] == 10
    b, net_b = fm.train(synth, cfg, steps=20, return_net=True, checkpoint=tmp_path / "b.pt",
                        resume=True)
    assert a.steps == b.steps == 20
    for (k, va), (_, vb) in zip(net_a.state_dict().items(), net_b.state_dict().items()):
        np.testing.assert_allclose(va.numpy(), vb.numpy(), atol=1e-5, err_msg=k)
    curve = pd.read_csv(tmp_path / "a.csv")
    assert curve["step"].tolist() == [5, 10, 15, 20] and np.isfinite(curve["loss"]).all()


def test_time_budget_sets_total_steps(synth):
    pytest.importorskip("torch")
    cfg = tiny_cfg(fpm_warmup_steps=5)
    res = fm.train(synth, cfg, hours=6 / 3600, measure_steps=10)
    assert res.steps >= 20 and res.seconds < 30
    assert res.total_steps >= res.steps


def test_train_script_synthetic_smoke(tmp_path):
    pytest.importorskip("torch")
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text("fpm_hidden: 32\nfpm_meta_hidden: 8\nfpm_n_blocks: 1\n"
                        "fpm_batch_size: 64\nfpm_threads: 2\nfpm_warmup_steps: 2\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASMI_")}
    env["CASMI_LOGS"] = str(tmp_path / "logs")
    env["CASMI_PROCESSED"] = str(tmp_path / "processed")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_fpmodel.py"),
                          "--config", str(cfg_path), "--synthetic", "2000", "--steps", "6",
                          "--tag", "t_holdout", "--out-dir", str(tmp_path / "models"),
                          "--n-jobs", "1"], env=env, capture_output=True, text=True,
                         timeout=600, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-3000:]
    m = tmp_path / "models"
    assert (m / "fp_t_holdout.npz").is_file() and (m / "fp_t_holdout_ckpt.pt").is_file()
    assert (m / "fp_t_calibration.npz").is_file()
    assert (tmp_path / "logs" / "fp_t_holdout_curve.csv").is_file()
    model = fm.FpModel.load(m / "fp_t_holdout.npz")
    assert model.extra["steps"] == 6 and model.spec.hidden == 32

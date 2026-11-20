"""Tests for casmi.encoder (contrastive transformer encoder) on tiny synthetic data, CPU."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi import encoder as E
from casmi.config import Config

ROOT = Path(__file__).resolve().parents[1]

TINY = dict(enc_d_model=32, enc_n_layers=2, enc_n_heads=2, enc_d_ff=64, enc_n_freq=8,
            enc_emb_dim=32, enc_tower_hidden=64, enc_max_tokens=32, enc_batch_size=16,
            enc_n_neg=8, enc_warmup_steps=10, enc_lr=3e-3, enc_log_every_steps=10,
            enc_eval_every_min=1000.0)


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_import_without_torch():
    code = ("import sys\n"
            "class B:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'torch':\n"
            "            raise ImportError('blocked')\n"
            "sys.meta_path.insert(0, B())\n"
            "import casmi.encoder as E, casmi.pipeline as P\n"
            "assert 'encoder' in P.stage_names()\n"
            "assert 'torch' not in sys.modules\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_build_queries_groups_and_singles():
    # molecule 0: three [M+H]+ family spectra, one [M-H]-; molecule 1: one spectrum;
    # molecule 2: six spectra of one family (two chunks); molecule 3: only an empty spectrum.
    mol = np.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 3])
    pol = np.array([1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1])
    add = np.array([0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0])   # [M+H]+, [M-H2O+H]+, [M-H]-
    npk = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0])
    ce = np.array([20, 40, 60, 20, 20, 10, 20, 30, 40, 50, 60, 20], float)
    qs = E.build_queries(mol, pol, add, npk, 4, ce)
    m0 = qs.members[qs.mol == 0]
    k0 = qs.kind[qs.mol == 0]
    groups = [sorted(r[r >= 0].tolist()) for r, k in zip(m0, k0) if k == 0]
    assert sorted(groups) == [[0, 1, 2], [3]]
    assert int((k0 == 1).sum()) == 3                       # singles of the 3-spectrum group
    assert int((qs.mol == 1).sum()) == 1                   # one spectrum: no extra single
    g2 = [r[r >= 0] for r, k in zip(qs.members[qs.mol == 2], qs.kind[qs.mol == 2]) if k == 0]
    assert len(g2) == 2 and sorted(np.concatenate(g2).tolist()) == [5, 6, 7, 8, 9, 10]
    assert all(len(r) <= 4 for r in g2)
    assert qs.members[qs.mol == 3].tolist() == [[11, -1, -1, -1]]
    sel = E.query_mask(qs, "singles")
    assert sel[qs.mol == 1].all() and sel[qs.mol == 3].all()   # fallback: no single queries
    assert not sel[(qs.mol == 0) & (qs.kind == 0)].any()


def test_rank_metrics_ties_and_absent():
    scores = np.array([0.9, 0.5, 0.5, 0.1, 0.3, 0.2, 0.4])
    pos = np.array([False, True, False, False, False, False, False])
    off = np.array([0, 4, 7])               # molecule 1 has no positive
    m = E.rank_metrics(scores, pos, off)
    assert m["rank"].tolist() == [3, 0]     # the tie counts against the truth
    assert m["mrr"] == pytest.approx((1 / 3) / 2)
    assert m["recall"] == 0.5 and m["top1"] == 0.0 and m["top5"] == 0.5


def test_lib_weights_shares():
    libs = np.array(["a"] * 4 + ["b"] * 2 + ["c"] * 3, dtype=object)
    st = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6])
    w = E.lib_weights(libs, st, (("a", 0.5),))
    assert w.sum() == pytest.approx(1.0)
    assert w[libs == "a"].sum() == pytest.approx(0.5)
    rest = {"b": 2 ** 0.75, "c": 3 ** 0.75}
    assert w[libs == "b"].sum() == pytest.approx(0.5 * rest["b"] / sum(rest.values()))
    # structure 0 has three groups in "a": each weighs 1/sqrt(3) of structure 1's group
    assert w[0] / w[3] == pytest.approx(1 / np.sqrt(3))


def test_unpack_and_hash_match_chem_packing():
    from casmi.chem import fp_mol, morgan_fp

    fp = morgan_fp(fp_mol("CC(=O)Oc1ccccc1C(=O)O"))
    bits = E.unpack_bits(fp[None])[0]
    assert bits.shape == (4096,)
    assert np.array_equal(np.packbits(bits), fp)
    h = E.fp_hash(np.stack([fp, fp, np.zeros_like(fp)]))
    assert h[0] == h[1] != h[2]


def test_negatives_window_exclusions_cap_topup():
    mod = _script("build_encoder_bundle")
    uni_mass = np.array([100.0, 100.0005, 100.0008, 100.5, 101.0, 300.0, 300.001, 300.002])
    uni_hash = np.array([1, 2, 1, 4, 5, 6, 7, 8], dtype=np.uint64)
    st_mass = np.array([100.0, 300.0])
    st_uni = np.array([0, 5])
    st_hash = np.array([1, 6], dtype=np.uint64)
    off, idx, stats = mod.negatives(st_mass, st_uni, st_hash, uni_mass, uni_hash, ppm=10.0,
                                    min_da=0.002, n_max=1, n_min=1, seed=0)
    a = idx[off[0]:off[1]].tolist()
    assert a == [1]                         # 0 is itself, 2 has an identical fingerprint
    b = idx[off[1]:off[2]].tolist()
    assert len(b) == 1 and b[0] in (6, 7)   # capped at n_max
    off, idx, stats = mod.negatives(st_mass[:1], st_uni[:1], st_hash[:1], uni_mass, uni_hash,
                                    ppm=1.0, min_da=0.0001, n_max=10, n_min=2, seed=0)
    assert idx.tolist() == [1, 3]           # empty window -> nearest masses, identical skipped
    assert stats["topped_up"] == 1


def test_eval_pack_subset(tmp_path):
    root, _ = E.synthetic_bundle(tmp_path / "b", n_struct=16, n_np=60, n_eval=6)
    pack = E.Bundle.load(root).eval_packs()["SYN"]
    sub = pack.subset(3, seed=1)
    assert len(sub.mol) == 3
    assert sub.pool_off[-1] == len(sub.pool_idx) == len(sub.pool_pos)
    assert set(np.unique(sub.q["q_mol"]).tolist()) == {0, 1, 2}
    assert sub.q["q_off"][-1] == len(sub.q["q_mz"])
    for i, mid in enumerate(sub.mol["molecule_id"]):
        j = int(np.flatnonzero(pack.mol["molecule_id"] == mid)[0])
        assert np.array_equal(sub.pool_idx[sub.pool_off[i]:sub.pool_off[i + 1]],
                              pack.pool_idx[pack.pool_off[j]:pack.pool_off[j + 1]])


def test_tokens_keep_most_intense_and_masks():
    torch = pytest.importorskip("torch")
    mz = torch.tensor([[[100.0, 150.0, 200.0, 0.0], [120.0, 130.0, 0.0, 0.0]]])
    it = torch.tensor([[[1.0, 0.2, 0.5, 0.0], [0.3, 1.0, 0.0, 0.0]]])
    valid = it > 0
    mm = {"prec": torch.tensor([[300.0, 300.0]]), "ce": torch.tensor([[20.0, float("nan")]]),
          "ne": torch.tensor([[1.0, 1.0]]), "mvalid": torch.tensor([[True, True]])}
    qm = {"adduct": torch.tensor([0]), "polarity": torch.tensor([1]), "inst": torch.tensor([0]),
          "sim": torch.tensor([0]), "mass": torch.tensor([299.0])}
    tok = E.assemble_tokens(mz, it, valid, mm, qm, max_tokens=3)
    assert sorted(tok["mz"][0].tolist()) == [100.0, 130.0, 200.0]
    assert tok["valid"].all()
    ce = tok["ce"][0]
    assert torch.isnan(ce[tok["mz"][0] == 130.0]).all() and (ce[tok["mz"][0] == 100.0] == 20).all()
    assert float(tok["mean_ce"][0]) == 20.0 and float(tok["n_members"][0]) == 2.0
    g = torch.Generator().manual_seed(0)
    aug = {"mz_ppm": 3.0, "peak_drop": 0.5, "int_sigma": 0.3, "noise_prob": 1.0, "noise_max": 4,
           "noise_int_min": 0.01, "noise_int_max": 0.02}
    tok = E.assemble_tokens(mz, it, valid, mm, qm, max_tokens=8, aug=aug, gen=g)
    assert torch.isfinite(tok["mz"]).all() and torch.isfinite(tok["int"]).all()
    assert float(tok["int"].max()) == pytest.approx(1.0)
    assert int(tok["valid"].sum()) >= 2     # each member keeps its base peak


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("torch")
    import torch

    torch.set_num_threads(1)
    d = tmp_path_factory.mktemp("enc")
    root, npfp = E.synthetic_bundle(d / "bundle", n_struct=48, n_np=200)
    cfg = Config().replace(**TINY)
    out = {}
    for arm, steps in (("bce", 60), ("focal_con", 300)):
        out[arm] = E.train(root, npfp, d / arm, cfg, arm=arm, hours=1.0, device="cpu",
                           max_steps=steps, resume=False)
    return d, root, npfp, cfg, out


def test_train_outputs_and_learning(trained):
    d, root, npfp, cfg, out = trained
    for arm in ("bce", "focal_con"):
        o = d / arm
        assert (o / f"encoder_{arm}.pt").is_file() and (o / "summary.json").is_file()
        log = pd.read_csv(o / "train_log.csv")
        assert np.isfinite(log["loss"]).all() and len(log) >= 5
        ev = [json.loads(line) for line in (o / "eval.jsonl").read_text().splitlines()]
        assert ev[-1]["tag"] == "final"
    s = out["focal_con"]
    assert s["steps"] == 300
    assert set(k.split(":")[1].split("/")[0] for k in s["final"]) == {"con", "fpcos", "bern"}
    assert "SYN:con/all" not in out["bce"]["final"]
    # random ranking over pools of about 20 gives MRR near 0.2; the learnable task beats it
    assert s["final"]["SYN:con/all"] > 0.3


def test_resume_continues(trained):
    d, root, npfp, cfg, _ = trained
    o = d / "resume"
    E.train(root, npfp, o, cfg, arm="bce", hours=1.0, device="cpu", max_steps=20, resume=False)
    s = E.train(root, npfp, o, cfg, arm="bce", hours=1.0, device="cpu", max_steps=30, resume=True)
    assert s["steps"] == 30


def test_model_load_score_and_frame(trained):
    d, root, npfp, cfg, _ = trained
    m = E.EncoderModel.load(d / "focal_con" / "encoder_focal_con.pt")
    assert m.spec.d_model == 32 and m.net.tower is not None
    pack = E.Bundle.load(root).eval_packs()["SYN"]
    np_fp = np.load(npfp)
    ev = E.evaluate_packs(m, {"SYN": pack}, np_fp, cfg, modes=("all",), return_ranks=True)
    assert ev["SYN"]["con/all/rank"].shape == (len(pack.mol),)
    # test-schema frame -> molecule encodings -> candidate scores
    q = pack.q
    rows = []
    for i in range(len(q["q_mol"])):
        a, b = int(q["q_off"][i]), int(q["q_off"][i + 1])
        rows.append({"molecule_id": f"M{int(q['q_mol'][i])}", "spectrum_id": f"s{i}",
                     "ms2_mzs": q["q_mz"][a:b].astype(float).tolist(),
                     "ms2_normalized_intensities": q["q_int"][a:b].astype(float).tolist(),
                     "base_peak_intensity": 1e5, "adduct": "[M+H]+", "ionization_mode": "positive",
                     "instrument_type": "timsTOF", "precursor_mz": float(q["q_prec"][i]),
                     "collision_energy_orig": "20", "collision_energy_ev": [float(q["q_ce"][i])],
                     "collision_energy_orig_units": "eV"})
    df = pd.DataFrame(rows)
    enc = m.encode_frame(df, cfg)
    assert enc["z"].shape == (len(pack.mol), 32) and enc["proba"].shape == (len(pack.mol), 4096)
    assert np.allclose(np.linalg.norm(enc["z"], axis=1), 1.0, atol=1e-5)
    cand = np_fp[pack.pool_idx[pack.pool_off[0]:pack.pool_off[1]]]
    sc = m.score(enc["z"][0], enc["proba"][0], cand, cfg)
    assert set(sc) == {"con", "fpcos", "bern"} and all(len(v) == len(cand) for v in sc.values())
    # the frame route gives the same embedding as the pack route for molecule 0
    qs = E.build_queries(q["q_mol"], q["q_pol"], q["q_add"], np.diff(q["q_off"]), 4, q["q_ce"])
    meta = {"prec": q["q_prec"], "ce": q["q_ce"], "ne": q["q_ne"], "add": q["q_add"],
            "pol": q["q_pol"], "inst": q["q_inst"]}
    from casmi.io import mass_tables

    mol_mass = mass_tables(df)[0]["M"].to_numpy()
    assert len(mol_mass) == len(pack.mol)
    z, lg = m.encode_store(q["q_mz"], q["q_int"].astype(np.float32), q["q_off"], meta, qs, mol_mass)
    ze, _ = m.aggregate(z, lg, qs, len(pack.mol), "all")
    assert np.allclose(ze[0], enc["z"][0], atol=1e-4)


def test_pipeline_stage_adds_columns(trained, tmp_path):
    d, root, npfp, cfg, _ = trained
    from casmi import pipeline

    pack = E.Bundle.load(root).eval_packs()["SYN"]
    q = pack.q
    rows = []
    for i in range(2):
        a, b = int(q["q_off"][i]), int(q["q_off"][i + 1])
        rows.append({"molecule_id": "M0", "spectrum_id": f"s{i}",
                     "ms2_mzs": q["q_mz"][a:b].astype(float).tolist(),
                     "ms2_normalized_intensities": q["q_int"][a:b].astype(float).tolist(),
                     "base_peak_intensity": 1e5, "adduct": "[M+H]+", "ionization_mode": "positive",
                     "instrument_type": "timsTOF", "precursor_mz": float(q["q_prec"][i]),
                     "collision_energy_orig": "20", "collision_energy_ev": [20.0],
                     "collision_energy_orig_units": "eV"})
    df = pd.DataFrame(rows)
    from casmi.io import mass_tables

    mols, spectra = mass_tables(df)
    pool = pd.DataFrame({"molecule_id": ["M0"] * 3, "ik14": ["A", "B", "C"],
                         "smiles": ["CCO", "c1ccccc1O", "CC(=O)O"],
                         "np_row": pd.array([-1, pd.NA, -1], dtype="Int64")})
    st = pipeline.State(test_df=df, molecules=mols, spectra=spectra, pool=pool)
    cfg2 = cfg.replace(enc_model_path=str(d / "focal_con" / "encoder_focal_con.pt"))
    st = pipeline.get_stage("encoder")(st, cfg2)
    for c in E.ENCODER_COLUMNS:
        assert c in st.pool.columns
    assert np.isfinite(st.pool["enc_con"]).all() and np.isfinite(st.pool["enc_bern"]).all()
    assert sorted(st.pool["enc_con_rank"].tolist()) == [1.0, 2.0, 3.0]


def test_metadata_only_queries(trained):
    d, root, npfp, cfg, _ = trained
    m = E.EncoderModel.load(d / "focal_con" / "encoder_focal_con.pt")
    n = 3
    qs = E.build_queries(np.arange(n), np.ones(n), np.zeros(n), np.zeros(n), 4)
    meta = {"prec": np.full(n, 300.0), "ce": np.full(n, np.nan), "ne": np.zeros(n),
            "add": np.zeros(n), "pol": np.ones(n), "inst": np.zeros(n)}
    z, lg = m.encode_store(np.zeros(0, np.float32), np.zeros(0, np.float32),
                           np.zeros(n + 1, np.int64), meta, qs, np.full(n, 299.0))
    assert z.shape == (n, 32) and np.isfinite(z).all() and np.isfinite(lg).all()


def test_extend_bundle_with_simulated_spectra(tmp_path):
    pytest.importorskip("torch")
    root, npfp = E.synthetic_bundle(tmp_path / "b", n_struct=16, n_np=60, n_eval=6)
    b = E.Bundle.load(root)
    np_fp = np.load(npfp)
    np_mass = np.sort(np.random.default_rng(0).uniform(200, 400, 60))   # unused rows only
    np_mass[np.asarray(b.st["st_uni"])[np.asarray(b.st["st_uni"]) < 60]] = \
        np.asarray(b.st["st_mass"])[np.asarray(b.st["st_uni"]) < 60]
    rows = np.array([0, 0, 0, 55, 55, 57])      # row 0 is a bundle structure; 55, 57 are new
    mz = np.tile(np.array([60.0, 80.0, 120.0], np.float32), len(rows))
    it = np.tile(np.array([0.2, 1.0, 0.5], np.float32), len(rows))
    off = np.arange(0, 3 * len(rows) + 1, 3)
    meta = {"prec": np_mass[rows] + 1.007, "mass": np_mass[rows], "ce": np.full(len(rows), 40.0),
            "ne": np.ones(len(rows)), "add": np.zeros(len(rows)), "pol": np.ones(len(rows)),
            "inst": np.zeros(len(rows))}
    info = E.extend_bundle(root, tmp_path / "ext", mz=mz, intensity=it, offsets=off, meta=meta,
                           np_row=rows, np_fp=np_fp, np_mass=np_mass, share=0.25)
    x = E.Bundle.load(tmp_path / "ext")
    n0, s0 = len(b.spec_off) - 1, len(b.st["st_uni"])
    assert len(x.spec_off) - 1 == n0 + 6 and len(x.st["st_uni"]) == s0 + 2
    assert x.s["spec_sim"][n0:].tolist() == [1] * 6 and x.s["spec_sim"][:n0].sum() == 0
    assert x.s["spec_struct"][n0:n0 + 3].tolist() == [int(np.flatnonzero(np.asarray(b.st["st_uni"]) == 0)[0])] * 3
    assert np.array_equal(x.st["st_fp"][s0], np_fp[55])
    sim_groups = x.g["grp_lib"] == x.info["libs"].index("simulated")
    assert x.g["grp_w"][sim_groups].sum() == pytest.approx(0.25)
    assert x.g["grp_w"].sum() == pytest.approx(1.0)
    for st in (s0, s0 + 1):                  # new structures: negatives exclude themselves
        neg = x.neg_idx[x.neg_off[st]:x.neg_off[st + 1]]
        assert len(neg) > 0 and int(x.st["st_uni"][st]) not in neg.tolist()
    assert set(x.eval_packs()) == {"SYN"} and info["extension"]["n_new_structures"] == 2
    cfg = Config().replace(**TINY)
    s = E.train(tmp_path / "ext", npfp, tmp_path / "o", cfg, arm="focal_con", hours=1.0,
                device="cpu", max_steps=5, resume=False)
    assert s["steps"] == 5

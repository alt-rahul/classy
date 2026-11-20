"""C2-PC evaluation of trained encoders over full PubChem windows (ROADMAP R4 gate).

    python scripts/eval_encoder_pc.py --bundle data/processed/encoder/bundle_v2 \\
        --model bce=models/encoder/v1/encoder_bce.pt --model focal_con=... \\
        --np-dir data/processed/db/np --pubchem-dir data/processed/db/pubchem \\
        --n-jobs 4 --budget-min 600 --out c2pc.json

Molecules: the ``pc_syn_tims`` pack of the bundle (SYN-TIMS holdout molecules with no
spectra in any other training library, i.e. without public spectra; queries are the
validation queries, cleaned). Condition C2: the encoders never trained on them; the pool
is every NP-tier and PubChem-tier row within ``mass_tol_ppm`` of M (``casmi.db`` lookup,
NP rows win per ik14), truth = rows whose ik14 is a truth key of the molecule (ik14 group
and same-mkey DB keys). Candidate fingerprints: ``np_fp.npy`` for NP rows, else
``chem.morgan_fps`` (std_mol + InChI round trip, as for targets) in ``--n-jobs`` processes.

By default only molecules whose structure is in PubChem (``in_pubchem``; the class 2
definition) are used. Molecules are processed in a fixed hash order, optionally split into
``--n-shards`` interleaved shards (one per Kaggle CPU kernel; ``--merge`` combines their
``.ranks.parquet`` files), until ``--max-mol`` or ``--budget-min`` is reached, so a partial
run is an unbiased subsample. Metrics per model and score (con,
fpcos, bern; query mode ``enc_query_mode``): MRR@25, top-1, top-25, top-200 recall, truth in
pool; per-molecule ranks go to ``<out>.ranks.parquet``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "casmi").is_dir():
    sys.path.insert(0, str(ROOT / "src"))


def _fp_chunk(smiles: list) -> np.ndarray:
    from casmi.chem import morgan_fps

    return morgan_fps(smiles)[0]


def _rank(scores: np.ndarray, pos: np.ndarray) -> int:
    if not pos.any():
        return 0
    s = np.nan_to_num(scores, nan=-np.inf)
    best = s[pos].max()
    return 1 + int(((s >= best) & ~pos).sum())


def metrics(rank: np.ndarray) -> dict:
    ok = rank > 0
    rr = np.where(ok & (rank <= 25), 1.0 / np.maximum(rank, 1), 0.0)
    n = len(rank)
    f = (lambda x: round(float(x), 4)) if n else (lambda x: float("nan"))
    return {"n": int(n), "mrr": f(rr.mean() if n else 0), "top1": f((ok & (rank <= 1)).mean() if n else 0),
            "top5": f((ok & (rank <= 5)).mean() if n else 0),
            "top25": f((ok & (rank <= 25)).mean() if n else 0),
            "top200": f((ok & (rank <= 200)).mean() if n else 0), "recall": f(ok.mean() if n else 0)}


def summarize(ranks: pd.DataFrame) -> dict:
    res = {"n_molecules": int(len(ranks)),
           "median_pool": float(ranks["n_pool"].median()) if len(ranks) else 0.0,
           "truth_in_pool": float((ranks["n_pos"] > 0).mean()) if len(ranks) else 0.0, "metrics": {}}
    for c in ranks.columns:
        if ":" in c:
            res["metrics"][c] = metrics(ranks[c].to_numpy(np.int64))
    return res


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=None)
    p.add_argument("--model", action="append", default=[], metavar="ARM=PATH")
    p.add_argument("--np-dir", default=None)
    p.add_argument("--pubchem-dir", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--all-molecules", action="store_true",
                   help="also molecules whose structure is not in PubChem (default: in_pubchem only)")
    p.add_argument("--merge", nargs="+", default=None, metavar="RANKS",
                   help="combine shard .ranks.parquet files into --out and exit")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--threads", type=int, default=4, help="torch threads")
    p.add_argument("--max-mol", type=int, default=0)
    p.add_argument("--budget-min", type=float, default=0.0)
    p.add_argument("--batch-mol", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.merge:
        ranks = pd.concat([pd.read_parquet(f) for f in args.merge], ignore_index=True)
        ranks = ranks.drop_duplicates("molecule_id").sort_values("molecule_id", kind="mergesort")
        res = summarize(ranks)
        Path(args.out).write_text(json.dumps(res, indent=1))
        ranks.to_parquet(Path(args.out).with_suffix(".ranks.parquet"), index=False)
        print(json.dumps(res, indent=1))
        return res
    import torch

    from casmi import encoder as E
    from casmi.chem import mp_context
    from casmi.config import Config
    from casmi.db import CandidateDB
    from casmi.log import get_logger, log_run

    log = get_logger("eval_encoder_pc")
    log_run(log, args)
    torch.set_num_threads(args.threads)
    cfg = Config()
    t0 = time.time()
    pack = E.EvalPack.load(Path(args.bundle) / "pc_syn_tims", "C2-PC")
    if not args.all_molecules and "in_pubchem" in pack.mol.columns:
        pack = pack.take(np.flatnonzero(pack.mol["in_pubchem"].astype(bool).to_numpy()))
    mol = pack.mol
    n_eligible = int(len(mol))
    keys = [set(str(k).split(";")) for k in mol["truth_keys"]]
    order = np.argsort([hashlib.sha1(f"{cfg.seed}|pc|{m}".encode()).hexdigest() for m in mol["mkey"]],
                       kind="mergesort")
    order = order[args.shard::max(1, args.n_shards)]
    if args.max_mol > 0:
        order = order[:args.max_mol]
    pack = pack.take(order)                   # molecules in sorted index order
    mol = pack.mol
    keys = [set(str(k).split(";")) for k in mol["truth_keys"]]
    order = np.argsort([hashlib.sha1(f"{cfg.seed}|pc|{m}".encode()).hexdigest() for m in mol["mkey"]],
                       kind="mergesort")
    dbh = CandidateDB(np_path=Path(args.np_dir) / "np.parquet", pubchem_dir=args.pubchem_dir, cfg=cfg)
    np_fp = np.load(Path(args.np_dir) / "np_fp.npy", mmap_mode="r")
    models = {}
    for spec in args.model:
        arm, _, path = spec.partition("=")
        models[arm] = E.EncoderModel.load(path, args.device)
    # molecule encodings (all molecules of the pack at once; cheap)
    q = pack.q
    qs = E.build_queries(q["q_mol"], q["q_pol"], q["q_add"], np.diff(q["q_off"]), 4, q["q_ce"])
    meta = {"prec": q["q_prec"], "ce": q["q_ce"], "ne": q["q_ne"], "add": q["q_add"], "pol": q["q_pol"],
            "inst": q["q_inst"]}
    enc = {}
    for arm, m in models.items():
        z, lg = m.encode_store(q["q_mz"], q["q_int"].astype(np.float32), q["q_off"], meta, qs,
                               mol["M"].to_numpy(np.float64), cfg.enc_batch_queries)
        ze, le = m.aggregate(z, lg, qs, len(mol), cfg.enc_query_mode)
        enc[arm] = (ze, 1.0 / (1.0 + np.exp(-le.astype(np.float64))))
    log.info("encoded %d molecules with %d models in %.0f s", len(mol), len(models), time.time() - t0)
    rows = []
    pool_sizes = []
    ctx = mp_context()
    t_start = time.time()
    with ctx.Pool(max(1, args.n_jobs)) as workers:
        for b0 in range(0, len(order), args.batch_mol):
            if args.budget_min > 0 and time.time() - t_start > args.budget_min * 60:
                log.info("time budget reached after %d molecules", b0)
                break
            idx = order[b0:b0 + args.batch_mol]
            pools = [dbh.lookup(float(mol["M"].iloc[i]), tiers=("np", "pubchem")) for i in idx]
            need = []
            for pl in pools:
                nr = pl["np_row"].to_numpy(np.int64)
                need.append(pl.loc[nr < 0, "smiles"].astype(object).tolist())
            flat = [s for lst in need for s in lst]
            chunks = [flat[a:a + 2000] for a in range(0, len(flat), 2000)]
            fps_flat = (np.concatenate(list(workers.imap(_fp_chunk, chunks))) if chunks
                        else np.zeros((0, E.N_BYTES), np.uint8))
            pos_f = 0
            for i, pl in zip(idx, pools):
                nr = pl["np_row"].to_numpy(np.int64)
                fps = np.zeros((len(pl), E.N_BYTES), np.uint8)
                if (nr >= 0).any():
                    fps[nr >= 0] = np_fp[nr[nr >= 0]]
                k = int((nr < 0).sum())
                fps[nr < 0] = fps_flat[pos_f:pos_f + k]
                pos_f += k
                pos = pl["ik14"].astype(str).isin(keys[i]).to_numpy()
                row = {"molecule_id": mol["molecule_id"].iloc[i], "mkey": mol["mkey"].iloc[i],
                       "n_pool": int(len(pl)), "n_pos": int(pos.sum()),
                       "n_pubchem_only": int((nr < 0).sum())}
                for arm, m in models.items():
                    ze, pr = enc[arm]
                    sc = m.score(ze[i], pr[i], fps, cfg)
                    for key, v in sc.items():
                        row[f"{arm}:{key}"] = _rank(v, pos)
                rows.append(row)
                pool_sizes.append(len(pl))
            dbh.release()
            done = b0 + len(idx)
            if done % (args.batch_mol * 10) == 0 or done == len(order):
                el = time.time() - t_start
                log.info("%d / %d molecules, %.0f s (%.1f s per molecule), median pool %.0f", done,
                         len(order), el, el / done, float(np.median(pool_sizes)))
    ranks = pd.DataFrame(rows)
    out = Path(args.out)
    ranks.to_parquet(out.with_suffix(".ranks.parquet"), index=False)
    res = summarize(ranks)
    res.update({"n_available": n_eligible, "shard": [args.shard, args.n_shards],
                "seconds": round(time.time() - t0, 1), "query_mode": cfg.enc_query_mode})
    out.write_text(json.dumps(res, indent=1))
    log.info("C2-PC: %s", json.dumps(res["metrics"]))
    return res


if __name__ == "__main__":
    main()

"""Build and evaluate the class 3 virtual-analog (VA) tier (ROADMAP R13; casmi.va).

Stages::

    # 1. local: seeds, train keys and holdout keys for the Kaggle jobs (data/processed/va_inputs)
    python scripts/build_va.py inputs

    # 2. Kaggle CPU kernels (scripts/kaggle_job.py), one per seed shard (seed_id % n_shards):
    python scripts/build_va.py build --inputs $DS_CASMI_VA_INPUTS --pubchem-dir $DS_CASMI_DB_PUBCHEM \\
        --shard 0 --n-shards 4 --out $WORK/va_part --n-jobs 4 --budget-h 10.5

    # 3. merge the shard outputs into data/processed/db/va (Kaggle kernel or local)
    python scripts/build_va.py merge --parts P0 P1 P2 P3 --inputs DIR --out data/processed/db/va

    # 4. reach of VA (+ GEN) on the C3h / C3e holdout cells
    python scripts/build_va.py reach --cells C3h,C3e [--gen] [--panel fast|full] [--max-mol N]

    # local end-to-end dry run on a seed sample (all stages, small)
    python scripts/build_va.py dryrun --n-seeds 1500

    # push the 4 build kernels and the merge kernel (wraps scripts/kaggle_job.py)
    python scripts/build_va.py jobs --n-shards 4 [--merge]

Outputs: ``data/processed/db/va/mass_XXXX.parquet`` + ``seeds.parquet`` + ``_index.json``
(dataset ``casmi-db-va``), ``data/processed/validation/va_holdout_hits.parquet`` (seeds that
produce holdout structures, for the C3h / C3e masks), ``data/processed/validation/
va_reach_<tag>.parquet`` and ``va_reach_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from casmi import paths, va  # noqa: E402
from casmi.config import Config, load_config  # noqa: E402

log = logging.getLogger("build_va")
INPUTS = paths.PROCESSED / "va_inputs"
HITS = paths.PROCESSED / "validation" / "va_holdout_hits.parquet"
SCRIPTS = Path(__file__).resolve().parent


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _git() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------

def stage_inputs(args: argparse.Namespace, cfg: Config) -> dict:
    from casmi import db as cdb
    from casmi.splits import load_holdout_v2
    from casmi.validation import skeleton_key, validation_dir

    out = paths.ensure_dir(args.out or INPUTS)
    t0 = time.time()
    np_path = cdb.np_dir() / "np.parquet"
    np_df = pd.read_parquet(np_path, columns=["ik14", "smiles", "formula", "mono_mass",
                                              "coconut_np_likeness", "pubmed_count", "sid_count"])
    st = pd.read_parquet(paths.PROCESSED / "train" / "structures.parquet",
                         columns=["ik14", "smiles", "formula", "mono_mass", "libs", "n_spectra",
                                  "rdkit_ok", "np_score", "pmid_count"])
    seeds = va.seed_table(np_df, st, cfg)
    if args.max_seeds:
        seeds = seeds[seeds["seed_id"] < args.max_seeds].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(seeds, preserve_index=False), out / "seeds.parquet",
                   compression="zstd")
    pq.write_table(pa.table({"ik14": pa.array(sorted(st["ik14"].astype(str).unique()))}),
                   out / "train_ik14.parquet", compression="zstd")
    ho = load_holdout_v2()
    rows = [(str(a), str(b)) for a, b in zip(ho["ik14"], ho["mkey"])]
    tk_p = validation_dir() / "truth_keys.parquet"
    if tk_p.exists():
        tk = pd.read_parquet(tk_p)
        for mk, v in zip(tk["mkey"], tk["db_ik14"]):
            for k in (str(v).split(";") if isinstance(v, str) and v else []):
                rows.append((k, str(mk)))
    hik = pd.DataFrame(rows, columns=["ik14", "mkey"]).drop_duplicates("ik14", keep="first")
    hik = hik.sort_values("ik14", kind="mergesort").reset_index(drop=True)
    hik.to_parquet(out / "holdout_ik14.parquet", index=False)
    smi = dict(zip(st["ik14"].astype(str), st["smiles"]))
    prim = ho[ho["primary"].astype(bool)] if "primary" in ho.columns else ho
    prim = prim.drop_duplicates("mkey")
    hm = pd.DataFrame({"mkey": prim["mkey"].astype(str).to_numpy(), "ik14": prim["ik14"].astype(str).to_numpy(),
                       "formula": prim["formula"].astype(str).to_numpy()})
    hm["smiles"] = [smi.get(k) for k in hm["ik14"]]
    hm["skeleton"] = [skeleton_key(s) if isinstance(s, str) else None for s in hm["smiles"]]
    hm = hm.sort_values("mkey", kind="mergesort").reset_index(drop=True)
    hm.to_parquet(out / "holdout_mols.parquet", index=False)
    info = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": _git(),
            "np_parquet_sha256": _sha256(np_path), "n_seeds": int(len(seeds)),
            "n_seeds_np": int((seeds["source"] == "np").sum()),
            "n_seeds_train": int((seeds["source"] == "train").sum()),
            "n_seeds_with_spectra": int((seeds["n_spectra"] > 0).sum()),
            "n_train_ik14": int(st["ik14"].nunique()), "n_holdout_ik14": int(len(hik)),
            "n_holdout_mols": int(len(hm)), "n_holdout_skeletons": int(hm["skeleton"].notna().sum()),
            "exclude_libs": list(cfg.va_exclude_libs), "seconds": round(time.time() - t0, 1)}
    (out / "inputs_info.json").write_text(json.dumps(info, indent=1) + "\n")
    log.info("inputs: %s", info)
    return info


# --------------------------------------------------------------------------------------
# build (one seed shard)
# --------------------------------------------------------------------------------------

def known_key_files(inputs: Path, pubchem_dir: Path | None, key_dir: Path,
                    seeds: pd.DataFrame) -> dict[str, Path]:
    key_dir = paths.ensure_dir(key_dir)
    files = {}
    t0 = time.time()
    if pubchem_dir is not None:
        shards = sorted(Path(pubchem_dir).rglob("mass_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no PubChem shards under {pubchem_dir}")
        k = va.keys_from_parquet(shards)
        np.save(key_dir / "pubchem.npy", k)
        files["pubchem"] = key_dir / "pubchem.npy"
        log.info("PubChem keys: %d from %d shards in %.0f s", len(k), len(shards), time.time() - t0)
        del k
    npk = va.sorted_keys(va.ik14_keys(pa.array(seeds.loc[seeds["source"] == "np", "ik14"].astype(str).tolist())))
    np.save(key_dir / "np.npy", npk)
    files["np"] = key_dir / "np.npy"
    trk = va.keys_from_parquet([inputs / "train_ik14.parquet"])
    np.save(key_dir / "train.npy", trk)
    files["train"] = key_dir / "train.npy"
    log.info("known keys: np %d, train %d", len(npk), len(trk))
    return files


def stage_build(args: argparse.Namespace, cfg: Config) -> dict:
    inputs = Path(args.inputs or INPUTS)
    t0 = time.time()
    seeds = pd.read_parquet(inputs / "seeds.parquet", columns=["seed_id", "ik14", "smiles", "source", "pri"])
    key_dir = Path(args.key_dir) if args.key_dir else Path(os.environ.get("TMPDIR", "/tmp")) / "va_keys"
    kf = known_key_files(inputs, Path(args.pubchem_dir) if args.pubchem_dir else None, key_dir, seeds)
    hold = va.holdout_lookup(pd.read_parquet(inputs / "holdout_ik14.parquet"),
                             pd.read_parquet(inputs / "holdout_mols.parquet"))
    sel = seeds[(seeds["seed_id"].to_numpy() % args.n_shards) == args.shard]
    if args.sample:
        rng = np.random.default_rng(cfg.seed + args.shard)
        sel = sel.iloc[np.sort(rng.choice(len(sel), min(args.sample, len(sel)), replace=False))]
    if args.max_seeds:
        sel = sel.iloc[:args.max_seeds]
    sel = sel.reset_index(drop=True)
    t_keys = time.time() - t0
    log.info("shard %d/%d: %d seeds (keys %.0f s)", args.shard, args.n_shards, len(sel), t_keys)
    info = va.build_products(sel, kf, hold, args.out, cfg, n_jobs=va.worker_count(args.n_jobs),
                             chunk=args.chunk, budget_s=args.budget_h * 3600 if args.budget_h else None)
    info.update({"shard": args.shard, "n_shards": args.n_shards, "keys_seconds": round(t_keys, 1),
                 "total_seconds": round(time.time() - t0, 1), "git": _git(),
                 "n_known": {k: int(np.load(v, mmap_mode="r").shape[0]) for k, v in kf.items()}})
    out = Path(args.out)
    (out / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    log.info("shard %d done: %d seeds, %d products, %d unique novel, %d holdout hits, %.0f s",
             args.shard, info["n_seeds_done"], info["n_products"], info["n_unique_novel"],
             info["n_hits"], info["total_seconds"])
    if not args.keep_keys:
        shutil.rmtree(key_dir, ignore_errors=True)
    return info


# --------------------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------------------

def _part_dirs(specs: list[str]) -> list[Path]:
    out = []
    for s in specs:
        p = Path(s)
        hits = sorted({q.parent for q in p.rglob("build_info.json")}) if p.is_dir() else []
        if not hits:
            raise FileNotFoundError(f"no build_info.json under {p}")
        out.extend(hits)
    return sorted(dict.fromkeys(out))


def stage_merge(args: argparse.Namespace, cfg: Config) -> dict:
    from casmi import generate as gen

    t0 = time.time()
    parts = _part_dirs(args.parts)
    out = paths.ensure_dir(args.out or va.va_dir())
    for old in out.glob("mass_*.parquet"):
        old.unlink()
    infos = [json.loads((p / "build_info.json").read_text()) for p in parts]
    filt = None
    if args.min_n_seeds > 1 or args.drop_rules:
        drop = {gen.RULE_IDS[r] for r in args.drop_rules}

        def filt(t: pa.Table) -> np.ndarray:  # noqa: F811
            ok = t.column("n_seeds").to_numpy() >= args.min_n_seeds
            if drop:
                ok &= ~np.isin(t.column("rule").to_numpy(), list(drop))
            return ok
    res = va.merge_parts(parts, out, row_filter=filt)
    inputs = Path(args.inputs or INPUTS)
    shutil.copyfile(inputs / "seeds.parquet", out / "seeds.parquet")
    hit_tabs = [pq.read_table(p / "holdout_hits.parquet") for p in parts if (p / "holdout_hits.parquet").exists()]
    hits_out = Path(args.hits_out) if args.hits_out else (out.parent / "va_holdout_hits.parquet")
    if hit_tabs:
        ht = pa.concat_tables(hit_tabs).to_pandas()
        ht = ht.sort_values(["mkey", "ik14", "seed_id", "rule"], kind="mergesort").reset_index(drop=True)
        paths.ensure_dir(hits_out.parent)
        ht.to_parquet(hits_out, index=False)
    names = [r.id for r in gen.RULES]
    stats = {}
    for k in ("n_raw", "n_products", "n_np", "n_pubchem", "n_train", "n_novel"):
        stats[k] = [int(sum(i["rule_stats"][k][j] for i in infos)) for j in range(len(names))]
    n_rows = res["n_rows"]
    ns = []
    for p in sorted(out.glob("mass_*.parquet")):
        ns.append(pq.read_table(p, columns=["n_seeds"]).column("n_seeds").to_numpy())
    ns = np.concatenate(ns) if ns else np.zeros(0, np.int32)
    inp_info = json.loads((inputs / "inputs_info.json").read_text()) if (inputs / "inputs_info.json").exists() else {}
    shards = sorted(out.glob("mass_*.parquet"))
    index = {
        "description": "class 3 virtual-analog tier: novel one-edit products of NP-tier and NP-like "
                       "train structures (casmi.va, ROADMAP R13)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": _git(),
        "schema": list(va.VA_COLUMNS), "bin_da": va.SHARD_DA, "sort": ["mono_mass", "ik14"],
        "mass_min": cfg.va_mass_min, "mass_max": cfg.va_mass_max, "n_rows": int(n_rows),
        "n_rows_before_merge": res["n_rows_before_merge"], "total_bytes": res["bytes"],
        "n_shards": len(shards), "shards": {p.name: res["rows_per_bucket"].get(int(p.stem.split("_")[1]), 0)
                                            for p in shards},
        "filter": {"min_n_seeds": args.min_n_seeds, "drop_rules": list(args.drop_rules)},
        "n_seeds_hist": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(ns, 10), return_counts=True))},
        "rules": [{"code": i, "id": r.id, "family": r.family, "delta": r.delta_formula} for i, r in enumerate(gen.RULES)],
        "rule_stats": stats,
        "builds": [{k: i.get(k) for k in ("shard", "n_shards", "n_seeds_input", "n_seeds_done", "n_seeds_failed",
                                          "n_seeds_cut", "n_products", "n_unique_novel", "n_hits", "wall_seconds",
                                          "worker_seconds", "total_seconds", "budget_stopped", "n_jobs", "git")}
                   for i in infos],
        "inputs": inp_info, "merge_seconds": round(time.time() - t0, 1),
    }
    (out / "_index.json").write_text(json.dumps(index, indent=1) + "\n")
    log.info("merge: %d rows (%d before), %.2f GB in %d shards, %.0f s", n_rows, res["n_rows_before_merge"],
             res["bytes"] / 1e9, len(shards), time.time() - t0)
    return index


# --------------------------------------------------------------------------------------
# reach
# --------------------------------------------------------------------------------------

def _analog_seeds(files: list[Path], keep: set[str]) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for f in files:
        if not f.exists():
            log.warning("analog hits %s missing", f)
            continue
        h = pd.read_parquet(f, columns=["molecule_id", "hit_rank", "ik14", "smiles"])
        h = h[h["molecule_id"].astype(str).isin(keep)]
        h = h.sort_values(["molecule_id", "hit_rank"], kind="mergesort")
        for mid, g in h.groupby("molecule_id", sort=True):
            out[str(mid)] = list(zip(g["ik14"].astype(str), g["smiles"].astype(str)))
    return out


def stage_reach(args: argparse.Namespace, cfg: Config) -> dict:
    from casmi import validation as V
    from casmi.splits import load_holdout_v2

    t0 = time.time()
    val_dir = V.validation_dir()
    hits_p = Path(args.hits) if args.hits else HITS
    seeds_p = Path(args.seeds) if args.seeds else (Path(args.va_dir or va.va_dir()) / "seeds.parquet")
    hits = pd.read_parquet(hits_p)
    seeds = pd.read_parquet(seeds_p, columns=["seed_id", "ik14", "pri", "source"])
    ho2 = load_holdout_v2()
    attrs = V.v2_attributes(ho2)
    tier = va.VATier(args.va_dir) if args.va_dir or va.va_dir().is_dir() else None
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    tabs, summary = [], {"cells": {}, "hits_file": str(hits_p), "n_hits": int(len(hits)),
                         "gen": bool(args.gen), "panel": args.panel}
    for s in sets:
        _, t = V.load_queries(s, val_dir)
        if args.max_mol:
            t = t.sort_values("molecule_id", kind="mergesort").iloc[:args.max_mol]
        masks = {c: V.cell_mask(t, attrs, c, args.panel) for c in cells}
        used = t[np.logical_or.reduce(list(masks.values()))].reset_index(drop=True)
        if not len(used):
            continue
        keys = V.truth_keys(used, cfg, n_jobs=1, cache_path=val_dir / "truth_keys.parquet")
        near = V.c3h_neighbours(used, cfg, n_jobs=1, cache_path=val_dir / "c3h_neighbours.parquet") \
            if "C3h" in cells else None
        gen_seeds = None
        if args.gen:
            files = [Path(f) for f in args.analog_hits] if args.analog_hits else \
                [paths.PROCESSED / "analogs" / f"hits_v1_{V.set_slug(s)}_C2.parquet"]
            gen_seeds = _analog_seeds(files, set(used["molecule_id"].astype(str)))
        win = None
        if tier is not None and tier.available():
            win = dict(zip(used["molecule_id"].astype(str),
                           va.window_counts(tier, used["mono_mass"].to_numpy(np.float64), cfg)))
        for c in cells:
            tc = t[masks[c]].reset_index(drop=True)
            if not len(tc):
                continue
            excl = V.exclusions_v2(tc, c, keys, near if c == "C3h" else None)
            ex = {m: e.db_ik14 | e.group for m, e in excl.items()}
            tk = {m: e.truth for m, e in excl.items()}
            gs = None
            if gen_seeds is not None:
                gs = {m: gen_seeds.get(m, []) for m in tc["molecule_id"].astype(str)}
            tab = va.reach_table(tc, ex, hits, seeds, gen_seeds=gs, truth_ik14=tk, cfg=cfg,
                                 n_jobs=va.worker_count(args.n_jobs) if args.n_jobs != 1 else 1)
            tab.insert(0, "cell", c)
            tab.insert(1, "set", s)
            tab["mkey"] = tc["mkey"].astype(str).to_numpy()
            tab["n_excluded"] = [len(ex[m]) for m in tab["molecule_id"]]
            if gen_seeds is not None:
                tab["n_analog_hits"] = [len(gen_seeds.get(m, [])) for m in tab["molecule_id"]]
            if win is not None:
                tab["va_window"] = [int(win.get(m, 0)) for m in tab["molecule_id"]]
            tabs.append(tab)
            log.info("reach %s %s: %s", s, c, va.summarize_reach(tab))
    tab = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    for c in cells:
        sub = tab[tab["cell"] == c] if len(tab) else tab
        summary["cells"][c] = va.summarize_reach(sub)
        summary["cells"][c]["by_set"] = {s: va.summarize_reach(sub[sub["set"] == s]) for s in sets} if len(sub) else {}
    summary["seconds"] = round(time.time() - t0, 1)
    if tier is not None and tier.available():
        summary["tier_rows"] = tier.index().get("n_rows")
    tag = args.tag
    out_p = val_dir / f"va_reach_{tag}.parquet"
    if len(tab):
        tab.to_parquet(out_p, index=False)
    (val_dir / f"va_reach_{tag}.json").write_text(json.dumps(summary, indent=1) + "\n")
    log.info("reach summary (%s): %s", tag, json.dumps(summary["cells"]))
    return summary


# --------------------------------------------------------------------------------------
# dry run, jobs
# --------------------------------------------------------------------------------------

def stage_dryrun(args: argparse.Namespace, cfg: Config) -> dict:
    """Small local end-to-end run: inputs (if missing), one sampled shard with the real
    PubChem keys, merge into a scratch tier, a lookup through db, and VA reach."""
    from casmi import db as cdb

    scratch = Path(args.scratch or (paths.PROCESSED / "va_dryrun"))
    if not (INPUTS / "seeds.parquet").exists():
        stage_inputs(argparse.Namespace(out=None, max_seeds=0), cfg)
    b = argparse.Namespace(inputs=str(INPUTS), pubchem_dir=str(cdb.pubchem_dir()) if not args.no_pubchem else None,
                           shard=0, n_shards=1, out=str(scratch / "part0"), n_jobs=args.n_jobs, budget_h=None,
                           chunk=50, sample=args.n_seeds, max_seeds=0, key_dir=str(scratch / "keys"),
                           keep_keys=False)
    info = stage_build(b, cfg)
    m = argparse.Namespace(parts=[str(scratch / "part0")], inputs=str(INPUTS), out=str(scratch / "va"),
                           hits_out=str(scratch / "va_holdout_hits.parquet"), min_n_seeds=1, drop_rules=[])
    idx = stage_merge(m, cfg)
    d = cdb.CandidateDB(cfg=cfg)
    d.va_path = scratch / "va"
    t = pq.read_table(sorted((scratch / "va").glob("mass_*.parquet"))[0]).to_pandas()
    look = d.lookup(float(t["mono_mass"].iloc[0]), tiers=("np", "va"))
    log.info("dry-run lookup at %.4f: %d rows, tiers %s", t["mono_mass"].iloc[0], len(look),
             look["tier"].value_counts().to_dict())
    return {"build": {k: info[k] for k in ("n_seeds_done", "n_products", "n_unique_novel", "n_hits",
                                           "wall_seconds", "worker_seconds")},
            "merge_rows": idx["n_rows"], "bytes": idx["total_bytes"]}


def stage_jobs(args: argparse.Namespace, cfg: Config) -> int:
    import kaggle_job  # noqa: F401  (scripts/ is on sys.path below)

    code = args.code
    base = ["--code", code, "--dataset", "casmi-va-inputs", "--timeout-hours", str(args.timeout_hours)]
    rc = 0
    if not args.merge:
        for i in range(args.n_shards):
            cmd = ('python "$CODE/scripts/build_va.py" build --inputs "$DS_CASMI_VA_INPUTS" '
                   f'--pubchem-dir "$DS_CASMI_DB_PUBCHEM" --shard {i} --n-shards {args.n_shards} '
                   f'--out "$WORK/va_part" --n-jobs 4 --budget-h {args.budget_h} --chunk {args.chunk}')
            rc |= kaggle_job.main(["push", "--slug", f"{args.slug_prefix}-{i}", *base,
                                   "--dataset", "casmi-db-pubchem", "--cmd", cmd])
        return rc
    srcs = []
    parts = []
    for i in range(args.n_shards):
        s = f"{args.slug_prefix}-{i}"
        srcs += ["--kernel-source", s]
        parts.append(f'"$KS_{s.upper().replace("-", "_")}"')
    cmd = ('python "$CODE/scripts/build_va.py" merge --parts ' + " ".join(parts) +
           ' --inputs "$DS_CASMI_VA_INPUTS" --out "$WORK/va" --hits-out "$WORK/va_holdout_hits.parquet"'
           + (f" --min-n-seeds {args.min_n_seeds}" if args.min_n_seeds > 1 else ""))
    return kaggle_job.main(["push", "--slug", f"{args.slug_prefix}-merge", *base, *srcs, "--cmd", cmd])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("inputs", "build", "merge", "reach", "dryrun", "jobs"))
    p.add_argument("--config", default=None, help="configs/<name>.yaml (default: Config defaults)")
    p.add_argument("--out", default=None)
    p.add_argument("--inputs", default=None, help="va_inputs directory (default data/processed/va_inputs)")
    p.add_argument("--pubchem-dir", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=4)
    p.add_argument("--n-jobs", type=int, default=1, help="processes (0 = all CPUs)")
    p.add_argument("--chunk", type=int, default=200, help="seeds per worker task")
    p.add_argument("--budget-h", type=float, default=None, help="stop scheduling seeds after this many hours")
    p.add_argument("--sample", type=int, default=0, help="build: random sample of the shard's seeds")
    p.add_argument("--max-seeds", type=int, default=0)
    p.add_argument("--key-dir", default=None)
    p.add_argument("--keep-keys", action="store_true")
    p.add_argument("--parts", nargs="+", default=[])
    p.add_argument("--hits-out", default=None)
    p.add_argument("--min-n-seeds", type=int, default=1, help="merge: plausibility filter on n_seeds")
    p.add_argument("--drop-rules", nargs="*", default=[], help="merge: rule ids to leave out")
    p.add_argument("--cells", default="C3h,C3e")
    p.add_argument("--sets", default="NP-TIMS,NP-OTHER")
    p.add_argument("--panel", default="full", choices=("fast", "full"))
    p.add_argument("--max-mol", type=int, default=0)
    p.add_argument("--gen", action="store_true", help="reach: also run GEN (two edits) from analog seeds")
    p.add_argument("--analog-hits", nargs="*", default=[])
    p.add_argument("--hits", default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--va-dir", default=None)
    p.add_argument("--tag", default="v1")
    p.add_argument("--n-seeds", type=int, default=1500, help="dryrun: sampled seeds")
    p.add_argument("--no-pubchem", action="store_true", help="dryrun: skip the PubChem key set")
    p.add_argument("--scratch", default=None)
    p.add_argument("--code", default="casmi-job-code", help="jobs: code dataset slug")
    p.add_argument("--slug-prefix", default="casmi-va-build")
    p.add_argument("--timeout-hours", type=float, default=12.0)
    p.add_argument("--merge", action="store_true", help="jobs: push the merge kernel instead")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config) if args.config else Config()
    if args.stage == "inputs":
        stage_inputs(args, cfg)
    elif args.stage == "build":
        if args.budget_h is None:
            args.budget_h = 0
        stage_build(args, cfg)
    elif args.stage == "merge":
        stage_merge(args, cfg)
    elif args.stage == "reach":
        stage_reach(args, cfg)
    elif args.stage == "dryrun":
        print(json.dumps(stage_dryrun(args, cfg), indent=1))
    elif args.stage == "jobs":
        sys.path.insert(0, str(SCRIPTS))
        if args.budget_h is None:
            args.budget_h = 10.5
        return stage_jobs(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

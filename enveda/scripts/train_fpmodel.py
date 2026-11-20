"""Train the fingerprint model (WP9, PLAN.md 4.4) and export it for numpy inference.

Thin wrapper over ``casmi.fpmodel``. Outputs (``--out-dir``, default ``models/``):
  fp_<tag>.npz                weights + spec for ``casmi.fpmodel.FpModel.load``
  fp_<tag>_ckpt.pt            torch checkpoint (every ``fpm_checkpoint_minutes``; --resume)
  fp_<base>_calibration.npz   per-bit calibration fitted on holdout molecules
                              (--exclude-holdout only; base = tag without "_holdout")
  logs/fp_<tag>_curve.csv     loss curve (plus the usual logs/train_fpmodel_<ts>.log)

Examples:
  python scripts/train_fpmodel.py --exclude-holdout --hours 4            # v1_holdout
  python scripts/train_fpmodel.py --all --hours 4                        # v1_full
  python scripts/train_fpmodel.py --synthetic 20000 --hours 0.02 --tag smoke --out-dir /tmp/x
  python scripts/train_fpmodel.py --benchmark --bench-seconds 45         # throughput
"""

from __future__ import annotations

import argparse
import json
import sys

from casmi import fpmodel, paths
from casmi.config import load_config
from casmi.log import get_logger, log_run

BENCH_VARIANTS = {
    "base": {},
    "base_unfused": {"fpm_fused_adamw": False},
    "bf16": {"fpm_bf16": True},
    "bf16_b1024": {"fpm_bf16": True, "fpm_batch_size": 1024},
    "sparse": {"fpm_sparse_embedding": True},
    "bf16_sparse": {"fpm_bf16": True, "fpm_sparse_embedding": True},
    "w512": {"fpm_hidden": 512},
    "w512_bf16": {"fpm_hidden": 512, "fpm_bf16": True},
    "bin02": {"fpm_mz_bin_da": 0.2},
    "bin02_bf16": {"fpm_mz_bin_da": 0.2, "fpm_bf16": True},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="configs/<name>.yaml (fpm_* fields)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--exclude-holdout", dest="exclude_holdout", action="store_true",
                   default=True, help="train on train minus holdout molecules (default)")
    g.add_argument("--all", dest="exclude_holdout", action="store_false",
                   help="train on all of train (final refit)")
    ap.add_argument("--hours", type=float, help="wall budget of the training loop "
                    "(default cfg.fpm_hours)")
    ap.add_argument("--steps", type=int, help="exact number of steps (reproduces a run)")
    ap.add_argument("--epochs", type=float, help="number of epochs instead of --hours")
    ap.add_argument("--tag", help="output tag (default v1_holdout / v1_full)")
    ap.add_argument("--max-spectra", type=int, help="random training subset (smoke runs)")
    ap.add_argument("--source", help="train-schema parquet file or dir, or a WP5 library dir "
                    "(default: data/processed/train/spectra if present, else "
                    "data/processed/library, else data/raw/train.parquet)")
    ap.add_argument("--synthetic", type=int, metavar="N",
                    help="train on N synthetic spectra instead (no train data needed)")
    ap.add_argument("--holdout-file", default=str(paths.SPLITS / "holdout_v1.csv"))
    ap.add_argument("--structures", default=str(paths.PROCESSED / "train" / "structures.parquet"),
                    help="ik14 -> mkey table for the holdout exclusion by mkey")
    ap.add_argument("--threads", type=int, help="torch threads (default cfg.fpm_threads)")
    ap.add_argument("--n-jobs", type=int, default=2, help="processes for target fingerprints")
    ap.add_argument("--out-dir", default=str(paths.MODELS))
    ap.add_argument("--resume", action="store_true", help="continue from fp_<tag>_ckpt.pt")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--float16", action="store_true", help="export weights as float16")
    ap.add_argument("--benchmark", action="store_true",
                    help="measure training throughput on synthetic batches and exit")
    ap.add_argument("--bench-seconds", type=float, default=45.0)
    ap.add_argument("--bench-peaks", type=int, default=100,
                    help="cleaned peaks per synthetic spectrum (public test mean: about 32)")
    ap.add_argument("--bench-variants", default="base,base_unfused,bf16,bf16_b1024,w512,bin02",
                    help=f"comma-separated subset of {sorted(BENCH_VARIANTS)}")
    return ap.parse_args(argv)


def run_benchmark(args: argparse.Namespace, cfg, lg) -> int:
    results = []
    for name in args.bench_variants.split(","):
        c = cfg.replace(**BENCH_VARIANTS[name])
        r = fpmodel.benchmark_throughput(c, seconds=args.bench_seconds, threads=args.threads,
                                         peaks=args.bench_peaks)
        r["variant"] = name
        results.append(r)
        lg.info("benchmark %-12s %7.0f spectra/s (model only %7.0f, prep %4.1f%%), "
                "epoch of 2.3M: %5.1f min, %.1fM params, %.0f peaks, %.0f entries/spectrum",
                name, r["spectra_per_s"], r["spectra_per_s_model_only"], 100 * r["prep_share"],
                r["epoch_minutes_2_3m"], r["params_m"], r["peaks_per_spectrum"],
                r["entries_per_spectrum"])
    out = paths.ensure_dir(paths.LOGS) / f"fp_benchmark_{args.bench_peaks}peaks.json"
    out.write_text(json.dumps(results, indent=1))
    lg.info("benchmark results written to %s", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lg = get_logger("train_fpmodel")
    log_run(lg, args)
    cfg = load_config(args.config)
    if args.benchmark:
        return run_benchmark(args, cfg, lg)
    tag = args.tag or ("v1_holdout" if args.exclude_holdout else "v1_full")
    out_dir = paths.ensure_dir(args.out_dir)
    # smoke runs: load a random 2x superset only (filters and holdout remove some rows)
    max_rows = 2 * args.max_spectra if args.max_spectra else None
    data = fpmodel.load_training_source(args.source, cfg, n_jobs=args.n_jobs,
                                        synthetic=args.synthetic, max_rows=max_rows)
    # synthetic spectra never match the real holdout (splits/holdout_v1.csv exists since WP7
    # ran on train): use the synthetic 5 percent holdout of split_training_data instead
    train, val, calib, info = fpmodel.split_training_data(
        data, cfg, exclude_holdout=args.exclude_holdout,
        holdout=None if args.synthetic else args.holdout_file,
        structures=None if args.synthetic else args.structures, max_spectra=args.max_spectra)
    del data
    npz = out_dir / f"fp_{tag}.npz"
    res = fpmodel.train(train, cfg, hours=args.hours, steps=args.steps, epochs=args.epochs,
                        val=val, out_npz=npz, checkpoint=out_dir / f"fp_{tag}_ckpt.pt",
                        resume=args.resume, curve_csv=paths.LOGS / f"fp_{tag}_curve.csv",
                        threads=args.threads, float16_export=args.float16,
                        extra_info={"tag": tag, "split": info,
                                    "exclude_holdout": args.exclude_holdout})
    lg.info("trained %d steps (%.2f epochs) in %.0f s, %.0f spectra/s, final loss %.4f",
            res.steps, res.epochs, res.seconds, res.spectra_per_s, res.final_loss)
    if args.exclude_holdout and calib is not None and not args.no_calibration:
        model = fpmodel.FpModel.load(npz)
        cal, metrics = fpmodel.calibrate(model, calib, cfg)
        base = tag[:-len("_holdout")] if tag.endswith("_holdout") else tag
        cal_path = cal.save(out_dir / f"fp_{base}_calibration.npz",
                            {"model": npz.name, **metrics})
        lg.info("calibration written to %s", cal_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

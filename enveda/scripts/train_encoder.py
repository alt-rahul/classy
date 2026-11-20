"""Train one arm of the contrastive transformer encoder (RESEARCH.md roadmap item 1).

    python scripts/train_encoder.py --arm focal_con --hours 4.5 --device cuda:0 \\
        --bundle data/processed/encoder/bundle_v1 --np-fp data/processed/db/np/np_fp.npy \\
        --out models/encoder/focal_con

Arms (``casmi.encoder.ARMS``): ``bce`` (fingerprint BCE only), ``focal`` (focal only),
``focal_con`` (focal + candidate-contrastive with a molecule tower), ``con``
(contrastive only). Config tunables are ``Config.enc_*`` (``--config`` YAML, or
``--set key=value`` repeatable, values parsed as YAML). Outputs in ``--out``: see
``casmi.encoder.train``. Rerunning with the same ``--out`` resumes from
``checkpoint.pt`` unless ``--no-resume``. ``--keep-checkpoint`` keeps the optimiser
checkpoint after the final export (deleted by default to keep outputs small).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "casmi").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from casmi import paths  # noqa: E402
from casmi.config import load_config  # noqa: E402
from casmi.log import get_logger, log_run  # noqa: E402


def parse_sets(items: list[str]) -> dict:
    import yaml

    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        out[k.strip()] = yaml.safe_load(v)
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="focal_con")
    p.add_argument("--bundle", default=None, help="default PROCESSED/encoder/bundle_v1")
    p.add_argument("--np-fp", default=None, help="default PROCESSED/db/np/np_fp.npy")
    p.add_argument("--out", default=None, help="default MODELS/encoder/<arm>")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--hours", type=float, default=None, help="default cfg.enc_hours")
    p.add_argument("--device", default=None, help="default cuda if available else cpu")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-max-mol", type=int, default=0, help="cap molecules per eval pack (0 = all)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--keep-checkpoint", action="store_true")
    p.add_argument("--eval-only", default=None, metavar="MODEL",
                   help="evaluate an exported encoder on the bundle's packs and exit")
    p.add_argument("--eval-out", default=None, help="--eval-only: JSON output path")
    args = p.parse_args(argv)
    log = get_logger(f"train_encoder_{args.arm}")
    log_run(log, args)
    import torch

    from casmi import encoder as E

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    if args.set:
        cfg = cfg.replace(**parse_sets(args.set))
    bundle = Path(args.bundle) if args.bundle else paths.PROCESSED / "encoder" / "bundle_v1"
    np_fp = Path(args.np_fp) if args.np_fp else paths.PROCESSED / "db" / "np" / "np_fp.npy"
    out = Path(args.out) if args.out else paths.MODELS / "encoder" / args.arm
    if args.eval_only:
        import numpy as np

        model = E.EncoderModel.load(args.eval_only, args.device or "cpu")
        packs = E.Bundle.load(bundle).eval_packs()
        if args.eval_max_mol > 0:
            packs = {k: v.subset(args.eval_max_mol, cfg.seed) for k, v in packs.items()}
        res = E.evaluate_packs(model, packs, np.load(np_fp, mmap_mode="r"), cfg)
        flat = E._flat_metrics(res)
        log.info("eval %s: %s", args.eval_only, json.dumps(flat))
        if args.eval_out:
            Path(args.eval_out).write_text(json.dumps({"model": str(args.eval_only), "flat": flat,
                                                       "detail": res}, indent=1, default=float))
        return {"flat": flat, "detail": res}
    if args.device and args.device.startswith("cuda"):
        log.info("cuda: %s, devices %d, %s", torch.version.cuda, torch.cuda.device_count(),
                 torch.cuda.get_device_name(torch.device(args.device)))
    summary = E.train(bundle, np_fp, out, cfg, arm=args.arm, hours=args.hours, device=args.device,
                      max_steps=args.max_steps, seed=args.seed, eval_max_mol=args.eval_max_mol,
                      resume=not args.no_resume)
    if not args.keep_checkpoint:
        ck = out / "checkpoint.pt"
        if ck.exists():
            ck.unlink()
    log.info("final: %s", json.dumps(summary.get("final", {})))
    return summary


if __name__ == "__main__":
    main()

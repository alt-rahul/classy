# Enveda - CASMI 2026: MS/MS spectrum to structure

Predict up to 25 ranked SMILES per `molecule_id` from Bruker timsTOF LC-MS/MS spectra.
Scored by MRR@25 on InChIKey14 after RDKit (2026.03.3) tautomer canonicalisation.

## Layout

```
enveda/
  data/
    raw/         competition files: train.parquet, test.parquet, sample_submission.csv (git-ignored)
    processed/   derived tables, indexes, feature caches (git-ignored)
    external/    PubChem / COCONUT / other public resources (git-ignored)
  src/           importable pipeline code (package: casmi)
  scripts/       CLI entry points (build indexes, train, predict, evaluate)
  notebooks/     exploration and the Kaggle submission notebook
  models/        trained weights and fitted artefacts (git-ignored)
  submissions/   every generated submission CSV, named <date>_<tag>.csv
  docs/          plan, experiment log, findings
  logs/          run logs (git-ignored)
```

## Data

- `data/raw/test.parquet` - 1,213 spectra / 400 molecules (public copy is drawn from train; replaced by the hidden set on re-run).
- `data/raw/sample_submission.csv` - format reference.
- `data/raw/train.parquet` - ~2.5M spectra / ~275k structures (arriving later, ~3 GB).

See `docs/PLAN.md` for the solution design.

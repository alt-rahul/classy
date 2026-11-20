# Engine sources in this dataset (third-party, Apache-2.0)

- `elfopt.cpp`: value-DAG LNS optimizer by goodpjw2008, from the public Kaggle dataset
  goodpjw2008/santa-2026-dag-lns-programs (Apache 2.0), unchanged. Its design follows the `lns.cpp` of
  prvsiyan's notebook "Santa is not happy" (Apache-2.0), as credited by its author.
- `*.cpp`, `*.inc` (except elfopt.cpp): Wang Jakie's graph-search strategies from the Kaggle notebook
  "Santa 2026 | Arithmetic Recipe Pool and MILP" (Apache 2.0), as extracted in the same goodpjw2008 dataset;
  `anneal_seeded.cpp` / `incremental_seeded.cpp` carry goodpjw2008's one-line seeding edit. See NOTICE_wj.md.
- `programs.csv`: per-puzzle shortest valid programs of our store, mostly public Apache-2.0 / CC0 programs
  (see THIRD_PARTY.md of the project).
- `runner.py`: our driver.
- `anneal_div.cpp`, `incremental_div.cpp` (from kaggle/engines/extra): Wang Jakie's `anneal.cpp` / `incremental.cpp`
  with `improve()` modified by us: donor tapes seed the crossover pool and stay in it, and the archive evicts its
  least distinctive member. The idea follows royalacecat's public "diverse" graph search; the code is ours.
- `elfopt_v.cpp` (from kaggle/engines/extra): goodpjw2008's `elfopt.cpp` plus our `--emit-plateau` option, which also
  prints the last equal-length program of the plateau walk; the runner keeps these as crossover donors.

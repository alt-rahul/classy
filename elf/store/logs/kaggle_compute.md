# Kaggle compute fleet log

Private Kaggle script kernels (user ograhulrajkumar, free CPU sessions, 4 vCPUs each) run program-improvement
engines on shards of our store; a local loop harvests their outputs and merges them through `offer` + `save_store`
(every program re-validated locally), then relaunches the slot with a fresh shard. The submit daemon picks gains up.

## Operate / resume
- Start (or resume after a container restart): `cd /home/user/classy/elf && setsid nohup python3 kaggle/fleet.py loop >> store/logs/kaggle_fleet.out 2>&1 &`
  (one copy only: `pgrep -f "fleet.py loop"`). All state is on disk (`store/kaggle_state.json`); kernels keep
  running on Kaggle while the loop is down and are harvested on the next poll.
- Status: `python3 kaggle/fleet.py status` (slots, merged gain, gain per CPU-hour per engine).
- One cycle by hand: `python3 kaggle/fleet.py poll`. Re-upload the dataset now: `python3 kaggle/fleet.py dataset --force`.
- Stop launching new kernels: `touch kaggle/STOP_LAUNCH` (loop keeps harvesting). Stop the loop: `touch kaggle/STOP`.
- Config (read every cycle): `kaggle/fleet_config.json` - `max_kernels` (concurrent slots), `budget_s` (engine time
  per session, Kaggle caps sessions at 12 h), `puzzles_per_kernel`, `min_len`, `len_power` (priority ~ len^p),
  `engines` (list, see kaggle/ENGINES.md). Changing engine sources or runner.py triggers a dataset re-version.
- Files: kaggle/runner.py (runs inside the kernel), kaggle/fleet.py (local controller), kaggle/kapi.py (REST via curl,
  credential injected by the proxy), kaggle/engines/ (public Apache-2.0 engine sources), kaggle/build/ (generated).
  Raw kernel outputs: store/kaggle_out/<slot>_v<version>/ (improved.csv, stats.json, kernel_log.json).

## Events
- 2026-11-20 19:01:01 UTC dataset created ograhulrajkumar/elf-compute with store total 133966 (23 files)
- 2026-11-20 19:01:31 UTC launched elf-c01 v1: 8 puzzles (len 273..277, 2198 instr), budget 0.2 h, engines elfopt,wj_anneal,wj_incremental,wj_compact_rotate,wj_anneal_seeded
- 2026-11-20 19:03:30 UTC launched elf-c01 v2: 8 puzzles (len 273..277, 2198 instr), budget 0.2 h, engines elfopt,wj_anneal,wj_incremental,wj_compact_rotate,wj_anneal_seeded
- 2026-11-20 19:14:26 UTC harvested elf-c01 v2 (complete): kernel gain 0, merged gain 0 on 0 puzzles (0 rejected), 0.6 CPU-h, store now 133966
- 2026-11-20 19:15:13 UTC dataset versioned ograhulrajkumar/elf-compute with store total 133966 (24 files)
- 2026-11-20 19:15:15 UTC launched elf-c01 v3: 30 puzzles (len 252..273, 7846 instr), budget 2.5 h, engines elfopt,wj_anneal,wj_anneal_seeded,wj_incremental,wj_incremental_seeded,wj_compact_rotate,wj_quartet
- 2026-11-20 19:15:16 UTC launched elf-c02 v1: 30 puzzles (len 230..251, 7236 instr), budget 2.5 h, engines elfopt,wj_anneal,wj_anneal_seeded,wj_incremental,wj_incremental_seeded,wj_compact_rotate,wj_quartet
- 2026-11-20 19:15:18 UTC launched elf-c03 v1: 30 puzzles (len 223..230, 6802 instr), budget 2.5 h, engines elfopt,wj_anneal,wj_anneal_seeded,wj_incremental,wj_incremental_seeded,wj_compact_rotate,wj_quartet
- 2026-11-20 19:15:20 UTC launched elf-c04 v1: 30 puzzles (len 215..223, 6557 instr), budget 2.5 h, engines elfopt,wj_anneal,wj_anneal_seeded,wj_incremental,wj_incremental_seeded,wj_compact_rotate,wj_quartet
- 2026-11-20 19:15:22 UTC launched elf-c05 v1: 30 puzzles (len 209..215, 6348 instr), budget 2.5 h, engines elfopt,wj_anneal,wj_anneal_seeded,wj_incremental,wj_incremental_seeded,wj_compact_rotate,wj_quartet
- 2026-11-20 19:15:23 UTC push elf-c06 FAILED: Maximum batch CPU session count of 5 reached.
- 2026-11-20 19:15:24 UTC push elf-c07 FAILED: Maximum batch CPU session count of 5 reached.
- 2026-11-20 19:15:25 UTC push elf-c08 FAILED: Maximum batch CPU session count of 5 reached.
- 2026-11-20 19:15:26 UTC push elf-c09 FAILED: Maximum batch CPU session count of 5 reached.
- 2026-11-20 19:15:27 UTC push elf-c10 FAILED: Maximum batch CPU session count of 5 reached.
- 2026-11-20 19:15:37 UTC account limit found: 'Maximum batch CPU session count of 5 reached' -> max_kernels set to 5
- 2026-11-20 19:15:56 UTC fleet loop started (pid 15372)
- 2026-11-20 19:20:57 UTC elf-c01 v3 ended with status error: 
- 2026-11-20 19:20:58 UTC harvested elf-c01 v3 (error): kernel gain ?, merged gain 0 on 0 puzzles (0 rejected), 0.0 CPU-h, store now 133964
- 2026-11-20 19:20:58 UTC   elf-c01 produced no stats.json (files: []); see /home/user/classy/elf/store/kaggle_out/elf-c01_v3
- 2026-11-20 19:21:17 UTC dataset versioned ograhulrajkumar/elf-compute with store total 133964 (24 files)
- 2026-11-20 19:21:18 UTC launched elf-c01 v4: 30 puzzles (len 252..273, 7846 instr), budget 2.5 h, engines elfopt,wj_anneal,wj_anneal_seeded,wj_incremental,wj_incremental_seeded,wj_compact_rotate,wj_quartet
- 2026-11-20 19:26:38 UTC dataset versioned ograhulrajkumar/elf-compute with store total 133960 (24 files)
- 2026-11-20 19:29:13 UTC fixed runner compile race (c01 v3 died on it; c02-c05 and c01 v4 survived), added recomb engine (elfopt LNS + CP-SAT recombination), adaptive band weights; next wave samples programs of 80-180 instructions
- 2026-11-20 19:29:28 UTC fleet loop started (pid 16470)
- 2026-11-20 19:34:29 UTC fleet code changed, re-exec
- 2026-11-20 19:34:29 UTC fleet loop started (pid 16470)
- 2026-11-20 19:47:55 UTC added wj_anneal_div / wj_incremental_div (donor-seeded crossover with diversity-preserving archive, our implementation of royalacecat's idea)
- 2026-11-20 19:49:52 UTC dataset versioned ograhulrajkumar/elf-compute with store total 133941 (26 files)
- 2026-11-20 20:20:19 UTC dataset versioned ograhulrajkumar/elf-compute with store total 133923 (27 files)
- 2026-11-20 20:22:14 UTC elfopt/recomb now use elfopt_v (emits equal-length plateau programs); the runner keeps up to 8 value-distinct donors per puzzle for the seeded/diverse WJ engines
- 2026-11-20 20:25:40 UTC dataset versioned ograhulrajkumar/elf-compute with store total 133923 (27 files)
- 2026-11-20 20:52:55 UTC lever B sources snapshot into kaggle/engines/leverB (16 files)
- 2026-11-20 20:55:45 UTC fleet code changed, re-exec
- 2026-11-20 20:55:45 UTC fleet loop started (pid 16470)
- 2026-11-20 20:56:16 UTC dataset versioned ograhulrajkumar/elf-compute with store total 133862 (43 files)
- 2026-11-20 20:59:25 UTC coordinator request: lever B deep group rebuild (research/round4/beyond, FLEET_SPEC.md) added as runner mode 'pydriver' (kernel rebuilds src/ + beyond/ tree, kernel-local store, grun.py 8 4 ids 8 10, then jrun.py on 60-bit; skips puzzles in runs_g50/g60 resp. runs_j60). Sources frozen in kaggle/engines/leverB (refresh: python3 kaggle/fleet.py snapshot-leverb). Slots from the next launches: c01-c04 leverB_g (40 puzzles, 3.5 h), c05 public engines on the 80-180 band. Local pydriver test: scored_0001 229 -> 228 in 5 min on 1 core.
- 2026-11-20 21:01:16 UTC fleet code changed, re-exec
- 2026-11-20 21:01:16 UTC fleet loop started (pid 16470)

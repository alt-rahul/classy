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

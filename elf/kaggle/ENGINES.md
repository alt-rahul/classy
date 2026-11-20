# Running an engine on the Kaggle fleet

The fleet (kaggle/fleet.py, log store/logs/kaggle_compute.md) runs any engine that speaks one of two stdin/stdout
protocols. Each kernel has 4 vCPUs, about 30 GB RAM, g++ and Python 3 (no internet, no pip); it compiles the engine
from source at start, runs `workers` (4) engine processes at a time on its shard, re-validates every returned program
and keeps strictly shorter ones.

## Protocols
- `"kind": "elf"` (goodpjw elfopt format). One process per call, one puzzle per call:
  stdin  `P <id> <k> s1..sk <m> t1..tm <limit> <regs> <n>` then n lines `<op> <i> <j> <d>` (current best program);
  args   `--time <seconds> --seed <int>` plus the engine's own `args`;
  stdout `R <id> <n_old> <n_new>` then n_new lines `<op> <i> <j> <d>`, only when strictly shorter (anything else ignored).
- `"kind": "wj"` (Wang Jakie format). stdin: `1`, then `limit regs rounds 0 seed`, `k s1..sk`, `m t1..tm` (unique,
  sorted), `ntapes`, then per tape `n` and n lines `op a b d` (tape 0 = current best; seeded engines also get
  near-length variants); stdout one line `n op a b d op a b d ...`.

## Plugging in a new engine (research agent)
1. Put the sources (single directory, unique file names) in `kaggle/engines/extra/`.
2. Add an entry to `engines` in `kaggle/fleet_config.json`, e.g.
   `{"name": "fable_x", "kind": "elf", "main": "x.cpp", "includes": ["x_util.h"], "time": 60, "args": [], "weight": 3}`
   Optional `"compile": ["g++", "-O3", "-march=native", "-std=c++17", "-o", "{exe}", "{src}/x.cpp"]`
   (`{exe}` and `{src}` are substituted). `weight` is the pick probability relative to other engines.
3. The next poll re-versions the dataset (engine fingerprint changed) and every kernel launched afterwards uses it.
   Kernels already running keep their engines until they finish. To switch the whole fleet, remove the old entries.
To ask the fleet operator for a run instead, write the request (engine, puzzle set, budget) to
`kaggle/requests/<name>.md`.

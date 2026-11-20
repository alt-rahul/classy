"""Wide-pool cone rebuild over the store. usage: wrun.py TAG KMIN TL WORKERS id1,id2,... [cminoff]
Per puzzle: for targets by descending exclusive-cone size k >= KMIN, search a rebuild with <= k-1 new values
(wide.cpp over all DAG values not depending on the target). Improvements are validated and offered to the store.
Resumable via runs_<TAG>.jsonl (puzzles already logged are skipped)."""
import sys, time, json, os
from multiprocessing import Pool
from wlib import *

def work(a):
    pid, kmin, tl, cminoff = a
    P = {p.id: p for p in load_puzzles()}; p = P[pid]
    st = load_store(); prog = st[pid]['prog']; L0 = len(prog)
    nodes, order = build_dag(p, prog)
    tried = set(); log = []; t00 = time.time()
    while True:
        cones = all_cones(p, nodes)
        cand = [(len(c[0]), t) for t, c in cones.items() if len(c[0]) >= kmin and (t, len(c[0])) not in tried]
        if not cand: break
        k, t = max(cand); tried.add((t, k))
        cone, desc, users = cones[t]
        pool = [v for v in list(dict.fromkeys(p.sv)) + order if v not in cone and v not in desc]
        t0 = time.time()
        sols, err = run_wide_enum(p.limit, t, pool, max(4, k - 1 - cminoff), k - 1, 1, tl, 300)
        ops = min(sols, key=len) if sols else None
        res = None
        if sols:
            pr, bp = best_splice(p, nodes, order, cone, sols, t, len(prog))
            res = 'noemit%d/%d' % (bp, len(sols))
            if pr is not None:
                prog = pr; nodes, order = build_dag(p, prog); res = 'ok'
                st = load_store()
                if offer(st, p, prog, 'wide'): save_store(st)
        log.append((t.bit_length(), k, None if ops is None else len(ops), res, round(time.time() - t0, 1)))
    return pid, L0, len(prog), log, round(time.time() - t00, 1)

if __name__ == '__main__':
    tag, kmin, tl, nw, ids = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(',')
    cminoff = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    RUN = os.path.join(HERE, 'runs_%s.jsonl' % tag)
    done = set(json.loads(l)[0] for l in open(RUN)) if os.path.exists(RUN) else set()
    ids = [i for i in ids if i not in done]
    with Pool(nw) as pool:
        for r in pool.imap_unordered(work, [(i, kmin, tl, cminoff) for i in ids]):
            open(RUN, 'a').write(json.dumps(r) + '\n')
            print(r[0], r[1], '->', r[2], 'tried', len(r[3]), 'found', sum(1 for x in r[3] if x[2] is not None), 'ok', sum(1 for x in r[3] if x[3] == 'ok'), r[4], 's', flush=True)

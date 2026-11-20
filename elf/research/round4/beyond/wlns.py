"""Deep LNS for big random-like puzzles: plateau walks with equal-cost floor rebuilds of single targets (wide2
enumeration) + group rebuilds whenever a group's joint exclusive cone exceeds floor*|A|.
usage: wlns.py TAG SECONDS_PER_PUZZLE WORKERS idsfile[,idsfile] [seed]"""
import sys, time, json, os, random
from multiprocessing import Pool
from wlib import *
from gcone2 import best_groups
from gcone import dsets
from grun import attempt

def work(a):
    pid, budget, seed = a
    rng = random.Random(seed * 1000003 + hash(pid) % 1000)
    P = {p.id: p for p in load_puzzles()}; p = P[pid]
    st = load_store(); prog = st[pid]['prog']; L0 = len(prog)
    bits = p.limit.bit_length(); thr = 7 if bits >= 56 else 6 if bits >= 46 else 5
    t00 = time.time(); stats = collections.Counter(); tried_groups = set(); hist = []
    best = prog
    def groups_pass():
        nonlocal prog, best
        while time.time() - t00 < budget:
            nodes, order = build_dag(p, prog)
            D = dsets(p, nodes, order)
            def gkey(A): return (A, frozenset(v for v in order if D[v] and D[v] <= A))
            gs = [g for g in best_groups(p, nodes, order, thr, 8) if gkey(g[2]) not in tried_groups]
            if not gs: return
            gain, jc, A = gs[0]; tried_groups.add(gkey(A))
            pr, res = attempt(p, prog, nodes, order, A, thr, 8)
            stats['g_' + res[:6]] += 1
            if pr is not None:
                prog = pr; best = pr; hist.append((round(time.time() - t00), len(pr)))
                s2 = load_store()
                if offer(s2, p, pr, 'wide-lns'): save_store(s2)
    groups_pass()
    while time.time() - t00 < budget:
        nodes, order = build_dag(p, prog)
        cones = all_cones(p, nodes)
        cand = [t for t, c in cones.items() if len(c[0]) >= thr - 1]
        if not cand: break
        t = rng.choice(cand); cone, desc, users = cones[t]; k = len(cone)
        pool = [v for v in list(dict.fromkeys(p.sv)) + order if v not in cone and v not in desc]
        sols, _ = run_wide_enum(p.limit, t, pool, k, k, 1, 6, 4)
        stats['q'] += 1
        cur = set(cone)
        sols = [s for s in sols if set(x[3] for x in s) != cur and len(s) <= k]
        if not sols: stats['noalt'] += 1; continue
        s = rng.choice(sols)
        nn, cands = splice_blocks(p, nodes, order, cone, [s])
        if not cands or len(nn) > len(nodes): stats['bad'] += 1; continue
        pr = try_emit5(p, nn, cands, 12.0)
        if pr is None: stats['nosched'] += 1; continue
        if len(pr) > len(prog): stats['bad'] += 1; continue
        if len(pr) < len(prog):
            best = pr; hist.append((round(time.time() - t00), len(pr))); stats['strict'] += 1
            s2 = load_store()
            if offer(s2, p, pr, 'wide-lns'): save_store(s2)
        else: stats['plateau'] += 1
        prog = pr
        groups_pass()
    return pid, L0, len(best), dict(stats), hist, round(time.time() - t00, 1), best

if __name__ == '__main__':
    tag, budget, nw = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    ids = []
    for f in sys.argv[4].split(','): ids += [x.strip() for x in open(os.path.join(HERE, f)).read().split(',') if x.strip()]
    RUN = os.path.join(HERE, 'runs_%s.jsonl' % tag)
    done = set(json.loads(l)[0] for l in open(RUN)) if os.path.exists(RUN) else set()
    ids = [i for i in ids if i not in done]
    with Pool(nw) as pool:
        for r in pool.imap_unordered(work, [(i, budget, seed) for i in ids]):
            open(RUN, 'a').write(json.dumps(r[:6]) + '\n')
            if r[2] < r[1]: open(os.path.join(HERE, 'improved_%s.jsonl' % tag), 'a').write(json.dumps([r[0], r[6]]) + '\n')
            print(r[0], r[1], '->', r[2], r[3], r[4], r[5], 's', flush=True)

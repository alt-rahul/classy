"""Group rebuild: remove the joint exclusive cone JC(A) of a group A of targets (all values that only targets of A
depend on) and rebuild the targets of A one after another with wide2 from the remaining DAG values (+ the new ones),
keeping the result only if the program gets shorter.
usage: grun.py TAG TL WORKERS idsfile1,idsfile2,... [maxk] [maxgroups]"""
import sys, time, json, os, itertools
from multiprocessing import Pool
from wlib import *
from gcone import dsets
from gcone2 import best_groups

def best_insert(p, sub, cur, nv, sv):
    """best position for block nv in order cur (partial DAG sub incl. nv). Returns ((peak, n_at_peak), order) or None."""
    pos = {v: i for i, v in enumerate(cur)}; bs = set(nv)
    lo = 0; hi = len(cur)
    for v in nv:
        for u in sub[v][1:]:
            if u in pos: lo = max(lo, pos[u] + 1)
            elif u not in sv and u not in bs: return None
    for i, w in enumerate(cur):
        if sub[w][1] in bs or sub[w][2] in bs: hi = min(hi, i)
    if lo > hi: return None
    best = None
    for ins in range(lo, hi + 1):
        od = cur[:ins] + nv + cur[ins:]
        pr = prof_of(p, sub, od); mx = max(pr)
        s = (mx, sum(1 for x in pr if x >= mx - 1))
        if best is None or s < best[0]: best = (s, od)
    return best

def attempt(p, prog, nodes, order, A, thr, tl):
    """rebuild the targets of the joint exclusive cone of A one by one; among the enumerated solutions of each target
    take the one whose best insertion keeps the register peak lowest."""
    sv = set(p.sv)
    D = dsets(p, nodes, order)
    JC = set(v for v in order if D[v] and D[v] <= A)
    tg = [v for v in order if v in JC and v in set(p.targets)]
    budget = len(JC) - 1
    pool = [v for v in list(dict.fromkeys(p.sv)) + order if v not in JC]
    sub = {v: nodes[v] for v in order if v not in JC}
    cur = [v for v in order if v not in JC]
    rem = len(tg); nfail = 0
    for t in sorted(tg):
        if t in sub or t in sv: rem -= 1; continue
        cmax = min(thr + 1, budget - (thr - 1) * (rem - 1))
        sols = []
        if cmax >= thr - 1:
            sols, _ = run_wide_enum(p.limit, t, pool, max(1, min(thr, cmax)), cmax, 1, tl, 6)
            if not sols and rem <= 3:   # a late failure wastes the whole group: search the top level longer once
                sols, _ = run_wide_enum(p.limit, t, pool, cmax, cmax, 1, tl * 4, 2)
        if not sols:
            # keep the old tree of this target: restore its ancestors inside the joint cone
            nfail += 1
            if nfail > max(1, len(tg) // 3): return None, 'nosol%d/%d' % (len(tg) - rem, len(tg))
            anc = set(); stk = [t]
            while stk:
                x = stk.pop()
                if x in anc or x not in JC or x in sub: continue
                anc.add(x); stk += list(nodes[x][1:])
            nv = [v for v in order if v in anc]
            sub2 = dict(sub); sub2.update({v: nodes[v] for v in nv})
            r = best_insert(p, sub2, cur, nv, sv)
            if r is None: return None, 'norestore'
            cur = r[1]; sub = sub2; budget -= len(nv); rem -= 1
            for v in nv:
                if v not in pool: pool.append(v)
            if budget < 0: return None, 'budget'
            continue
        m = min(len(s) for s in sols); best = None
        for s in sols:
            if len(s) > m: continue
            nv = [v for (op, a, b, v) in s if v not in sub and v not in sv]
            sub2 = dict(sub); sub2.update({v: (op, a, b) for (op, a, b, v) in s if v not in sub and v not in sv})
            r = best_insert(p, sub2, cur, nv, sv)
            if r is not None and (best is None or r[0] < best[0]): best = (r[0], r[1], sub2, nv)
        if best is None: return None, 'noins'
        _, cur, sub, nv = best
        budget -= len(nv); rem -= 1
        for v in nv:
            if v not in pool: pool.append(v)
    # DCE + completeness
    T = [x for x in dict.fromkeys(p.targets) if x not in sv]
    need = set(); stk = list(T)
    while stk:
        x = stk.pop()
        if x in need or x in sv: continue
        if x not in sub: return None, 'incomplete'
        need.add(x); op, a, b = sub[x]; stk += [a, b]
    nn = {v: r for v, r in sub.items() if v in need}
    cur = [v for v in cur if v in nn]
    if len(nn) >= len(nodes): return None, 'nogain'
    pr = try_emit5(p, nn, [cur], 40.0)
    if pr is None or len(pr) >= len(prog): return None, 'noemit%d' % pressure(p, nn, cur)
    return pr, 'ok'

def attempt2(p, prog, nodes, order, A, thr, tl):
    """pre-screened variant: find the floor rebuild of every target of the group over DAG - JC(A) first, drop the
    targets that have none, and rebuild the remaining sub-group if its joint cone is still larger than the sum."""
    sv = set(p.sv); Tset = set(p.targets)
    D = dsets(p, nodes, order)
    JC = set(v for v in order if D[v] and D[v] <= A)
    tg = [v for v in order if v in JC and v in Tset]
    pool = [v for v in list(dict.fromkeys(p.sv)) + order if v not in JC]
    S = {}
    slack = len(JC) - thr * len(tg)          # predicted gain
    for t in sorted(tg):
        sols, _ = run_wide_enum(p.limit, t, pool, thr, thr + (1 if slack >= 2 else 0), 1, tl, 6)
        if sols:
            m = min(len(x) for x in sols); S[t] = [x for x in sols if len(x) == m]
    if not S: return None, 'nosol0/%d' % len(tg)
    A2 = frozenset(t for t in A if t in S or t not in JC)
    JC2 = set(v for v in order if D[v] and D[v] <= A2)
    tg2 = [t for t in tg if t in JC2 and t in S]
    need = sum(len(S[t][0]) for t in tg2)
    if not tg2 or need >= len(JC2): return None, 'sub%d/%d:%d>=%d' % (len(tg2), len(tg), need, len(JC2))
    sub = {v: nodes[v] for v in order if v not in JC2}
    cur = [v for v in order if v not in JC2]
    for t in sorted(tg2):
        if t in sub: continue
        best = None
        for s in S[t]:
            if any((a not in sub and a not in sv and a not in set(x[3] for x in s)) or (b not in sub and b not in sv and b not in set(x[3] for x in s)) for (op, a, b, v) in s): continue
            nv = [v for (op, a, b, v) in s if v not in sub and v not in sv]
            sub2 = dict(sub); sub2.update({v: (op, a, b) for (op, a, b, v) in s if v not in sub and v not in sv})
            r = best_insert(p, sub2, cur, nv, sv)
            if r is not None and (best is None or r[0] < best[0]): best = (r[0], r[1], sub2)
        if best is None: return None, 'noins'
        _, cur, sub = best
    T = [x for x in dict.fromkeys(p.targets) if x not in sv]
    needv = set(); stk = list(T)
    while stk:
        x = stk.pop()
        if x in needv or x in sv: continue
        if x not in sub: return None, 'incomplete'
        needv.add(x); op, a, b = sub[x]; stk += [a, b]
    nn = {v: r for v, r in sub.items() if v in needv}
    cur = [v for v in cur if v in nn]
    if len(nn) >= len(nodes): return None, 'nogain'
    pr = try_emit5(p, nn, [cur], 40.0)
    if pr is None or len(pr) >= len(prog): return None, 'noemit%d' % pressure(p, nn, cur)
    return pr, 'ok%d/%d' % (len(tg2), len(tg))

def work(a):
    pid, tl, maxk, maxg = a
    P = {p.id: p for p in load_puzzles()}; p = P[pid]
    st = load_store(); prog = st[pid]['prog']; L0 = len(prog)
    bits = p.limit.bit_length(); thr = 7 if bits >= 56 else 6 if bits >= 46 else 5
    log = []; t00 = time.time(); tried = set(); ng = 0
    while ng < maxg and time.time() - t00 < 1200:
        nodes, order = build_dag(p, prog)
        gs = [g for g in best_groups(p, nodes, order, thr, maxk) if g[2] not in tried]
        if not gs: break
        gain, jc, A = gs[0]; tried.add(A); ng += 1
        t0 = time.time()
        pr, res = (attempt2 if os.environ.get('GRUN2') else attempt)(p, prog, nodes, order, A, thr, tl)
        if pr is not None:
            prog = pr
            st = load_store()
            if offer(st, p, prog, 'wide-group'): save_store(st)
        log.append((len(A), jc, gain, res, round(time.time() - t0, 1)))
    return pid, L0, len(prog), log, round(time.time() - t00, 1), prog

if __name__ == '__main__':
    tag, tl, nw = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    maxk = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    maxg = int(sys.argv[6]) if len(sys.argv) > 6 else 6
    ids = []
    for f in sys.argv[4].split(','): ids += [x.strip() for x in open(os.path.join(HERE, f)).read().split(',') if x.strip()]
    RUN = os.path.join(HERE, 'runs_%s.jsonl' % tag)
    done = set(json.loads(l)[0] for l in open(RUN)) if os.path.exists(RUN) else set()
    ids = [i for i in ids if i not in done]
    with Pool(nw) as pool:
        for r in pool.imap_unordered(work, [(i, tl, maxk, maxg) for i in ids]):
            open(RUN, 'a').write(json.dumps(r[:5]) + '\n')
            if r[2] < r[1]: open(os.path.join(HERE, 'improved_%s.jsonl' % tag), 'a').write(json.dumps([r[0], r[5]]) + '\n')
            if r[3]: print(r[0], r[1], '->', r[2], r[3], r[4], 's', flush=True)

"""Wide-pool cone rebuild driver helpers."""
import subprocess, collections, os, sys, time, json
from dagutil import *
import solver
HERE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(HERE, os.environ.get('WIDEBIN', 'wide'))

def build_dag(p, prog):
    """nodes: value -> (op,a,b) for needed values (DCE'd), order: list of values in program order, torder: target completion order"""
    nodes = {}; order = []
    for (op, a, b, v) in values(p, prog):
        if v in nodes or v in p.sv: continue
        nodes[v] = (op, a, b); order.append(v)
    sv = set(p.sv)
    need = set(); st = [t for t in set(p.targets) if t not in sv]
    while st:
        x = st.pop()
        if x in need or x in sv: continue
        need.add(x); op, a, b = nodes[x]; st += [a, b]
    order = [v for v in order if v in need]
    return {v: nodes[v] for v in order}, order

def cone_of(nodes, sv, T, t):
    users = collections.defaultdict(set)
    for v, (op, a, b) in nodes.items():
        users[a].add(v); users[b].add(v)
    cone = {t}; stack = [t]
    while stack:
        x = stack.pop(); op, a, b = nodes[x]
        for u in (a, b):
            if u in cone or u in sv or u in T: continue
            if users[u] <= cone: cone.add(u); stack.append(u)
    # descendants of t
    desc = set(); stack = [t]
    while stack:
        x = stack.pop()
        for w in users[x]:
            if w not in desc: desc.add(w); stack.append(w)
    return cone, desc, users

def all_cones(p, nodes):
    sv = set(p.sv); T = set(p.targets) - sv
    return {t: cone_of(nodes, sv, T, t) for t in T}

def run_wide(limit, t, pool, cmin, cmax, heavy=1, tl=120):
    inp = '%d %d %d %d %d\n%d\n%s\n' % (limit, t, cmin, cmax, heavy, len(pool), ' '.join(map(str, pool)))
    r = subprocess.run([WIDE, str(tl)], input=inp, capture_output=True, text=True)
    lines = r.stdout.strip().split('\n')
    if not lines or not lines[0].startswith('SOL'): return None, r.stderr
    ops = []
    for ln in lines[1:]:
        a, op, b, v = ln.split(); ops.append((op, int(a), int(b), int(v)))
    return ops, r.stderr

def steps_from(nodes, order):
    return [(nodes[v][0], nodes[v][1], nodes[v][2], v) for v in order]

def dfs_order(p, nodes, torder):
    sv = set(p.sv); done = set(); out = []
    def emit(v):
        st = [(v, 0)]
        while st:
            x, s = st.pop()
            if x in done or x in sv: continue
            op, a, b = nodes[x]
            if s == 0:
                st.append((x, 1)); st.append((b, 0)); st.append((a, 0))
            else:
                done.add(x); out.append(x)
    for t in torder: emit(t)
    return out

def splice(p, nodes, order, cone, newops, t):
    """returns new (nodes, order) candidates"""
    sv = set(p.sv)
    nn = {v: nodes[v] for v in order if v not in cone}
    newvals = []
    for (op, a, b, v) in newops:
        if v in nn or v in sv: continue
        nn[v] = (op, a, b); newvals.append(v)
    # DCE
    T = [x for x in dict.fromkeys(p.targets) if x not in sv]
    need = set(); st = list(T)
    while st:
        x = st.pop()
        if x in need or x in sv: continue
        need.add(x); op, a, b = nn[x]; st += [a, b]
    nn = {v: r for v, r in nn.items() if v in need}
    base = [v for v in order if v in nn and v not in newvals]
    newvals = [v for v in newvals if v in nn]
    cands = []
    pos = {v: i for i, v in enumerate(base)}
    leaves = set()
    for v in newvals:
        for u in nn[v][1:]:
            if u in pos: leaves.add(u)
    lastleaf = max([pos[u] for u in leaves], default=-1)
    users_t = [pos[v] for v in base if t in nn[v][1:]]
    firstuse = min(users_t, default=len(base))
    # original rank of t
    oi = order.index(t); rank = sum(1 for v in order[:oi] if v in pos)
    for ins in dict.fromkeys([max(lastleaf + 1, min(rank, firstuse)), lastleaf + 1, firstuse]):
        if lastleaf + 1 <= ins <= firstuse:
            cands.append(base[:ins] + newvals + base[ins:])
    # dfs orders
    tfirst = [v for v in order if v in set(T)]
    cands.append(dfs_order(p, nn, tfirst))
    return nn, cands

def try_emit(p, nn, cands):
    for od in cands:
        if len(od) != len(nn): continue
        prog = solver.regalloc(p.sv, steps_from(nn, od), p.targets, p.nreg)
        if prog is not None and check(p, prog) is None:
            return prog
    return None

import random
def list_sched(p, nn, hint, rng=None, noise=0.0):
    """greedy list scheduling of the value DAG nn (value -> (op,a,b)); hint: value -> preferred position."""
    sv = set(p.sv)
    us = collections.defaultdict(list); indeg = {}; rem = collections.Counter()
    for v, (op, a, b) in nn.items():
        ops = {a, b}
        indeg[v] = sum(1 for u in ops if u not in sv)
        for u in ops: us[u].append(v); rem[u] += 1
    ready = set(v for v in nn if indeg[v] == 0)
    out = []
    while ready:
        best = None; bs = 1e18
        for v in ready:
            op, a, b = nn[v]
            fr = (1 if rem[a] == 1 else 0) + (1 if b != a and rem[b] == 1 else 0)
            s = (1 if us.get(v) else 0) - fr + 1e-6 * hint.get(v, 0) + (rng.random() * noise if noise else 0)
            if s < bs: bs = s; best = v
        ready.discard(best); out.append(best)
        op, a, b = nn[best]
        rem[a] -= 1
        if b != a: rem[b] -= 1
        for y in us.get(best, ()):
            indeg[y] -= 1
            if indeg[y] == 0: ready.add(y)
    return out

def make_hint(nn, base_order, sv):
    hint = {v: float(i) for i, v in enumerate(base_order)}
    # new nodes: just after their latest operand
    changed = True
    while changed:
        changed = False
        for v, (op, a, b) in nn.items():
            if v in hint: continue
            ha = hint.get(a, -1.0 if a in sv else None); hb = hint.get(b, -1.0 if b in sv else None)
            if ha is None or hb is None: continue
            hint[v] = max(ha, hb) + 0.01; changed = True
    return hint

def try_emit2(p, nn, cands, base_order, tries=300, seed=1):
    pr = try_emit(p, nn, cands)
    if pr is not None: return pr
    sv = set(p.sv); hint = make_hint(nn, base_order, sv)
    rng = random.Random(seed)
    for k in range(tries):
        od = list_sched(p, nn, hint, rng, 0.0 if k == 0 else (0.3 if k % 3 == 1 else 1.0 if k % 3 == 2 else 2.0))
        pr = try_emit(p, nn, [od])
        if pr is not None: return pr
    return None

def pressure(p, nn, od):
    """max simultaneously live values (registers needed) for order od"""
    sv = list(dict.fromkeys(p.sv)); last = {}
    for i, v in enumerate(od):
        op, a, b = nn[v]; last[a] = i; last[b] = i
    live = sum(1 for s in sv if s in last); mx = live
    die = collections.Counter(last.values())
    for i, v in enumerate(od):
        live -= die[i]          # operands dying at i free before write
        mx = max(mx, live + 1)
        if v in last: live += 1
    return mx

def improve_order(p, nn, od, nreg, sweeps=6):
    """local search on a topological order: move single ops within their feasible window to reduce
    (peak, number of steps above nreg-1). Returns order (maybe still infeasible)."""
    sv = set(p.sv)
    def profile(od):
        last = {}
        for i, v in enumerate(od):
            op, a, b = nn[v]; last[a] = i; last[b] = i
        live = sum(1 for s in dict.fromkeys(p.sv) if s in last)
        die = collections.Counter(last.values()); prof = []
        for i, v in enumerate(od):
            live -= die[i]; prof.append(live + 1)
            if v in last: live += 1
        return prof
    def score(prof):
        mx = max(prof); return (mx, sum(1 for x in prof if x >= min(mx, nreg + 1)))
    od = list(od); cur = score(profile(od))
    for sw in range(sweeps):
        if cur[0] <= nreg: break
        improved = False
        prof = profile(od); mx = cur[0]
        peaks = [i for i, x in enumerate(prof) if x >= mx - 1]
        lo_pk, hi_pk = peaks[0], peaks[-1]
        pos = {v: i for i, v in enumerate(od)}
        users = collections.defaultdict(list)
        for v in od:
            op, a, b = nn[v]; users[a].append(v); users[b].append(v)
        cand = list(range(len(od)))
        random.Random(sw).shuffle(cand)
        for i0 in cand:
            v = od[i0]; i = od.index(v)
            op, a, b = nn[v]
            lo = max([pos_ for pos_ in (od.index(u) for u in (a, b) if u not in sv)], default=-1) + 1
            hi = min([od.index(w) for w in users[v]], default=len(od)) - 1
            if hi <= lo and lo == i: continue
            best = None
            # candidate positions: a sample of the window
            window = list(range(lo, hi + 1))
            if len(window) > 24: window = sorted(random.Random(i0 + sw).sample(window, 24))
            for j in window:
                if j == i: continue
                o2 = od[:i] + od[i + 1:]; o2.insert(j, v)
                s = score(profile(o2))
                if s < cur and (best is None or s < best[0]): best = (s, o2)
            if best:
                cur, od = best; improved = True
                if cur[0] <= nreg: break
        if not improved: break
    return od

def try_emit3(p, nn, cands, base_order):
    pr = try_emit(p, nn, cands)
    if pr is not None: return pr
    full = [c for c in cands if len(c) == len(nn)]
    full.sort(key=lambda c: pressure(p, nn, c))
    for c in full[:2]:
        od = improve_order(p, nn, c, p.nreg)
        pr = try_emit(p, nn, [od])
        if pr is not None: return pr
    return None

def run_wide_enum(limit, t, pool, cmin, cmax, heavy=1, tl=30, cap=200, binname='wide2'):
    inp = '%d %d %d %d %d\n%d\n%s\n' % (limit, t, cmin, cmax, heavy, len(pool), ' '.join(map(str, pool)))
    r = subprocess.run([os.path.join(HERE, binname), str(tl), str(cap)], input=inp, capture_output=True, text=True)
    sols = []; cur = None
    for ln in r.stdout.split('\n'):
        f = ln.split()
        if not f: continue
        if f[0] == 'SOL': cur = []; sols.append(cur)
        elif f[0] not in ('ENUM', 'NONE') and cur is not None: cur.append((f[1], int(f[0]), int(f[2]), int(f[3])))
    return sols, r.stderr

def best_splice(p, nodes, order, cone, sols, t, proglen):
    """try all solutions; return (prog, n_tried, best_pressure)"""
    bestp = 999
    for ops in sorted(sols, key=len):
        nn, cands = splice(p, nodes, order, cone, ops, t)
        if len(nn) >= len(nodes): continue
        for c in cands:
            if len(c) == len(nn): bestp = min(bestp, pressure(p, nn, c))
        pr = try_emit(p, nn, cands)
        if pr is not None and len(pr) < proglen: return pr, bestp
    return None, bestp

def prof_of(p, nn, od):
    last = {}
    for i, v in enumerate(od):
        op, a, b = nn[v]; last[a] = i; last[b] = i
    live = sum(1 for s in dict.fromkeys(p.sv) if s in last)
    die = collections.Counter(last.values()); prof = []
    for i, v in enumerate(od):
        live -= die[i]; prof.append(live + 1)
        if v in last: live += 1
    return prof

def perm_sched(p, nn, torder, nreg, iters=4000, seed=1, thr=None):
    """local search over the target order with lazy DFS emission; minimises (peak, overflow area)."""
    rng = random.Random(seed)
    thr = nreg - 3 if thr is None else thr
    def ev(to):
        od = dfs_order(p, nn, to); pr = prof_of(p, nn, od)
        return (max(pr), sum(x - thr for x in pr if x > thr)), od
    cur, od = ev(torder); to = list(torder); n = len(to)
    best = (cur, od, list(to))
    for it in range(iters):
        if cur[0] <= nreg and it > 0 and thr >= nreg - 3: break
        i = rng.randrange(n); j = rng.randrange(n)
        if i == j: continue
        t2 = list(to); x = t2.pop(i); t2.insert(j, x)
        s, o2 = ev(t2)
        if s <= cur:
            cur, od, to = s, o2, t2
            if s < best[0]: best = (s, o2, list(t2))
    return best

"""Pair-collision experiment/driver: enumerate equal-cost alternatives of the large cones of one puzzle,
then look for pairs (and sets) of targets whose alternatives share new nodes.
usage: penum.py PID KMIN TL NW [apply]"""
import sys, time, json, itertools, pickle
from multiprocessing import Pool
from wlib import *
ENUMBIN = os.path.join(HERE, 'wide_enum')

def enum_one(a):
    limit, t, pool, k, tl, cap = a
    inp = '%d %d %d %d %d\n%d\n%s\n' % (limit, t, k, k, 1, len(pool), ' '.join(map(str, pool)))
    r = subprocess.run([ENUMBIN, str(tl), str(cap)], input=inp, capture_output=True, text=True)
    sols = []; cur = None
    for ln in r.stdout.split('\n'):
        f = ln.split()
        if not f: continue
        if f[0] == 'SOL': cur = []; sols.append(cur)
        elif f[0] not in ('ENUM', 'NONE') and cur is not None: cur.append((f[1], int(f[0]), int(f[2]), int(f[3])))
    return t, [s for s in sols if len(s) <= k]

def analyse(p, prog, kmin, tl, nw, cap=4000):
    nodes, order = build_dag(p, prog)
    cones = all_cones(p, nodes)
    big = {t: c for t, c in cones.items() if len(c[0]) >= kmin}
    jobs = []
    for t, (cone, desc, users) in big.items():
        pool = [v for v in list(dict.fromkeys(p.sv)) + order if v not in cone and v not in desc]
        jobs.append((p.limit, t, pool, len(cone), tl, cap))
    with Pool(nw) as pl: res = dict(pl.map(enum_one, jobs))
    return nodes, order, cones, big, res

def find_pairs(p, nodes, big, res):
    """pairs of targets with alternatives whose union of new nodes is smaller than the two cones"""
    out = []
    ts = list(big)
    # index alternatives by node value
    for i, j in itertools.combinations(ts, 2):
        ci, cj = big[i][0], big[j][0]; di, dj = big[i][1], big[j][1]
        if i in dj or j in di: continue
        bad_i = cj | dj; bad_j = ci | di   # values alternatives of i (resp. j) must not read
        def ok(s, bad):
            return all(a not in bad and b not in bad for (op, a, b, v) in s)
        Ai = [s for s in res[i] if ok(s, bad_i)]; Aj = [s for s in res[j] if ok(s, bad_j)]
        idx = collections.defaultdict(list)
        for x, s in enumerate(Ai):
            for (op, a, b, v) in s[:-1]: idx[v].append(x)
        best = None
        for y, s in enumerate(Aj):
            vs = set(v for (op, a, b, v) in s)
            for v in vs:
                for x in idx.get(v, ()):
                    u = len(vs | set(w[3] for w in Ai[x]))
                    if u < len(ci) + len(cj) and (best is None or u < best[0]): best = (u, x, y)
        if best: out.append((len(ci) + len(cj) - best[0], i, j, Ai[best[1]], Aj[best[2]]))
    return sorted(out, key=lambda z: -z[0])

if __name__ == '__main__':
    pid, kmin, tl, nw = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    P = {p.id: p for p in load_puzzles()}; st = load_store(); p = P[pid]
    t0 = time.time()
    nodes, order, cones, big, res = analyse(p, st[pid]['prog'], kmin, tl, nw)
    print(pid, 'len', len(order), 'big cones', len(big), 'alts per target', sorted(len(v) for v in res.values()), '%.0fs' % (time.time() - t0))
    prs = find_pairs(p, nodes, big, res)
    print('pair gains', [(g, i.bit_length()) for g, i, j, a, b in prs][:20], 'npairs', len(prs))
    pickle.dump((res, prs), open('/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad/penum_%s.pkl' % pid, 'wb'))

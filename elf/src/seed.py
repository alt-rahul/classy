"""Round 3 seeding for shared-like puzzles (agent A's best-of-three, agent B's hub miner).

shared_like(p): not independent / tiny, and >= 30% of the targets have a continued-fraction relation
q*T_j - p*T_i = r of strength >= 12 bits to another target (strength = log2(T_i) - log2 p - log2 q -
log2(|r|+1), as in research/round3/generator_shared/seeded.py).
Variants (best one is offered):
  c  : mined constants (top 20 CF-relation values <= 1e5) are built first by a separate solve and then
       passed to the main solve as extra starts (agent A's seeded.py);
  h  : mined hubs (agent B's mine2 + seed_proto.pick with STRICT=1, plus gcds >= 1e4 of accepted hubs)
       are extra targets; the final regalloc on the original targets drops unused ones (DCE);
  ch : both.
"""
import sys, os, math, copy, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import solver
from solver import Solver, Cfg, regalloc, simulate


# ---------------- continued-fraction relations (agent A) ----------------
def cfrels(a, b, pm=10**7):
    """yield (p, q, r): q*a - p*b = r for the convergents p/q of a/b"""
    h0, h1, k0, k1 = 0, 1, 1, 0
    x, y = a, b
    while y:
        c = x // y; x, y = y, x - c * y
        h0, h1 = h1, c * h1 + h0; k0, k1 = k1, c * k1 + k0
        if h1 > pm or k1 > pm:
            break
        if h1 == 0:
            continue
        yield h1, k1, k1 * a - h1 * b


def best_rels(T):
    """for each pair i < j the strongest relation (s, p, q, r) of T[j] against T[i] (seeded.mine order)"""
    out = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            best = None
            for pp, q, r in cfrels(T[j], T[i]):
                s = math.log2(T[i]) - math.log2(pp) - math.log2(q) - math.log2(abs(r) + 1)
                if best is None or s > best[0]:
                    best = (s, pp, q, r)
            if best:
                out.append((i, j) + best)
    return out


def rel_fraction(p, thr=12, rels=None):
    T = p.targets
    rels = best_rels(T) if rels is None else rels
    linked = set()
    for (i, j, s, pp, q, r) in rels:
        if s > thr:
            linked.add(i); linked.add(j)
    return len(linked) / max(len(T), 1)


def mine_consts(p, thr=12, amax=10**5, rels=None):
    """agent A's seeded.mine: weights of small values appearing in strong relations"""
    T = p.targets; S = {}
    def add(v, w):
        v = abs(v)
        if 1 < v <= amax and v not in p.sv:
            S[v] = S.get(v, 0) + w
    rels = best_rels(T) if rels is None else rels
    for (i, j, s, pp, q, r) in rels:
        if s > thr:
            add(pp, s); add(q, s)
            if q == 1:
                a = round(r / pp); c = r - pp * a
                add(a, s / 2); add(c, s / 2)
            elif pp == 1:
                a = round(r / q); c = r - q * a
                add(a, s / 2); add(c, s / 2)
    return S


def top_consts(p, K=20, rels=None):
    S = mine_consts(p, rels=rels)
    return [v for v, w in sorted(S.items(), key=lambda x: -x[1])[:K]]


# ---------------- family test (agent B's fam.py) ----------------
def decade(p):
    dg = collections.Counter(len(str(t)) for t in p.targets)
    k = max((6, 9, 12, 15, 18), key=lambda k: dg[k])
    return k, [t for t in p.targets if len(str(t)) == k]


def fam(p):
    if p.limit <= 16:
        return 'tiny'
    if max(p.sv) > 40:
        return 'indep'
    k, r = decade(p)
    if len(r) >= max(4, len(p.targets) // 2 - 1):
        return 'mixed'
    return 'sm'


def shared_like(p, frac=0.3, rels=None):
    if fam(p) in ('tiny', 'indep'):
        return False
    return rel_fraction(p, rels=rels) >= frac


# ---------------- hub miner (agent B's mine2.mine + seed_proto.pick STRICT=1) ----------------
def mine_hubs(T, C=2500, M=2500, minB=10**7, minhits=2):
    import numpy as np
    cs = np.arange(-C, C + 1, dtype=np.int64); ms = np.arange(1, M + 1, dtype=np.int64)
    hit = collections.defaultdict(dict)
    for ti, t in enumerate(T):
        if t < minB:
            continue
        v = t + cs
        for lo in range(0, len(ms), 256):
            mm = ms[lo:lo + 256]
            R = v[:, None] % mm[None, :]
            ci, mi = np.nonzero(R == 0)
            for a, b in zip(ci.tolist(), mi.tolist()):
                m = int(mm[b]); c = int(cs[a]); B = (t + c) // m
                if B < minB:
                    continue
                d = hit[B]
                if ti not in d or abs(c) + m < abs(d[ti][1]) + d[ti][0]:
                    d[ti] = (m, c)
    return {B: d for B, d in hit.items() if len(d) >= minhits}


def _gain(B, d, T):
    g = -B.bit_length()
    for ti, (m, c) in d.items():
        g += max(0, T[ti].bit_length() - m.bit_length() - abs(c).bit_length() - 2)
    return g


def pick_hubs(T, hubs, maxhubs=10, ming=20):
    """seed_proto.pick with STRICT=1: gain >= 20 bits and >= 2 newly explained targets per hub"""
    sc = {B: _gain(B, d, T) for B, d in hubs.items()}
    rank = sorted((B for B in hubs if sc[B] >= ming), key=lambda B: -sc[B])
    cov = set(); out = []
    for B in rank:
        d = hubs[B]
        new = [ti for ti in d if ti not in cov]
        if any(abs(B - b2) <= 6000 for b2, _ in out):
            continue
        if len(new) >= 2:
            out.append((B, d)); cov |= set(d)
        if len(out) >= maxhubs:
            break
    return out


def hub_seeds(p):
    """hidden hubs + gcds >= 1e4 between accepted hubs (seed_proto.seeds), values <= limit"""
    T = p.targets
    sel = pick_hubs(T, mine_hubs(T))
    Tset = set(T) | set(p.sv)
    H = [B for B, d in sel if B not in Tset]
    allh = [B for B, d in sel]
    G = set()
    for i in range(len(allh)):
        for j in range(i):
            g = math.gcd(allh[i], allh[j])
            if g >= 10**4 and g not in Tset and g not in allh:
                G.add(g)
    return [x for x in H + sorted(G) if x <= p.limit]


# ---------------- seeded solves ----------------
def clone(p, sv, targets):
    q = copy.copy(p); q.sv = list(sv); q.targets = sorted(set(targets)); return q


def run_steps(p, kw):
    s = Solver(p, Cfg(**kw))
    return s.run(), s.stats


def seeded(p, smalls, extra_targets, kw):
    """smalls built first (separate solve), then the main solve with them as extra starts and
    extra_targets added; regalloc on the original targets (unused extras are dead code)."""
    smalls = [v for v in smalls if v not in p.sv and v <= p.limit]
    stA = run_steps(clone(p, p.sv, smalls), kw)[0] if smalls else []
    extra = [x for x in extra_targets if x <= p.limit]
    stB, stats = run_steps(clone(p, list(p.sv) + smalls, list(p.targets) + extra), kw)
    steps = list(stA) + list(stB)
    simulate(p.sv, steps, p.limit)
    return regalloc(p.sv, steps, p.targets, p.nreg), steps


def variants(p, kw, which=('c', 'h', 'ch')):
    """returns {variant: prog or None}, plus the mined seeds"""
    rels = best_rels(p.targets)
    consts = top_consts(p, rels=rels)
    hubs = hub_seeds(p) if any('h' in v for v in which) else []
    res = {}
    for v in which:
        sm = consts if 'c' in v else []
        ex = hubs if 'h' in v else []
        if not sm and not ex:
            res[v] = None
            continue
        prog, steps = seeded(p, sm, ex, kw)
        res[v] = prog if prog is not None and check(p, prog) is None else None
    return res, consts, hubs

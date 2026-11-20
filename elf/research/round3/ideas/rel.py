"""Relation-chain search (round 3, survivor S1b).

t = (Z +- d) * M2 +- c2   with   Z = (v -+ c1) / M1,   v a pool value,
all constants c1, M1, d, M2, c2 from the small values of L0 u L1 of the current pool (or absent).
Z is a hidden hub: it is never enumerated in a level set, it is found by meeting the reverse closure
of the target with the reverse closures of the pool values. Every plan found is exact arithmetic.
"""
import numpy as np, ctypes as C
import eng3
from eng3 import engine, _lib

CMAX = 4096

def small_consts(cmax=CMAX):
    """(values, costs) of all integers 1..cmax that sit in L0..L2 of the current engine levels."""
    vs, cs = [], []
    for v in range(1, cmax + 1):
        k = engine.level_of(v, 2)
        if k >= 0:
            vs.append(v); cs.append(k)
    return np.array(vs, dtype=np.int64), np.array(cs, dtype=np.int64)

class Rel:
    def __init__(self, limit, cmax=CMAX):
        self.limit = limit
        self.cv, self.cc = small_consts(cmax)
        # additive constants: 0 (free) plus the small consts with both signs
        self.av = np.concatenate([[0], self.cv, -self.cv]).astype(np.int64)
        self.ac = np.concatenate([[0], self.cc + 1, self.cc + 1]).astype(np.int64)
        m = (self.cv >= 2) & (self.cc <= 1)
        self.mv = np.concatenate([[1], self.cv[m]]).astype(np.int64)
        self.mc = np.concatenate([[0], self.cc[m] + 1]).astype(np.int64)
        self.dcost = np.full(cmax + 1, 99, dtype=np.int64)
        self.dcost[self.cv] = self.cc + 1
        self.dcost[0] = 0
        self.cmax = cmax

    def closure(self, x):
        """All (Z, cost, c, M) with x = Z * M + c (c signed, |c| const or 0; M const or 1), Z >= 2,
        Z * M <= limit. Arrays sorted by Z."""
        r = x - self.av                      # Z * M
        ok = (r >= 2) & (r <= self.limit)
        r = r[ok]; ac = self.ac[ok]; av = self.av[ok]
        rem = r[:, None] % self.mv[None, :]
        ii, jj = np.nonzero(rem == 0)
        Z = r[ii] // self.mv[jj]
        cost = ac[ii] + self.mc[jj]
        keep = Z >= 2
        Z, cost, c, M = Z[keep], cost[keep], av[ii][keep], self.mv[jj][keep]
        o = np.lexsort((cost, Z))
        Z, cost, c, M = Z[o], cost[o], c[o], M[o]
        first = np.ones(len(Z), dtype=bool); first[1:] = Z[1:] != Z[:-1]
        return Z[first], cost[first], c[first], M[first]

    def index_pool(self, pool, minv=1 << 16):
        zs, cs, c1, m1, vv = [], [], [], [], []
        for v in pool:
            if v < minv: continue
            Z, cost, c, M = self.closure(v)
            zs.append(Z); cs.append(cost); c1.append(c); m1.append(M); vv.append(np.full(len(Z), v, dtype=np.int64))
        if not zs:
            self.PZ = np.zeros(0, dtype=np.int64); return
        Z = np.concatenate(zs); o = np.argsort(Z, kind='stable')
        self.PZ = Z[o]; self.Pc = np.concatenate(cs)[o]; self.Pc1 = np.concatenate(c1)[o]
        self.PM = np.concatenate(m1)[o]; self.Pv = np.concatenate(vv)[o]

    def best(self, t, maxcost):
        """Cheapest relation plan for t with cost <= maxcost: (cost, steps) or None."""
        if len(self.PZ) == 0: return None
        Z, cost, c, M = self.closure(t)
        sel = cost <= maxcost
        Z, cost, c, M = Z[sel], cost[sel], c[sel], M[sel]
        if len(Z) == 0: return None
        lo = np.searchsorted(self.PZ, Z - self.cmax, 'left'); hi = np.searchsorted(self.PZ, Z + self.cmax, 'right')
        best = None
        for i in np.nonzero(hi > lo)[0]:
            a, b = lo[i], hi[i]
            d = Z[i] - self.PZ[a:b]
            tot = cost[i] + self.Pc[a:b] + self.dcost[np.abs(d)]
            j = int(np.argmin(tot))
            if tot[j] <= maxcost and (best is None or tot[j] < best[0]):
                best = (int(tot[j]), int(Z[i]), int(c[i]), int(M[i]), int(d[j]), int(self.PZ[a + j]), int(self.Pc1[a + j]), int(self.PM[a + j]), int(self.Pv[a + j]))
        if best is None: return None
        tot, Zt, c2, M2, d, Zv, c1, M1, v = best
        if tot <= 0: return None
        st = []
        def const(x):
            return engine.expand(x)
        cur = v
        if c1:
            st += const(abs(c1)); nxt = cur - c1
            st.append(('-', cur, c1, nxt) if c1 > 0 else ('+', cur, -c1, nxt)); cur = nxt
        if M1 != 1:
            st += const(M1); st.append(('/', cur, M1, cur // M1)); cur //= M1
        assert cur == Zv
        if d:
            st += const(abs(d)); nxt = cur + d
            st.append(('+', cur, d, nxt) if d > 0 else ('-', cur, -d, nxt)); cur = nxt
        assert cur == Zt
        if M2 != 1:
            st += const(M2); st.append(('*', cur, M2, cur * M2)); cur *= M2
        if c2:
            st += const(abs(c2)); nxt = cur + c2
            st.append(('+', cur, c2, nxt) if c2 > 0 else ('-', cur, -c2, nxt)); cur = nxt
        assert cur == t, (cur, t, best)
        for s in st:
            if not (1 <= s[3] <= self.limit) or s[1] < 1 or s[2] < 1: return None
        return tot, st

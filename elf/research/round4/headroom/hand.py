"""Semi-automatic structured construction for practice_0015 (exp 2).
Stage A: solver builds the hubs (as a puzzle). Stage B: every target is finished from a big value H already
built (hub, target or intermediate) by T = p*H + c or T = p*(H + a) + c (and twin forms), the constants p, a, c
are costed exactly on L0..L2 of ALL values built so far (no pool cap; register feasibility checked at the end)."""
import sys, json, time, pickle, copy, itertools
from rs import *
P = {p.id: p for p in load_puzzles('practice')}

class Builder:
    def __init__(self, p, steps=()):
        self.p = p; self.steps = list(steps); self.vals = list(dict.fromkeys(list(p.sv) + [s[3] for s in self.steps]))
        self.vset = set(self.vals); self.dirty = True
    def smalls(self, cap=1 << 26):
        return [v for v in self.vals if v <= cap]
    def lev(self):
        if self.dirty:
            engine.build(self.smalls(), self.p.limit, 2, 0); self.dirty = False
    def cost(self, c, maxc=5):
        if c == 0: return 0, []
        if c in self.vset: return 0, []
        self.lev()
        r = engine.find_pair(c, maxc)
        if r is None: return None
        return r[0], r[1]
    def add(self, st):
        for s in st:
            if s[3] not in self.vset:
                self.steps.append(tuple(s)); self.vals.append(s[3]); self.vset.add(s[3]); self.dirty = True
    def op(self, o, a, b):
        v = {'+': a + b, '-': a - b, '*': a * b, '/': a // b}[o]
        assert v >= 1 and v <= self.p.limit and (o != '/' or a % b == 0)
        self.add([(o, a, b, v)]); return v

def plans(B, T, bigs, pmax=1 << 24):
    """all finishing plans for T: list of (cost, steps)"""
    out = []
    lim = B.p.limit
    def chain(seq):
        # seq: start value, then (op, const) ...; returns (cost, steps) or None
        v = seq[0]; cost = 0; st = []
        for o, c in seq[1:]:
            if c == 0: continue
            r = B.cost(c)
            if r is None: return None
            cost += r[0] + 1; st += r[1]
            nv = {'+': v + c, '-': v - c, '*': v * c}[o]
            if nv < 1 or nv > lim: return None
            st.append((o, v, c, nv)); v = nv
        return (cost, st) if v == T else None
    for H in bigs:
        if H == T: continue
        if abs(T - H) < pmax:
            d = T - H
            r = chain([H, ('+' if d > 0 else '-', abs(d))])
            if r: out.append(r)
        q = T // H
        for p in (q, q + 1):
            if p < 2 or p > pmax: continue
            c = T - p * H
            r = chain([H, ('*', p), ('+' if c > 0 else '-', abs(c))])
            if r: out.append(r)
            # p*(H+a)+c
            a0 = c // p
            for a in (a0, a0 + 1):
                c2 = T - p * (H + a)
                if a == 0: continue
                r = chain([H, ('+' if a > 0 else '-', abs(a)), ('*', p), ('+' if c2 > 0 else '-', abs(c2))])
                if r: out.append(r)
            # factor p = p1*p2 with inner constant: p1*(p2*H + a) + c  (e.g. 32*(31*B-256)+1)
            for p1 in range(2, 2050):
                if p % p1: continue
                p2 = p // p1
                if p2 < 2: continue
                a0 = c // p1
                for a in (a0, a0 + 1):
                    c2 = T - p1 * (p2 * H + a)
                    if a == 0 or abs(c2) > 4: continue
                    r = chain([H, ('*', p2), ('+' if a > 0 else '-', abs(a)), ('*', p1), ('+' if c2 > 0 else '-', abs(c2))])
                    if r: out.append(r)
    return out

def finish(B, targets, bigs, verbose=1):
    todo = [t for t in targets if t not in B.vset]
    tidx = {t: i for i, t in enumerate(B.p.targets)}
    left = []
    while todo:
        best = None
        for T in todo:
            ps = plans(B, T, bigs)
            if ps:
                c, st = min(ps, key=lambda x: x[0])
                if best is None or c < best[0]: best = (c, T, st)
        if best is None: break
        c, T, st = best
        n0 = len(B.steps); B.add(st)
        if verbose: print('T%d cost %d (+%d): %s' % (tidx[T], c, len(B.steps) - n0, ' ; '.join('%d%s%d' % (s[1], s[0], s[2]) for s in st)), flush=True)
        todo.remove(T); bigs = bigs + [s[3] for s in st if s[3] > 1 << 30]
    return todo

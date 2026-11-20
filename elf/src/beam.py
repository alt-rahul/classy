"""Beam search over target order (round 2, item 4).

State = solver pool after committing some targets. Expanding a state builds fresh levels L0..L3 for
its pool, finds the cheapest pair plan (<= maxc) for every uncovered target, and makes one child per
target among the best `branch` by regret score (cost - alpha * bits). Targets with no pair plan get a
window / Horner / fallback plan through the normal solver. Children are ranked by
ops so far + gamma * (sum of the parent's cheapest costs of the targets still uncovered).
"""
import math, random, time
import engine
from solver import Solver, Cfg, regalloc, simulate


class State:
    __slots__ = ('pool', 'last_use', 'created', 'uses', 'steps', 'unc', 'rnd', 'ops')

    def clone(self):
        s = State()
        s.pool = list(self.pool); s.last_use = dict(self.last_use); s.created = dict(self.created)
        s.uses = dict(self.uses); s.steps = list(self.steps); s.unc = set(self.unc)
        s.rnd = self.rnd; s.ops = self.ops
        return s


def plans_for(sv, st, ts, maxc):
    """Cheapest pair plan per target on current levels: dict t -> (cost, steps)."""
    out = {}
    for t in ts:
        r = engine.find_pair(t, maxc)
        if r is not None:
            out[t] = r
    return out


def beam_solve(p, cfg, width=8, branch=4, alpha=0.1, gamma=0.7, maxc=7, unk=9, seed=0, deadline=None):
    rng = random.Random(seed)
    sol = Solver(p, cfg)
    sol.n_l3 = 0
    s0 = State()
    s0.pool = list(dict.fromkeys(p.sv)); s0.last_use = {v: 0 for v in s0.pool}; s0.created = dict(s0.last_use)
    s0.uses = {}; s0.steps = []; s0.unc = set(p.targets) - set(p.sv); s0.rnd = 0; s0.ops = 0
    beam = [(0.0, s0)]
    while True:
        if all(not s.unc for _, s in beam):
            break
        cands = []
        for _, s in beam:
            if not s.unc:
                cands.append((s.ops, rng.random(), s, None))
                continue
            sol.uses = s.uses
            order = sol.pool_order(s.pool, s.last_use)
            ts = sorted(s.unc)
            engine.build(order, p.limit, 3, 0)
            pl = plans_for(p.sv, s, ts, maxc)
            costs = {t: (pl[t][0] if t in pl else unk) for t in ts}
            scored = sorted(pl, key=lambda t: (pl[t][0] - alpha * t.bit_length(), pl[t][0], t))
            pick = scored[:branch]
            opts = [(t, pl[t][0], pl[t][1]) for t in pick]
            if not opts:
                sol.deep_age = None; sol.evicted = False
                c, t, stp, how = sol.search_round(ts, order, s.pool)
                opts = [(t, c, stp)]
            tot = sum(costs.values())
            for (t, c, stp) in opts:
                rest = tot - costs[t]
                cands.append((s.ops + c + gamma * rest, rng.random(), s, (t, c, stp)))
        cands.sort(key=lambda x: (x[0], x[1]))
        nb = []
        seen = set()
        for sc, _, s, mv in cands:
            if len(nb) >= width:
                break
            if mv is None:
                nb.append((sc, s)); continue
            c2 = s.clone()
            sol.uses = c2.uses
            c2.rnd += 1
            n0 = len(c2.steps)
            sol.commit(c2.pool, mv[2], c2.last_use, c2.created, c2.rnd, c2.unc, c2.steps)
            c2.ops += len(c2.steps) - n0
            key = (frozenset(c2.unc), c2.ops)
            if key in seen:
                continue
            seen.add(key)
            nb.append((sc, c2))
        beam = nb
        if deadline is not None and time.time() > deadline:
            # finish greedily from the best state
            break
    best = None
    for _, s in beam:
        if s.unc:
            # complete with the plain solver loop from this state
            s = finish(p, cfg, s)
        simulate(p.sv, s.steps, p.limit)
        prog = regalloc(p.sv, s.steps, p.targets, p.nreg)
        if prog is not None and (best is None or len(prog) < len(best)):
            best = prog
    return best


def finish(p, cfg, s):
    sol = Solver(p, cfg)
    sol.uses = s.uses; sol.n_l3 = 0; sol.deep_age = None; sol.evicted = False
    while s.unc:
        s.rnd += 1
        order = sol.pool_order(s.pool, s.last_use)
        ts = sorted(s.unc)
        best = sol.regret_round(ts, order, s.pool) if cfg.regret > 0 else sol.search_round(ts, order, s.pool)
        sol.commit(s.pool, best[2], s.last_use, s.created, s.rnd, s.unc, s.steps)
    return s

"""Round-3 solver variants (subclass of src/solver.Solver; src is untouched)."""
import sys, time, collections, json
import eng3
from eng3 import engine
from common import *
from solver import Solver, Cfg, regalloc, simulate


class S3(Solver):
    """hub_g > 0: before the regret pick, scan L1/L2 for a hub value u whose presence in the pool
    lowers the summed pair costs of the uncovered targets by at least hub_g more than u costs;
    commit u alone and continue."""
    def regret_round(self, ts, pool_order, pool):
        cfg = self.cfg
        G = getattr(cfg, 'hub_g', 0)
        if G > 0 and len(ts) >= 2:
            engine.build(pool_order, self.p.limit, 3, 0)
            D = getattr(cfg, 'hub_default', 9)
            base = eng3.all_costs(ts, 7, D)
            best = None
            for lu in getattr(cfg, 'hub_levels', (1, 2)):
                for h in eng3.hub_scan(ts, base, lu, getattr(cfg, 'hub_lw', 3), G, 2):
                    key = (h[1], h[2], -lu, -h[0])
                    if best is None or key > best[0]:
                        best = (key, lu, h[0])
            if best is not None:
                _, lu, u = best
                st = engine.expand(u)
                self.stats['hubgain'] = self.stats.get('hubgain', 0) + best[0][0]
                return (lu, u, st, 'hub')
        return Solver.regret_round(self, ts, pool_order, pool)


class LA(Solver):
    """Plan-choice lookahead: the target is picked by regret score on exact pair costs; among its
    enumerated plans of minimal cost (and optionally cost+1) choose the one minimising
    extra_cost + sum of the other targets' pair costs with the plan's new values in the pool.
    cfg: la_k (max plans evaluated), la_plus (also evaluate plans of cost+1), la_d (default cost),
    la_t (number of best targets by regret score whose plans compete)."""
    def regret_round(self, ts, pool_order, pool):
        cfg, limit = self.cfg, self.p.limit
        K = getattr(cfg, 'la_k', 6); D = getattr(cfg, 'la_d', 8); NT = getattr(cfg, 'la_t', 1)
        plus = getattr(cfg, 'la_plus', 0)
        if len(ts) < 2 or len(pool_order) < getattr(cfg, 'la_minpool', 12):
            return Solver.regret_round(self, ts, pool_order, pool)
        engine.build(pool_order, limit, 3, 0)
        base = eng3.all_costs(ts, cfg.regret_maxc, None)
        known = [(base[i] - cfg.regret * ts[i].bit_length(), base[i], ts[i]) for i in range(len(ts)) if base[i] is not None]
        if not known:
            return Solver.regret_round(self, ts, pool_order, pool)
        known.sort()
        cands = []   # (extra, t, steps)
        pset = set(pool_order)
        for (sc, c, t) in known[:NT]:
            extra0 = sc - known[0][0]
            if c <= 3:
                cands.append((extra0, c, t, engine.expand(t)))
                continue
            for dc in range(plus + 1):
                if c + dc > 7: break
                recs = eng3.enum_pairs(t, c + dc)
                if len(recs) > K:
                    recs = self.rng.sample(recs, K)
                for rec in recs:
                    cands.append((extra0 + dc, c + dc, t, eng3.plan_steps(t, rec)))
        if len(cands) == 1:
            e, c, t, st = cands[0]
            self.deep_age = None
            return (c, t, st, 'reg')
        best = None
        tset = set(ts)
        for (e, c, t, st) in cands:
            newv = [s[3] for s in st if s[3] not in pset]
            got = set(newv) & tset
            rest = [u for u in ts if u not in got]
            engine.build(pool_order + newv, limit, 3, 0)
            cs = eng3.all_costs(rest, 7, D)
            phi = e + sum(cs)
            key = (phi, c)
            if best is None or key < best[0]:
                best = (key, c, t, st)
        self.stats['la_eval'] = self.stats.get('la_eval', 0) + len(cands)
        self.deep_age = None
        return (best[1], best[2], best[3], 'la')


class EV(Solver):
    """Eviction variants. ev_mode 1: covered targets are evicted last. ev_mode 2: evict by usefulness:
    score(v) = sum over uncovered targets of max(0, ev_d - (lev(t|v) + 1)) where lev(t|v) is the lowest
    level holding x with t = v op x (levels of the pool before the commit); lowest score first,
    then largest. ev_mode 3: both."""
    def commit(self, pool, st, last_use, created, rnd, uncovered, steps):
        self._unc = uncovered
        self._tset = set(self.p.targets)
        return Solver.commit(self, pool, st, last_use, created, rnd, uncovered, steps)

    def evict(self, pool, last_use, created, cap, protect):
        if len(pool) <= cap:
            return
        mode = getattr(self.cfg, 'ev_mode', 1); D = getattr(self.cfg, 'ev_d', 6)
        cand = [v for v in pool if v > self.cfg.keep_small and v not in protect]
        sc = {}
        if mode in (2, 3):
            unc = list(self._unc)
            for v in cand:
                s = 0
                for t in unc:
                    k = eng3.resid_lev(t, v, 3)
                    if k >= 0 and k + 1 < D:
                        s += D - (k + 1)
                sc[v] = s
        cand.sort(key=lambda v: ((v in self._tset) if mode in (1, 3) else 0, sc.get(v, 0), -v.bit_length(), last_use.get(v, 0), created.get(v, 0)))
        k = len(pool) - cap
        drop = set(cand[:k])
        if drop:
            self.evicted = True
            self.stats['ev_scored'] = self.stats.get('ev_scored', 0) + sum(1 for v in drop if sc.get(v, 0) > 0)
        pool[:] = [v for v in pool if v not in drop]


class BIG(Solver):
    """Wide pool: no eviction (run with a fake nreg), the search pool is the whole history.
    Regret rounds use L3 while the pool is small (<= big_l3 values), else L2 only (pairs to cost 5);
    if nothing is found on L2, fall back to the normal round on the 46 most recent / smallest values."""
    def regret_round(self, ts, pool_order, pool):
        cfg, limit = self.cfg, self.p.limit
        if len(pool_order) <= getattr(cfg, 'big_l3', 52):
            return Solver.regret_round(self, ts, pool_order, pool)
        sz = engine.build(pool_order, limit, 2, 0)
        self.deep_age = None
        best = engine.regret_pick(ts, 5, cfg.regret)
        if cfg.verbose:
            print('   BIG pool', len(pool_order), 'levels', sz, 'costs', eng3.all_costs(ts, 5, 9))
        if best is not None:
            return (best[0], best[1], best[2], 'big')
        self.stats['bigfail'] = self.stats.get('bigfail', 0) + 1
        sub = sorted(pool, key=lambda v: (-v.bit_length(), 0))[-46:]
        sub = self.pool_order(sub, {})
        return Solver.regret_round(self, ts, sub, pool)


def width(sv, steps, targets):
    """Registers actually needed by the DCE'd program (regalloc with unlimited registers)."""
    prog = regalloc(sv, steps, targets, 100000)
    return max(max(i, j, d) for (_, i, j, d) in prog) + 1, len(prog)


class PEEL(Solver):
    """Regret ordering on min(pair cost, chain cost). Chain (peel) plans: t = (y op v1) op v2 with
    y in L<=3 and v1, v2 in L<=1: shapes the pair search cannot see. peel_min: only targets whose pair
    cost is >= peel_min (or unknown) get the peel search."""
    def regret_round(self, ts, pool_order, pool):
        cfg, limit = self.cfg, self.p.limit
        if len(pool_order) < getattr(cfg, 'peel_minpool', 10):
            return Solver.regret_round(self, ts, pool_order, pool)
        engine.build(pool_order, limit, 3, 0)
        self.n_l3 += 1
        self.deep_age = None
        base = eng3.all_costs(ts, cfg.regret_maxc, None)
        pmin = getattr(cfg, 'peel_min', 6); pmax = getattr(cfg, 'peel_max', 7)
        tiers = eng3.TIERS[:getattr(cfg, 'peel_tiers', 4)]
        best = None
        for t, b in zip(ts, base):
            plan = None; c = b
            if b is None or b >= pmin:
                r = eng3.peel(t, min(pmax, (b - 1) if b is not None else pmax), tiers)
                if r is not None:
                    c, plan = r
                p3 = getattr(cfg, 'peel3', 0)
                if p3 and (c is None or c >= 7):
                    r = eng3.peel3(t, min(p3, (c - 1) if c is not None else p3))
                    if r is not None:
                        c, plan = r
                        self.stats['p3hit'] = self.stats.get('p3hit', 0) + 1
            if c is None:
                continue
            key = (c - cfg.regret * t.bit_length(), c, t)
            if best is None or key < best[0]:
                best = (key, c, t, plan)
        if best is None:
            return Solver.regret_round(self, ts, pool_order, pool)
        _, c, t, plan = best
        if plan is None:
            r = engine.find_pair(t, c)
            return (r[0], t, r[1], 'reg')
        self.stats['peelsave'] = self.stats.get('peelsave', 0) + 1
        return (c, t, plan, 'peel')

    # optional: protect chain bases from eviction (peel_protect=1). A pool value y is a base for an
    # uncovered target t if t = (y op v1) op v2 with v1, v2 in L<=1 (levels of the pre-commit pool).
    def commit(self, pool, st, last_use, created, rnd, uncovered, steps):
        self._unc = uncovered
        return Solver.commit(self, pool, st, last_use, created, rnd, uncovered, steps)

    def evict(self, pool, last_use, created, cap, protect):
        if len(pool) <= cap or not getattr(self.cfg, 'peel_protect', 0):
            return Solver.evict(self, pool, last_use, created, cap, protect)
        prot = set(protect)
        pset = set(pool)
        for t in self._unc:
            for (la, lb, full) in eng3.TIERS:
                c = eng3._lib.peel2(t, la, lb, 0, full, 4, eng3._PB)
                if c >= 0:
                    if eng3._PB[0] in pset and eng3._PB[0] > self.cfg.keep_small:
                        prot.add(eng3._PB[0])
                    break
        self.stats['protected'] = self.stats.get('protected', 0) + len(prot) - len(protect)
        if len(pool) - len([v for v in pool if v in prot]) < len(pool) - cap:
            prot = set(protect)
        return Solver.evict(self, pool, last_use, created, cap, prot)


def solve3(p, cls=S3, **kw):
    s = cls(p, Cfg(**kw))
    st = s.run()
    simulate(p.sv, st, p.limit)
    prog = regalloc(p.sv, st, p.targets, p.nreg)
    return prog, s.stats, st


if __name__ == '__main__':
    which = sys.argv[1]
    P = {p.id: p for p in load_puzzles(which)}
    ids = sys.argv[2].split(',')
    cfgs = [json.loads(c) for c in sys.argv[3:]]
    tot = [0] * len(cfgs)
    for i in ids:
        p = P[i]
        row = []
        for k, c in enumerate(cfgs):
            t0 = time.time()
            kw = dict(pool_cap=46, evict_size=1, regret=0.1); kw.update(c)
            cls = {'S3': S3, 'LA': LA, 'EV': EV, 'BIG': BIG, 'PEEL': PEEL}[kw.pop('cls', 'S3')]
            prog, stats, st = solve3(p, cls, **kw)
            err = check(p, prog) if prog else 'regalloc'
            n = len(prog) if prog and not err else 10**6
            tot[k] += n
            row.append('%d (%.0fs%s)' % (n, time.time() - t0, ' ' + ','.join('%s%d' % (k, stats[k]) for k in ('hub', 'peel', 'la') if stats.get(k))))
        print(i, p.family, len(p.targets), max(p.targets).bit_length(), ' | '.join(row), flush=True)
    print('TOTAL', tot)

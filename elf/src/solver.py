"""Greedy 'cheapest target next' solver over a live pool of register values.

SSA steps are tracked by value; registers are assigned afterwards (regalloc) with
dead-code elimination. Every produced program is validated by the caller.
"""
import math, random, time
import engine

OPF = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a // b,
}


def regalloc(sv, steps, targets, nreg=48):
    """steps: list of (op, a_val, b_val, r_val) in execution order; operands refer to the most
    recent definition of that value. Returns a register program or None if > nreg live."""
    cur = {}
    defs = []
    nid = 0
    start_ids = []
    for v in sv:
        if v in cur:
            start_ids.append(None)
            continue
        cur[v] = nid
        start_ids.append(nid)
        nid += 1
    val_of = {}
    for i, v in enumerate(sv):
        if start_ids[i] is not None:
            val_of[start_ids[i]] = v
    for (op, a, b, r) in steps:
        ia, ib = cur[a], cur[b]
        cur[r] = nid
        val_of[nid] = r
        defs.append((op, ia, ib, nid))
        nid += 1
    # DCE: keep first producer of each uncovered target and everything feeding kept steps
    tset = set(targets) - set(sv)
    first_prod = {}
    for k, (op, ia, ib, ir) in enumerate(defs):
        v = val_of[ir]
        if v in tset and v not in first_prod:
            first_prod[v] = k
    if len(first_prod) != len(tset):
        return None
    need = [False] * len(defs)
    for k in first_prod.values():
        need[k] = True
    def_of = {d[3]: k for k, d in enumerate(defs)}
    for k in range(len(defs) - 1, -1, -1):
        if need[k]:
            op, ia, ib, ir = defs[k]
            for x in (ia, ib):
                j = def_of.get(x)
                if j is not None:
                    need[j] = True
    defs = [d for k, d in enumerate(defs) if need[k]]
    last = {}
    for k, (op, ia, ib, ir) in enumerate(defs):
        last[ia] = k
        last[ib] = k
    reg = {}
    free = set(range(nreg))
    for i, sid in enumerate(start_ids):
        if sid is not None and sid in last:
            reg[sid] = i
            free.discard(i)
    prog = []
    for k, (op, ia, ib, ir) in enumerate(defs):
        ra, rb = reg[ia], reg[ib]
        for x in (ia, ib):
            if last.get(x) == k and x in reg:
                free.add(reg.pop(x))
        if not free:
            return None
        d = ra if ra in free else (rb if rb in free else min(free))
        free.discard(d)
        prog.append([op, ra, rb, d])
        if ir in last:
            reg[ir] = d
        else:
            free.add(d)
    return prog


def simulate(sv, steps, limit):
    """Check SSA steps by value (sanity check)."""
    known = set(sv)
    for (op, a, b, r) in steps:
        assert a in known and b in known, (op, a, b, r)
        if op == '-':
            assert a > b
        if op == '/':
            assert a % b == 0
        assert OPF[op](a, b) == r, (op, a, b, r)
        assert 1 <= r <= limit
        known.add(r)
    return known


class Cfg:
    def __init__(self, **kw):
        self.pool_cap = 40
        self.cheap_accept = 4     # commit without building L3 if a plan this cheap exists
        self.pair3_k = 3          # targets (smallest first) given the L3 pair search
        self.win_k = 1            # targets given the window search when nothing <= 7
        self.win_budget = 1.5e6
        self.horner_k = 12
        self.seed = 0
        self.keep_small = 64
        self.order = 'small'      # tie-break among equal cost: smallest target first
        self.verbose = 0
        self.wemin = 0.02
        self.l3_maxage = 4
        self.horner_above = 10
        self.lookahead = 0
        self.maxlev = 7
        self.cap = 20_000_000
        self.pair_max = 11
        self.deep_maxage = 1
        self.noise = 0.0
        self.evict_margin = 0
        self.la_slack = 1
        self.la_default = 7
        self.evict_size = 0       # 1: evict largest values first (by bit length, LRU within)
        self.evict_unused = 0     # 1: evict values never used as an operand first (LRU within)
        self.evict_big = 0        # >0: evict values with bit_length >= (min target bits - evict_big) first
        self.__dict__.update(kw)


class Solver:
    def __init__(self, p, cfg):
        self.p = p
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        engine.set_seed(cfg.seed + 1)
        engine.set_wemin(cfg.wemin)
        self.stats = {}
        self.times = {}
        self.uses = {}

    # ---------- deep search helpers ----------
    def window(self, t, bestcost):
        return engine.find_window(t, bestcost, self.cfg.win_budget)

    def horner(self, t, bestcost):
        """t = X*Y +- Z with X in L<=2, Z in L<=3, Y solved by pair search (<= 7)."""
        cand = engine.horner_candidates(t, 2, 3)
        if not cand:
            return None
        scored = []
        for (x, z, cx, cz, sg) in cand:
            y = (t - z) // x if sg > 0 else (t + z) // x
            if y < 1:
                continue
            c0 = cx + (cz + 2 if z else 1)
            est = c0 + y.bit_length() / 5.5
            scored.append((est, c0, x, z, sg, y))
        scored.sort()
        best = None
        for (est, c0, x, z, sg, y) in scored[:self.cfg.horner_k]:
            lim_c = (bestcost if best is None else best[0]) - c0 - 1
            if lim_c < 0:
                continue
            r = engine.find_pair(y, min(7, lim_c))
            if r is None:
                continue
            cy, sty = r
            tot = c0 + cy
            if best is None or tot < best[0]:
                stx = engine.expand(x)
                stz = engine.expand(z) if z else []
                mid = x * y
                st = sty + stx + stz + [('*', y, x, mid)]
                if z:
                    st.append(('+' if sg > 0 else '-', mid, z, t))
                best = (tot, st)
        return best

    def fallback(self, t):
        """Guaranteed (slow, poor) decomposition: Horner chain with greedy bits/op."""
        steps_acc = []
        cost = 0
        cur = t
        for depth in range(200):
            r = engine.find_pair(cur, 7)
            if r is not None:
                return cost + r[0], r[1] + steps_acc
            cand = engine.horner_candidates(cur, 2, 3)
            bx = None
            for (x, z, cx, cz, sg) in cand:
                cst = cx + (cz + 2 if z else 1)
                sc = math.log2(x) / cst
                if bx is None or sc > bx[0]:
                    bx = (sc, x, z, cx, cz, sg)
            if bx is None:
                return None
            _, x, z, cx, cz, sg = bx
            y = (cur - z) // x if sg > 0 else (cur + z) // x
            stx = engine.expand(x)
            stz = engine.expand(z) if z else []
            mid = y * x
            tail = stx + stz + [('*', y, x, mid)]
            if z:
                tail.append(('+' if sg > 0 else '-', mid, z, cur))
            steps_acc = tail + steps_acc
            cost += cx + (cz + 2 if z else 1)
            cur = y
        return None

    # ---------- main loop ----------
    def plan_ok(self, pool, st):
        known = set(pool)
        for (op, a, b, r) in st:
            if a not in known or b not in known:
                return False
            known.add(r)
        return True

    def cheap_tier(self, ts, pool_order):
        cfg = self.cfg
        if cfg.lookahead > 1:
            return self.lookahead(ts, pool_order)
        best = None
        for t in ts:
            mc = 5 if best is None else best[0] - 1
            if mc < 0:
                break
            r = engine.find_pair(t, mc)
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], t, r[1], 'p2')
        return best

    def search_round(self, ts, pool_order, pool):
        cfg, limit = self.cfg, self.p.limit
        # 1. cheap tier (uses stale deep levels when present; verified)
        engine.build(pool_order, limit, 2, 1)
        best = self.cheap_tier(ts, pool_order)
        if engine.derive_failed() or (best is not None and not self.plan_ok(pool, best[2])):
            self.stats['redo'] = self.stats.get('redo', 0) + 1
            engine.build(pool_order, limit, 2, 0)
            best = self.cheap_tier(ts, pool_order)
        if best is not None and best[0] <= cfg.cheap_accept:
            return best
        # 2. deep tier: fresh levels unless the stale ones are still fully derivable
        if self.deep_age is None or self.evicted or self.deep_age >= cfg.deep_maxage:
            engine.build(pool_order, limit, cfg.maxlev, 0, cfg.cap)
            self.n_l3 += 1
            self.deep_age = 0
            self.evicted = False
        else:
            engine.build(pool_order, limit, 2, 1)
        k3 = cfg.pair3_k if (best is None or best[0] > 6) else len(ts)
        for t in ts[:k3]:
            mc = cfg.pair_max if best is None else best[0] - 1
            if mc < 4:
                break
            r = engine.find_pair(t, mc)
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], t, r[1], 'p3')
        if best is None or best[0] > 7:
            for t in ts[:cfg.win_k]:
                bc = 99 if best is None else best[0]
                r = self.window(t, bc)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], t, r[1], 'win')
            if best is None or best[0] > cfg.horner_above:
                t = ts[0]
                bc = 99 if best is None else best[0]
                r = self.horner(t, bc)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], t, r[1], 'hor')
        if best is None:
            t = ts[0]
            r = self.fallback(t)
            assert r is not None, 'fallback failed'
            best = (r[0], t, r[1], 'fb')
        assert not engine.derive_failed()
        return best

    def lookahead(self, ts, pool_order):
        """Cheap-tier choice with one-step lookahead: minimise own cost + sum of the other
        targets' cheap-tier costs after committing the plan."""
        cfg, limit = self.cfg, self.p.limit
        D = cfg.la_default
        cost = {}
        plans = {}
        for t in ts:
            r = engine.find_pair(t, 5)
            if r is not None:
                cost[t] = r[0]
                plans[t] = r[1]
        if not cost:
            return None
        cands = sorted(cost, key=lambda t: (cost[t], t))
        cmin = cost[cands[0]]
        cands = [t for t in cands if cost[t] <= cmin + cfg.la_slack][:cfg.lookahead]
        if len(cands) == 1:
            t = cands[0]
            return (cost[t], t, plans[t], 'p2')
        base = {u: cost.get(u, D) for u in ts}
        tot0 = sum(base.values())
        bestsc = None
        pset = set(pool_order)
        for t in cands:
            newv = [s[3] for s in plans[t] if s[3] not in pset]
            engine.build(pool_order + newv, limit, 2, 1)
            gain = 0
            covered = set(newv)
            for u in ts:
                if u == t or u in covered:
                    gain += base[u]
                    continue
                c0 = base[u]
                r = engine.find_pair(u, min(5, c0 - 1))
                if r is not None:
                    gain += c0 - r[0]
            sc = cost[t] - gain
            if bestsc is None or sc < bestsc[0]:
                bestsc = (sc, t)
        engine.build(pool_order, limit, 2, 1)
        t = bestsc[1]
        r = engine.find_pair(t, 5)
        return (r[0], t, r[1], 'la')

    def run(self):
        p, cfg = self.p, self.cfg
        steps = []
        pool = list(dict.fromkeys(p.sv))
        last_use = {v: 0 for v in pool}
        created = {v: 0 for v in pool}
        uncovered = set(p.targets) - set(p.sv)
        rnd = 0
        self.n_l3 = 0
        self.deep_age = None
        self.evicted = False
        st_counts = self.stats
        while uncovered:
            rnd += 1
            order = self.pool_order(pool, last_use)
            if cfg.noise > 0:
                ts = sorted(uncovered, key=lambda t: t.bit_length() + cfg.noise * self.rng.random())
            else:
                ts = sorted(uncovered)
            best = self.search_round(ts, order, pool)
            assert self.plan_ok(pool, best[2])
            c, t, st, how = best
            st_counts[how] = st_counts.get(how, 0) + 1
            if cfg.verbose:
                print('round', rnd, how, 'cost', c, 'bits %.1f' % math.log2(t), 'uncov', len(uncovered), flush=True)
            self.commit(pool, st, last_use, created, rnd, uncovered, steps)
            if self.deep_age is not None:
                self.deep_age += 1
        st_counts['l3builds'] = self.n_l3
        return steps

    def commit(self, pool, st, last_use, created, rnd, uncovered, steps):
        p, cfg = self.p, self.cfg
        known_now = set(pool)
        new_vals = set(s[3] for s in st if s[3] not in known_now)
        room = p.nreg - 1 - len(new_vals)
        if len(pool) > room:
            used = set()
            for s in st:
                used.add(s[1]); used.add(s[2])
            self.evict(pool, last_use, created, room, protect=used)
            known_now = set(pool) | (known_now - set(pool))  # values still referenced must be in pool
            known_now = set(pool)
        for (op, a, b, r) in st:
            if r in known_now:
                continue
            assert a in known_now and b in known_now, 'operand evicted'
            steps.append((op, a, b, r))
            known_now.add(r)
            pool.append(r)
            created[r] = rnd
            last_use[a] = rnd
            last_use[b] = rnd
            last_use[r] = rnd
            self.uses[a] = self.uses.get(a, 0) + 1
            self.uses[b] = self.uses.get(b, 0) + 1
            uncovered.discard(r)
        if len(pool) > cfg.pool_cap + cfg.evict_margin:
            self.evict(pool, last_use, created, cfg.pool_cap, protect=set())

    def pool_order(self, pool, last_use):
        # small values first (used for L2 o L0 extension), then by recency
        return sorted(pool, key=lambda v: (v > self.cfg.keep_small, -last_use.get(v, 0), v))

    def evict(self, pool, last_use, created, cap, protect):
        if len(pool) <= cap:
            return
        cand = [v for v in pool if v > self.cfg.keep_small and v not in protect]
        if self.cfg.evict_size:
            cand.sort(key=lambda v: (-v.bit_length(), last_use.get(v, 0), created.get(v, 0)))
        elif self.cfg.evict_unused:
            cand.sort(key=lambda v: (self.uses.get(v, 0) > 0, last_use.get(v, 0), created.get(v, 0)))
        elif self.cfg.evict_big > 0:
            bb = min(self.p.targets).bit_length() - self.cfg.evict_big
            cand.sort(key=lambda v: (v.bit_length() < bb, last_use.get(v, 0), created.get(v, 0)))
        else:
            cand.sort(key=lambda v: (last_use.get(v, 0), created.get(v, 0)))
        k = len(pool) - cap
        drop = set(cand[:k])
        if drop:
            self.evicted = True
        pool[:] = [v for v in pool if v not in drop]


def solve(p, cfg):
    """Run the solver and return a validated-shape register program (or None)."""
    s = Solver(p, cfg)
    st = s.run()
    simulate(p.sv, st, p.limit)
    prog = regalloc(p.sv, st, p.targets, p.nreg)
    return prog, s.stats

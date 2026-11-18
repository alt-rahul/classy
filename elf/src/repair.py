"""Ruin-and-recreate post-pass on a finished SSA program.

For each target t: remove the steps that exist only to build t (its private subtree D(t)),
then look for a cheaper derivation of t from ALL other values of the program (computed
anywhere, starts included). New steps are inserted right after their operands become
available. Accepted only if strictly shorter and register allocation still fits.
"""
import engine
self_thr = [64]
from solver import regalloc


def _ssa(sv, steps):
    """Return (vals, defs) with ids; ids < len(sv_unique) are starts."""
    cur = {}
    vals = []
    for v in sv:
        if v not in cur:
            cur[v] = len(vals)
            vals.append(v)
    nstart = len(vals)
    defs = []
    for (op, a, b, r) in steps:
        ia, ib = cur[a], cur[b]
        rid = len(vals)
        vals.append(r)
        cur[r] = rid
        defs.append((op, ia, ib, rid))
    return vals, defs, nstart


def dce(sv, steps, targets):
    vals, defs, nstart = _ssa(sv, steps)
    tset = set(targets) - set(sv)
    need = [False] * len(defs)
    seen = set()
    for k, (op, ia, ib, ir) in enumerate(defs):
        v = vals[ir]
        if v in tset and v not in seen:
            seen.add(v)
            need[k] = True
    pos = {d[3]: k for k, d in enumerate(defs)}
    for k in range(len(defs) - 1, -1, -1):
        if need[k]:
            for x in defs[k][1:3]:
                if x in pos:
                    need[pos[x]] = True
    return [steps[k] for k in range(len(steps)) if need[k]]


def private_subtree(defs, k, protected):
    """Steps (indices) used only to build defs[k] (k included). protected: step indices that
    must not be removed (first producers of other targets)."""
    pos = {d[3]: i for i, d in enumerate(defs)}
    uses = {}
    for i, (op, ia, ib, ir) in enumerate(defs):
        for x in (ia, ib):
            uses.setdefault(x, []).append(i)
    D = {k}
    stack = [k]
    while stack:
        i = stack.pop()
        for x in defs[i][1:3]:
            j = pos.get(x)
            if j is None or j in D or j in protected:
                continue
            if all(u in D for u in uses.get(x, [])):
                D.add(j)
                stack.append(j)
    # a second pass: removing more members can make further steps private
    changed = True
    while changed:
        changed = False
        for j in range(len(defs)):
            if j in D or j in protected:
                continue
            x = defs[j][3]
            us = uses.get(x, [])
            if us and all(u in D for u in us):
                D.add(j)
                changed = True
    return D


def improve(p, steps, maxlev_small=2, passes=2, verbose=False):
    """Returns improved steps (by value), never longer than the DCE'd input."""
    steps = dce(p.sv, steps, p.targets)
    tset = set(p.targets) - set(p.sv)
    for _ in range(passes):
        improved = False
        order = [s[3] for s in steps if s[3] in tset]
        for t in order:
            vals, defs, nstart = _ssa(p.sv, steps)
            first = {}
            for i, (op, ia, ib, ir) in enumerate(defs):
                v = vals[ir]
                if v in tset and v not in first:
                    first[v] = i
            k = first[t]
            protected = set(first.values()) - {k}
            D = private_subtree(defs, k, protected)
            if len(D) <= 1:
                continue
            # t must be rebuilt before its first consumer outside D
            tid = defs[k][3]
            fu = len(defs)
            for i, d in enumerate(defs):
                if i not in D and (d[1] == tid or d[2] == tid):
                    fu = i
                    break
            keep_vals = []
            seenv = set()
            for i in range(nstart):
                if vals[i] not in seenv:
                    seenv.add(vals[i]); keep_vals.append(vals[i])
            for i, d in enumerate(defs):
                if i >= fu:
                    break
                if i not in D:
                    v = vals[d[3]]
                    if v not in seenv and v != t:
                        seenv.add(v); keep_vals.append(v)
            ml = maxlev_small if len(keep_vals) <= self_thr[0] else 1
            engine.build(keep_vals, p.limit, ml, 0)
            r = engine.find_pair(t, len(D) - 1)
            if r is None:
                continue
            c, plan = r
            # rebuild step list: remove D, insert plan steps after operands are available
            base = [steps[i] for i in range(len(steps)) if i not in D]
            avail_pos = {}
            for v in p.sv:
                avail_pos.setdefault(v, -1)
            for i, s in enumerate(base):
                avail_pos.setdefault(s[3], i)
            inserts = []  # (position, order, step)
            for j, s in enumerate(plan):
                op, a, b, rr = s
                if rr in avail_pos and avail_pos[rr] < 10**9 and rr != t:
                    continue
                pa = avail_pos.get(a)
                pb = avail_pos.get(b)
                if pa is None or pb is None:
                    inserts = None
                    break
                at = max(pa, pb)
                inserts.append((at, j, s))
                avail_pos[rr] = at
            if inserts is None:
                continue
            new = []
            ins_by = {}
            for (at, j, s) in inserts:
                ins_by.setdefault(at, []).append((j, s))
            for (j, s) in sorted(ins_by.get(-1, [])):
                new.append(s)
            for i, s in enumerate(base):
                new.append(s)
                for (j, s2) in sorted(ins_by.get(i, [])):
                    new.append(s2)
            if len(new) >= len(steps):
                continue
            known = set(p.sv)
            ok = True
            for (op, a, b, rr) in new:
                if a not in known or b not in known:
                    ok = False
                    break
                known.add(rr)
            if not ok:
                continue
            prog = regalloc(p.sv, new, p.targets, p.nreg)
            if prog is None:
                continue
            steps = dce(p.sv, new, p.targets)
            improved = True
            if verbose:
                print('repair', t, 'saved', len(D) - c)
        if not improved:
            break
    return steps

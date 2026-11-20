"""Joint-cone group rebuild: for pairs of targets whose joint exclusive cone (values used only by the two) has c ops,
rebuild both independently from the remaining DAG values with wide2 so that the total is < c.
usage: jrun.py TAG TL WORKERS idsfile1,idsfile2,..."""
import sys, time, json, os, itertools
from multiprocessing import Pool
from wlib import *
from jcone import joint_cone

def desc_of(users, roots):
    d = set(); st = list(roots)
    while st:
        x = st.pop()
        for w in users.get(x, ()):
            if w not in d: d.add(w); st.append(w)
    return d

def work(a):
    pid, tl = a
    P = {p.id: p for p in load_puzzles()}; p = P[pid]
    st = load_store(); prog = st[pid]['prog']; L0 = len(prog)
    log = []; t00 = time.time(); tried = set()
    bits = p.limit.bit_length()
    while True:
        nodes, order = build_dag(p, prog)
        sv = set(p.sv); T = set(p.targets) - sv
        users = collections.defaultdict(set)
        for v, (op, a_, b_) in nodes.items(): users[a_].add(v); users[b_].add(v)
        single = {t: len(joint_cone(nodes, sv, T, users, [t])) for t in T}
        thr = 7 if bits >= 56 else 6 if bits >= 46 else 5
        cmin1 = thr - 1
        pairs = []
        for i, j in itertools.combinations(sorted(T), 2):
            if (i, j) in tried: continue
            if single[i] + single[j] < 2 * thr - 3: continue
            jc = joint_cone(nodes, sv, T, users, [i, j])
            if len(jc) >= 2 * thr + 1 and len(jc) > single[i] + single[j]: pairs.append((len(jc), i, j, jc))
        if not pairs: break
        pairs.sort(key=lambda x: -x[0])
        c, i, j, jc = pairs[0]; tried.add((i, j))
        dsc = desc_of(users, jc)
        pool = [v for v in list(dict.fromkeys(p.sv)) + order if v not in jc and v not in dsc]
        t0 = time.time(); res = 'none'
        si, _ = run_wide_enum(p.limit, i, pool, cmin1, c - 1 - cmin1, 1, tl, 6)
        if si:
            b = min(len(s) for s in si)
            sj, _ = run_wide_enum(p.limit, j, pool, cmin1, c - 1 - b, 1, tl, 6)
            if sj:
                res = 'noemit'; ntry = 0
                combos = sorted(((len(set(x[3] for x in a1) | set(x[3] for x in a2)), k1, k2) for k1, a1 in enumerate(si) for k2, a2 in enumerate(sj)))
                for tot, k1, k2 in combos:
                    if tot >= c: break
                    nn, cands = splice_blocks(p, nodes, order, jc, [si[k1], sj[k2]])
                    if len(nn) >= len(nodes): continue
                    pr = try_emit(p, nn, cands)
                    if pr is None and ntry < 3:
                        ntry += 1
                        pr = try_emit5(p, nn, cands, 40.0)
                    if pr is not None and len(pr) < len(prog):
                        prog = pr; res = 'ok'
                        st = load_store()
                        if offer(st, p, prog, 'wide-joint'): save_store(st)
                        break
        log.append((c, single[i], single[j], res, round(time.time() - t0, 1)))
        if time.time() - t00 > 1500: break
    return pid, L0, len(prog), log, round(time.time() - t00, 1), prog

if __name__ == '__main__':
    tag, tl, nw = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    ids = []
    for f in sys.argv[4].split(','): ids += [x.strip() for x in open(os.path.join(HERE, f)).read().split(',') if x.strip()]
    RUN = os.path.join(HERE, 'runs_%s.jsonl' % tag)
    done = set(json.loads(l)[0] for l in open(RUN)) if os.path.exists(RUN) else set()
    ids = [i for i in ids if i not in done]
    with Pool(nw) as pool:
        for r in pool.imap_unordered(work, [(i, tl) for i in ids]):
            open(RUN, 'a').write(json.dumps(r[:5]) + '\n')
            if r[2] < r[1]: open(os.path.join(HERE, 'improved_%s.jsonl' % tag), 'a').write(json.dumps([r[0], r[5]]) + '\n')
            if r[3]: print(r[0], r[1], '->', r[2], r[3], r[4], 's', flush=True)

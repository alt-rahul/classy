"""Resumable runner for the seeding variants of src/seed.py (plain, c, h, ch); offers the best valid
program of each puzzle to the store (scored set). Per-puzzle variant lengths go to store/runs_<tag>.json.
usage: seed_run.py --tag sd1 --cfg '{"pool_cap":46,"evict_size":1,"regret":0.1,"peel":1}'
                   [--which practice] [--ids ...] [--variants c,h,ch] [--shared-only] [--workers 2] [--repair]"""
import sys, os, json, time, argparse, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def work(args):
    pid, cfgd, which, vlist, rep = args
    import seed, solver
    P = {p.id: p for p in load_puzzles(which)}
    p = P[pid]
    t0 = time.time()
    out = {}
    try:
        res = {}
        if 'plain' in vlist:
            prog, _ = solver.solve(p, solver.Cfg(**cfgd))
            res['plain'] = prog
        sv = [v for v in vlist if v != 'plain']
        nc = nh = 0
        if sv:
            r2, consts, hubs = seed.variants(p, cfgd, sv)
            res.update(r2); nc, nh = len(consts), len(hubs)
        if rep:
            import repair
            from repair_store import prog_to_steps
            repair.self_thr[0] = 150
            for v, prog in list(res.items()):
                if prog is None:
                    continue
                st2 = repair.improve(p, prog_to_steps(p, prog), passes=3)
                p2 = solver.regalloc(p.sv, st2, p.targets, p.nreg)
                if p2 is not None and len(p2) < len(prog) and check(p, p2) is None:
                    res[v] = p2
        best = None
        for v, prog in res.items():
            if prog is not None and check(p, prog) is None:
                out[v] = len(prog)
                if best is None or len(prog) < len(best[1]):
                    best = (v, prog)
            else:
                out[v] = None
        out['nconst'] = nc; out['nhub'] = nh
        return pid, best, out, time.time() - t0
    except Exception:
        return pid, None, {'error': traceback.format_exc()[-800:]}, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--cfg', default='{"pool_cap":46,"evict_size":1,"regret":0.1,"peel":1}')
    ap.add_argument('--which', default='scored')
    ap.add_argument('--ids', default='')
    ap.add_argument('--classes', default='')
    ap.add_argument('--variants', default='c,h,ch')
    ap.add_argument('--shared-only', action='store_true')
    ap.add_argument('--minbits', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--repair', action='store_true')
    ap.add_argument('--order', default='bits', help='bits: largest max-target first; id')
    a = ap.parse_args()
    cfgd = json.loads(a.cfg)
    vlist = a.variants.split(',')
    import seed
    puzzles = load_puzzles(a.which)
    P = {p.id: p for p in puzzles}
    RUNS = os.path.join(ROOT, 'store', 'runs_%s.json' % a.tag)
    done = json.load(open(RUNS)) if os.path.exists(RUNS) else {}
    ids = [p.id for p in puzzles if p.limit > 20]
    if a.ids:
        want = a.ids.split(',')
        ids = [i for i in want if i in P]
    if a.classes:
        cl = set(a.classes.split(','))
        ids = [i for i in ids if pclass(P[i]) in cl]
    ids = [i for i in ids if i not in done and max(P[i].targets).bit_length() >= a.minbits]
    if a.shared_only:
        ids = [i for i in ids if seed.shared_like(P[i])]
    if a.order == 'bits':
        ids.sort(key=lambda i: -max(P[i].targets).bit_length())
    st = load_store() if a.which == 'scored' else {}
    print('seed_run', a.tag, 'todo', len(ids), 'variants', vlist, 'cfg', cfgd, flush=True)
    t0 = time.time(); gain = 0; nimp = 0
    with Pool(a.workers) as pool:
        for k, (pid, best, out, dt) in enumerate(pool.imap_unordered(work, [(i, cfgd, a.which, vlist, a.repair) for i in ids])):
            p = P[pid]
            old = st[pid]['len'] if pid in st else None
            ok = False
            if best is not None and a.which == 'scored':
                ok = offer(st, p, best[1], a.tag + '_' + best[0])
                if ok:
                    gain += old - len(best[1]); nimp += 1
                    save_store(st)
            out['store_before'] = old; out['t'] = round(dt, 1)
            done[pid] = out
            json.dump(done, open(RUNS + '.tmp', 'w')); os.replace(RUNS + '.tmp', RUNS)
            print('%d/%d %s %s bits %d store %s %s %s %.0fs  gain %d improved %d elapsed %.0fs' % (
                k + 1, len(ids), pid, pclass(p), max(p.targets).bit_length(), old, json.dumps(out),
                'stored' if ok else '', dt, gain, nimp, time.time() - t0), flush=True)
    print('seed_run done', a.tag, 'improved', nimp, 'gain', gain, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()

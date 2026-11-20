"""Multi-start driver (round 2, items 3/4): for each puzzle run a list of configs (e.g. random target
orders via perm_seed, regret alphas, seeds) and keep the best validated program in the store.
Resumable through store/runs_<tag>.json (puzzle -> best length found under this tag).
usage: restarts.py --tag T --classes c1,c2 --base '{cfg}' --grid '[{override}, ...]' [--budget s]"""
import sys, os, json, time, argparse, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def work(args):
    pid, base, grid, budget, which, rep = args
    import solver
    P = {p.id: p for p in load_puzzles(which)}
    p = P[pid]
    t0 = time.time()
    best = None
    n = 0
    try:
        for ov in grid:
            if time.time() - t0 > budget:
                break
            cfgd = dict(base); cfgd.update(ov)
            bkw = cfgd.pop('beam', None)
            if bkw is not None:
                import beam
                prog = beam.beam_solve(p, solver.Cfg(**cfgd), **bkw)
            else:
                prog, stats = solver.solve(p, solver.Cfg(**cfgd))
            n += 1
            if prog is None:
                continue
            if rep:
                import repair
                from repair_store import prog_to_steps
                repair.self_thr[0] = 150
                st2 = repair.improve(p, prog_to_steps(p, prog), passes=3)
                p2 = solver.regalloc(p.sv, st2, p.targets, p.nreg)
                if p2 is not None and len(p2) < len(prog):
                    prog = p2
            if check(p, prog) is not None:
                continue
            if best is None or len(prog) < len(best[0]):
                best = (prog, ov)
    except Exception:
        return pid, None, {'error': traceback.format_exc()[-800:]}, time.time() - t0, n
    return pid, best, {}, time.time() - t0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--base', default='{}')
    ap.add_argument('--grid', default='[{}]')
    ap.add_argument('--classes', default='')
    ap.add_argument('--ids', default='')
    ap.add_argument('--which', default='scored')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--budget', type=float, default=1e9, help='seconds per puzzle')
    ap.add_argument('--order', default='len')
    ap.add_argument('--repair', action='store_true')
    a = ap.parse_args()
    base = json.loads(a.base)
    grid = json.loads(a.grid)
    puzzles = load_puzzles(a.which)
    P = {p.id: p for p in puzzles}
    RUNS = os.path.join(ROOT, 'store', 'runs_%s.json' % a.tag)
    done = json.load(open(RUNS)) if os.path.exists(RUNS) else {}
    ids = [p.id for p in puzzles if p.limit > 20]
    if a.ids:
        want = set(a.ids.split(',')); ids = [i for i in ids if i in want]
    if a.classes:
        cl = set(a.classes.split(',')); ids = [i for i in ids if pclass(P[i]) in cl]
    ids = [i for i in ids if i not in done]
    st = load_store() if a.which == 'scored' else {}
    if a.order == 'len' and st:
        ids.sort(key=lambda i: -st[i]['len'])
    print('tag', a.tag, 'todo', len(ids), 'grid', len(grid), flush=True)
    t0 = time.time(); nimp = 0; gain = 0; last = time.time()
    with Pool(a.workers) as pool:
        for k, (pid, best, info, dt, n) in enumerate(pool.imap_unordered(
                work, [(i, base, grid, a.budget, a.which, a.repair) for i in ids])):
            p = P[pid]
            if best is None:
                print('FAIL', pid, info, flush=True); done[pid] = None; continue
            prog, ov = best
            old = st[pid]['len'] if pid in st else None
            if a.which == 'scored' and offer(st, p, prog, a.tag):
                nimp += 1; gain += old - len(prog)
            done[pid] = len(prog)
            print('%d/%d %s len %d old %s runs %d best %s %.0fs elapsed %.0fs improved %d gain %d' % (
                k + 1, len(ids), pid, len(prog), old, n, json.dumps(ov), dt, time.time() - t0, nimp, gain), flush=True)
            if time.time() - last > 60:
                if a.which == 'scored':
                    save_store(st)
                json.dump(done, open(RUNS + '.tmp', 'w')); os.replace(RUNS + '.tmp', RUNS)
                last = time.time()
    if a.which == 'scored':
        save_store(st)
    json.dump(done, open(RUNS + '.tmp', 'w')); os.replace(RUNS + '.tmp', RUNS)
    print('done', a.tag, 'improved', nimp, 'gain', gain, '%.0fs' % (time.time() - t0))
    if a.which == 'scored':
        print_breakdown(st, puzzles)


if __name__ == '__main__':
    main()

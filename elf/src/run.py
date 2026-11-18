"""Parallel, resumable runner: solve puzzles with a config, keep best validated programs in the store."""
import sys, os, json, time, argparse, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def runs_path(tag):
    return os.path.join(ROOT, 'store', 'runs_%s.json' % tag)


def work(args):
    pid, cfgd, which = args
    import solver, tiny
    P = {p.id: p for p in load_puzzles(which)}
    p = P[pid]
    t0 = time.time()
    try:
        if p.limit <= 20:
            st = tiny.solve_tiny(p.sv, p.targets, p.limit)
            prog = solver.regalloc(p.sv, st, p.targets, p.nreg)
            stats = {'tiny': 1}
        else:
            cfg = solver.Cfg(**cfgd)
            prog, stats = solver.solve(p, cfg)
            if prog is None and cfg.pool_cap > 30:
                cfg.pool_cap -= 6
                prog, stats = solver.solve(p, cfg)
    except Exception as e:
        return pid, None, {'error': traceback.format_exc()[-800:]}, time.time() - t0
    return pid, prog, stats, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--which', default='scored')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ids', default='')
    ap.add_argument('--classes', default='')
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--skip', default='', help='comma list of tags whose done puzzles are skipped')
    a = ap.parse_args()
    cfgd = json.loads(a.cfg)
    puzzles = load_puzzles(a.which)
    P = {p.id: p for p in puzzles}
    RUNS = runs_path(a.tag)
    done = json.load(open(RUNS)) if os.path.exists(RUNS) else {}
    runs = done
    ids = [p.id for p in puzzles]
    if a.ids:
        want = set(a.ids.split(','))
        ids = [i for i in ids if i in want]
    if a.classes:
        cl = set(a.classes.split(','))
        ids = [i for i in ids if pclass(P[i]) in cl]
    if not a.redo:
        ids = [i for i in ids if i not in done]
    for tg in a.skip.split(','):
        if tg and os.path.exists(runs_path(tg)):
            d2 = json.load(open(runs_path(tg)))
            ids = [i for i in ids if i not in d2]
    st = load_store() if a.which == 'scored' else {}
    print('tag', a.tag, 'todo', len(ids), flush=True)
    t0 = time.time()
    nimp = 0
    gain = 0
    last_save = time.time()
    with Pool(a.workers) as pool:
        for k, (pid, prog, stats, dt) in enumerate(pool.imap_unordered(work, [(i, cfgd, a.which) for i in ids])):
            p = P[pid]
            if prog is None:
                print('FAIL', pid, stats, flush=True)
                done[pid] = None
                continue
            err = check(p, prog)
            if err:
                print('INVALID', pid, err, flush=True)
                done[pid] = None
                continue
            old = st[pid]['len'] if pid in st else None
            if a.which == 'scored' and offer(st, p, prog, a.tag):
                nimp += 1
                gain += (old - len(prog)) if old else 0
            done[pid] = len(prog)
            if (k + 1) % 10 == 0 or time.time() - last_save > 60:
                print('%d/%d %s len %d old %s %.1fs elapsed %.0fs improved %d gain %d' % (
                    k + 1, len(ids), pid, len(prog), old, dt, time.time() - t0, nimp, gain), flush=True)
            if time.time() - last_save > 60:
                if a.which == 'scored':
                    save_store(st)
                json.dump(runs, open(RUNS + '.tmp', 'w'))
                os.replace(RUNS + '.tmp', RUNS)
                last_save = time.time()
    if a.which == 'scored':
        save_store(st)
    json.dump(runs, open(RUNS + '.tmp', 'w'))
    os.replace(RUNS + '.tmp', RUNS)
    print('done', a.tag, 'improved', nimp, 'gain', gain, '%.0fs' % (time.time() - t0))
    if a.which == 'scored':
        print_breakdown(st, puzzles)


if __name__ == '__main__':
    main()

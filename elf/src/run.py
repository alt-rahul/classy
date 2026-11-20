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
    if '_class' in cfgd:  # per-class overrides
        cfgd = dict(cfgd)
        over = cfgd.pop('_class')
        cfgd.update(over.get(pclass(p), {}))
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
        if prog is not None and REPAIR[0] and p.limit > 20:
            import repair
            from repair_store import prog_to_steps
            repair.self_thr[0] = 150
            st2 = repair.improve(p, prog_to_steps(p, prog), passes=3)
            p2 = solver.regalloc(p.sv, st2, p.targets, p.nreg)
            if p2 is not None and len(p2) < len(prog) and check(p, p2) is None:
                stats['rep'] = len(prog) - len(p2)
                prog = p2
    except Exception as e:
        return pid, None, {'error': traceback.format_exc()[-800:]}, time.time() - t0
    return pid, prog, stats, time.time() - t0


REPAIR = [0]


def _init(rep):
    REPAIR[0] = rep


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
    ap.add_argument('--order', default='id', help="id | len (longest stored program first) | opt (highest ops/target first) | optasc (lowest first) | bits (largest target first)")
    ap.add_argument('--repair', action='store_true', help='apply the repair post-pass to each result')
    ap.add_argument('--top', type=int, default=0, help='keep only the first N puzzles after ordering')
    ap.add_argument('--nstarts', default='', help='comma list of allowed numbers of distinct start values')
    ap.add_argument('--minbits', type=int, default=0, help='only puzzles whose largest target has >= this many bits')
    ap.add_argument('--maxbits', type=int, default=99)
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
    if a.nstarts:
        ns = set(int(x) for x in a.nstarts.split(','))
        ids = [i for i in ids if len(set(P[i].sv)) in ns]
    ids = [i for i in ids if a.minbits <= max(P[i].targets).bit_length() <= a.maxbits]
    if not a.redo:
        ids = [i for i in ids if i not in done]
    for tg in a.skip.split(','):
        if tg and os.path.exists(runs_path(tg)):
            d2 = json.load(open(runs_path(tg)))
            ids = [i for i in ids if i not in d2]
    st = load_store() if a.which == 'scored' else {}
    if a.order == 'len' and st:
        ids.sort(key=lambda i: -st[i]['len'])
    elif a.order == 'opt' and st:
        ids.sort(key=lambda i: -st[i]['len'] / len(P[i].targets))
    elif a.order == 'bits':
        ids.sort(key=lambda i: (-max(P[i].targets).bit_length(), i))
    elif a.order == 'optasc' and st:
        ids.sort(key=lambda i: st[i]['len'] / len(P[i].targets))
    if a.top:
        ids = ids[:a.top]
    print('tag', a.tag, 'todo', len(ids), flush=True)
    t0 = time.time()
    nimp = 0
    gain = 0
    last_save = time.time()
    with Pool(a.workers, initializer=_init, initargs=(int(a.repair),)) as pool:
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
            if (k + 1) % 10 == 0 or time.time() - last_save > 15:
                print('%d/%d %s len %d old %s %.1fs elapsed %.0fs improved %d gain %d' % (
                    k + 1, len(ids), pid, len(prog), old, dt, time.time() - t0, nimp, gain), flush=True)
            if time.time() - last_save > 15:
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

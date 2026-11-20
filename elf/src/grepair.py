"""Global repair (round 3 survivor S2): repair.improve with the history threshold lifted, so every
target is re-derived with L2 over ALL values of the stored program. Resumable: per-puzzle results go
to store/runs_<tag>.json; improvements are offered to the store (validated, strictly shorter only).
usage: grepair.py --tag r3S2 --classes 'indep>=20' --maxbits 44 [--skip-tags r3C] [--workers 1]"""
import sys, os, json, time, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def work(args):
    pid, prog, thr, passes = args
    import repair, solver
    from repair_store import prog_to_steps
    repair.self_thr[0] = thr
    P = {p.id: p for p in load_puzzles('scored')}
    p = P[pid]
    t0 = time.time()
    steps = prog_to_steps(p, prog)
    st2 = repair.improve(p, steps, passes=passes)
    newp = solver.regalloc(p.sv, st2, p.targets, p.nreg)
    return pid, newp, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r3S2')
    ap.add_argument('--classes', default='indep>=20')
    ap.add_argument('--maxbits', type=int, default=44)
    ap.add_argument('--minbits', type=int, default=0)
    ap.add_argument('--skip-tags', default='', help='skip puzzles whose stored tag starts with one of these')
    ap.add_argument('--ids', default='')
    ap.add_argument('--thr', type=int, default=10**9)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    puzzles = load_puzzles('scored')
    P = {p.id: p for p in puzzles}
    st = load_store()
    RUNS = os.path.join(ROOT, 'store', 'runs_%s.json' % a.tag)
    done = json.load(open(RUNS)) if os.path.exists(RUNS) else {}
    cl = set(a.classes.split(','))
    skt = [t for t in a.skip_tags.split(',') if t]
    ids = [p.id for p in puzzles if pclass(p) in cl and a.minbits <= max(p.targets).bit_length() <= a.maxbits
           and p.limit > 20 and p.id not in done and not any(st[p.id]['tag'].startswith(t) for t in skt)]
    if a.ids:
        want = set(a.ids.split(','))
        ids = [i for i in ids if i in want]
    print('grepair', a.tag, 'todo', len(ids), 'ops', sum(st[i]['len'] for i in ids), flush=True)
    t0 = time.time(); gain = 0; nimp = 0
    with Pool(a.workers) as pool:
        for k, (pid, prog, dt) in enumerate(pool.imap_unordered(work, [(i, st[i]['prog'], a.thr, a.passes) for i in ids])):
            p = P[pid]
            old = st[pid]['len']
            ok = prog is not None and offer(st, p, prog, a.tag)
            if ok:
                gain += old - len(prog); nimp += 1
                save_store(st)
            done[pid] = [old, len(prog) if prog else None, round(dt, 1)]
            json.dump(done, open(RUNS + '.tmp', 'w')); os.replace(RUNS + '.tmp', RUNS)
            print('%d/%d %s %s %d -> %s %s %.0fs  total gain %d (%.0f ops/CPU-h)' % (
                k + 1, len(ids), pid, pclass(p), old, len(prog) if prog else None, 'stored' if ok else '',
                dt, gain, gain / max(time.time() - t0, 1) * 3600 / a.workers), flush=True)
    print('grepair done', a.tag, 'improved', nimp, 'gain', gain, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()

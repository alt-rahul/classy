"""Apply the ruin-and-recreate post-pass (repair.py) to programs in the store, in parallel."""
import sys, os, json, time, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def prog_to_steps(p, prog):
    reg = [None] * p.nreg
    for i, v in enumerate(p.sv):
        reg[i] = v
    steps = []
    for op, i, j, d in prog:
        a, b = reg[i], reg[j]
        r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b if b else 0}[op]
        steps.append((op, a, b, r))
        reg[d] = r
    return steps


def work(args):
    pid, prog, thr, passes = args
    import repair, solver
    repair.self_thr[0] = thr
    P = {p.id: p for p in load_puzzles('scored')}
    p = P[pid]
    t0 = time.time()
    if p.limit <= 20 or not prog:
        return pid, None, 0
    steps = prog_to_steps(p, prog)
    st2 = repair.improve(p, steps, passes=passes)
    newp = solver.regalloc(p.sv, st2, p.targets, p.nreg)
    return pid, newp, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=int, default=150)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ids', default='')
    a = ap.parse_args()
    puzzles = load_puzzles('scored')
    P = {p.id: p for p in puzzles}
    st = load_store()
    ids = [p.id for p in puzzles if st[p.id]['tag'] != 'sample']
    if a.ids:
        want = set(a.ids.split(','))
        ids = [i for i in ids if i in want]
    print('repair todo', len(ids), flush=True)
    t0 = time.time(); gain = 0; nimp = 0; last = time.time()
    with Pool(a.workers) as pool:
        for k, (pid, prog, dt) in enumerate(pool.imap_unordered(work, [(i, st[i]['prog'], a.thr, a.passes) for i in ids])):
            if prog is None:
                continue
            old = st[pid]['len']
            if offer(st, P[pid], prog, st[pid]['tag'] + '+rep'):
                gain += old - len(prog); nimp += 1
            if time.time() - last > 60:
                save_store(st); last = time.time()
                print('%d/%d gain %d improved %d %.0fs' % (k + 1, len(ids), gain, nimp, time.time() - t0), flush=True)
    save_store(st)
    print('repair done gain', gain, 'improved', nimp, '%.0fs' % (time.time() - t0))
    print_breakdown(st, puzzles)


if __name__ == '__main__':
    main()

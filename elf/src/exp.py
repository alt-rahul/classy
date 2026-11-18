"""Config experiment on scored puzzles: run a cfg on given ids sequentially, compare with the
store, offer improvements (validated, merge-safe). usage: exp.py <ids> '<cfg json>' <tag>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import solver
ids = sys.argv[1].split(',')
cfgd = json.loads(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else 'exp'
P = {p.id: p for p in load_puzzles('scored')}
st = load_store()
tot_new = tot_old = 0
T0 = time.time()
for pid in ids:
    p = P[pid]
    t0 = time.time()
    prog, stats = solver.solve(p, solver.Cfg(**cfgd))
    dt = time.time() - t0
    old = st[pid]['len']
    n = len(prog) if prog else 10**9
    tot_new += min(n, 10**6); tot_old += old
    imp = prog is not None and offer(st, p, prog, tag)
    print(pid, pclass(p), 'new', n, 'store', old, 'imp' if imp else '', '%.1fs' % dt, flush=True)
save_store(st)
print('TOTAL new', tot_new, 'store', tot_old, '%.0fs' % (time.time() - T0), cfgd)

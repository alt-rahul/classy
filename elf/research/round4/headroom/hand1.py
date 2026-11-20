import sys
from hand import *
p = P['practice_0015']; T = p.targets
G, B2, Y, h2 = 522040, 85327960040, 148686388720, 783790856000
P0, H9, X3, H23 = 81121293120, 1127100976310, 122702665707, 23695877541009
hubs = [B2, Y, h2, P0, H9, X3, H23, T[5]]
name = sys.argv[1]; seedsA = hubs + ([G] if 'G' in name else [])
base = dict(pool_cap=46, evict_size=1, regret=0.1, sig=0.0, ajit=0.0)
a = copy.copy(p); a.targets = sorted(set(seedsA))
sa = PEEL(a, Cfg(**base)).run()
print('stage A', len(sa), flush=True)
B = Builder(p, sa)
left = finish(B, T, hubs)
print('after finish', len(B.steps), 'left', [T.index(t) for t in left], flush=True)
if left:
    c = copy.copy(p)
    sm = sorted(B.smalls(1 << 20))[:30]
    c.sv = list(dict.fromkeys(list(p.sv) + sm + hubs))[:44]; c.targets = left
    sc = PEEL(c, Cfg(**base)).run()
    B.add(sc)
prog = regalloc(p.sv, B.steps, p.targets, p.nreg)
print('TOTAL', len(prog) if prog else None, check(p, prog) if prog else 'regalloc', flush=True)
pickle.dump((prog, B.steps), open('hand1_%s.pkl' % name, 'wb'))

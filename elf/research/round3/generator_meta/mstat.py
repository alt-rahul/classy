import sys,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from bases import links
from fam import fam
C=collections.Counter(); Cm=collections.Counter(); per=collections.defaultdict(set)
for w in ('practice','scored'):
  for p in load_puzzles(w):
    f=fam(p)
    if f in('tiny','indep') or len(p.targets)<20: continue
    for j,i,mj,mi,r,H in links(p.targets,maxm=3000):
        if H<10**8: continue
        for m in (mi,mj):
            if m>1: C[m]+=1; per[m].add(p.id)
print('multiplier atoms by number of puzzles:')
print(sorted(((len(per[m]),m) for m in per),reverse=True)[:120])

import sys
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from nt import factor
which=sys.argv[1]; ids=sys.argv[2:]
for p in load_puzzles(which):
    if p.id[-4:] in ids:
        print(p.id,p.family,p.sv,'L=',p.limit,factor(p.limit))
        for t in p.targets:
            f=factor(t); print('  %20d %2d  %s'%(t,t.bit_length(),' * '.join('%d^%d'%(a,b) if b>1 else str(a) for a,b in sorted(f.items()))))

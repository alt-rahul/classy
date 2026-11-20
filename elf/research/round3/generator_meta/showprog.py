import sys
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from nt import factor
from attrib import marginal
from fam import decade
pid=sys.argv[1]; st=load_store()
for p in load_puzzles('scored'):
    if p.id[-4:]==pid:
        res,_=marginal(p,st[p.id]['prog']); k,r=decade(p)
        print(p.id,p.sv,p.limit,st[p.id]['len'])
        for t in p.targets:
            f=factor(t); print('  %20d %2d %s c=%d  %s'%(t,t.bit_length(),'R' if t in r else ' ',res[t],' * '.join('%d^%d'%(a,b) if b>1 else str(a) for a,b in sorted(f.items()))))

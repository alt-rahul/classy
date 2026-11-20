import sys,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
for p in load_puzzles('practice'):
    if p.family in('mixed','shared','multiplicative') and len(p.targets)>=20:
        dg=collections.Counter(len(str(t)) for t in p.targets)
        print(p.id[-4:],p.family[:5],len(p.sv),len(p.targets),'Ldig',len(str(p.limit)),'%.2f'%(max(p.targets)/p.limit),' '.join('%d:%d'%(k,v) for k,v in sorted(dg.items())))

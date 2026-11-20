import sys, time
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *; import solver
st=load_store(); P={p.id:p for p in load_puzzles('scored')}
for pid in sys.argv[1:]:
    p=P[pid]; t0=time.time(); r=solver.solve(p, solver.Cfg())
    prog=r[0] if isinstance(r,tuple) else r
    print(pid,'store',st[pid]['len'],'solver',len(prog) if prog else None, check(p,prog) if prog else '', '%.0fs'%(time.time()-t0),flush=True)

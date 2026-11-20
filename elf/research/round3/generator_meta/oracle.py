import sys,json,time
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from solver import solve,Cfg
pid=sys.argv[1]; extra=[int(x) for x in sys.argv[2].split(',') if x]
kw={"pool_cap":46,"regret":0.1,"evict_size":1}
for p in load_puzzles('practice'):
    if p.id[-4:]==pid:
        p.targets=sorted(set(p.targets)|set(extra))
        t0=time.time(); prog,st=solve(p,Cfg(**kw)); assert check(p,prog) is None
        print(pid,'extra',len(extra),'ops',len(prog),'%.0fs'%(time.time()-t0),flush=True)

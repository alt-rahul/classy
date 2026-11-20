import sys, time
from wlib import *
P={p.id:p for p in load_puzzles()}; st=load_store()
pid=sys.argv[1]; p=P[pid]; nodes,order=build_dag(p,st[pid]['prog'])
print(pid,len(order),'orig pressure',pressure(p,nodes,order))
for nreg,hint in [(48,order),(48,dfs_order(p,nodes,[v for v in order if v in set(p.targets)])),(46,order),(44,order)]:
    t0=time.time(); od=cp_sched(p,nodes,hint,nreg,tl=40,workers=3)
    print('nreg',nreg,'hint pressure',pressure(p,nodes,hint),'->',None if od is None else pressure(p,nodes,od),'%.0fs'%(time.time()-t0),flush=True)

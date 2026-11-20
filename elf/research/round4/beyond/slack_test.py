import sys, time
from wlib import *
P={p.id:p for p in load_puzzles()}; st=load_store()
for pid in sys.argv[1:]:
    p=P[pid]; nodes,order=build_dag(p,st[pid]['prog'])
    res=[]
    for nreg in (46,44,42,40):
        t0=time.time(); od=c_sched(p,nodes,order,nreg,15,1,3)
        res.append((nreg, od is not None, round(time.time()-t0,1)))
        if od is None: break
    print(pid,len(order),'orig',pressure(p,nodes,order),res,flush=True)

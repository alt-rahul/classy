import sys, time
from wlib import *
P={p.id:p for p in load_puzzles()}; st=load_store()
for pid in sys.argv[1:]:
    p=P[pid]; nodes,order=build_dag(p,st[p.id]['prog'])
    T=[v for v in order if v in set(p.targets)]
    od=dfs_order(p,nodes,T)
    t0=time.time()
    (sc,od2,to)=perm_sched(p,nodes,T,40,iters=3000,thr=40)
    print(pid,'orig',pressure(p,nodes,order),'dfs',pressure(p,nodes,od),'perm-opt',sc,'%.0fs'%(time.time()-t0))

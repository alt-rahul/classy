import sys, time, random
from multiprocessing import Pool
from wlib import *
def one(a):
    pid,t,tl=a
    P={p.id:p for p in load_puzzles()}; st=load_store(); p=P[pid]
    nodes,order=build_dag(p,st[pid]['prog']); cones=all_cones(p,nodes)
    cone,desc,users=cones[t]
    pool=[v for v in list(dict.fromkeys(p.sv))+order if v not in cone and v not in desc]
    t0=time.time(); sols,err=run_wide_enum(p.limit,t,pool,5,8,1,tl,1)
    return (len(cone), min(map(len,sols)) if sols else None, round(time.time()-t0,1))
if __name__=='__main__':
    pid=sys.argv[1]; tl=float(sys.argv[2])
    P={p.id:p for p in load_puzzles()}; st=load_store(); p=P[pid]
    nodes,order=build_dag(p,st[pid]['prog'])
    T=[t for t in dict.fromkeys(p.targets) if t in nodes]
    random.Random(1).shuffle(T)
    with Pool(3) as pl: res=pl.map(one,[(pid,t,tl) for t in T[:12]])
    print(pid,'tl',tl,'(cone, found cost, secs):',res)

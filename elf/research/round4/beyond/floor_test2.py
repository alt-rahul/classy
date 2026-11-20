import sys, time, random
from wlib import *
pid=sys.argv[1]; tl=float(sys.argv[2])
P={p.id:p for p in load_puzzles()}; st=load_store(); p=P[pid]
nodes,order=build_dag(p,st[pid]['prog']); cones=all_cones(p,nodes)
T=[t for t in dict.fromkeys(p.targets) if t in nodes]
random.Random(1).shuffle(T)
for t in [T[i] for i in map(int,sys.argv[3].split(','))]:
    cone,desc,users=cones[t]
    pool=[v for v in list(dict.fromkeys(p.sv))+order if v not in cone and v not in desc]
    sols,err=run_wide_enum(p.limit,t,pool,len(cone),len(cone),1,tl,1)
    print(len(cone),[len(s) for s in sols],err.strip().replace('\n',' | '))

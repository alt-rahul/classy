import sys, time, random
from wlib import *
from gcone2 import best_groups
pid=sys.argv[1]; nmoves=int(sys.argv[2])
P={p.id:p for p in load_puzzles()}; st=load_store(); p=P[pid]; prog=st[pid]['prog']
bits=p.limit.bit_length(); thr=7 if bits>=56 else 6 if bits>=46 else 5
rng=random.Random(5); t0=time.time(); acc=0
def pot(prog):
    nodes,order=build_dag(p,prog); g=best_groups(p,nodes,order,thr,8)
    used=set(); tot=0
    for gain,jc,A in g:
        if A&used: continue
        used|=A; tot+=gain
    return tot, (g[0][0] if g else 0), len(g)
print(pid,len(prog),'potential (disjoint sum, best group gain, #groups):',pot(prog),flush=True)
for it in range(nmoves):
    nodes,order=build_dag(p,prog); cones=all_cones(p,nodes)
    cand=[t for t,c in cones.items() if len(c[0])>=thr-1]
    t=rng.choice(cand); cone,desc,users=cones[t]; k=len(cone)
    pool=[v for v in list(dict.fromkeys(p.sv))+order if v not in cone and v not in desc]
    sols,_=run_wide_enum(p.limit,t,pool,k,k,1,6,4)
    sols=[s for s in sols if set(x[3] for x in s)!=set(cone) and len(s)<=k]
    if not sols: continue
    s=rng.choice(sols)
    nn,cands=splice_blocks(p,nodes,order,cone,[s])
    if not cands or len(nn)>len(nodes): continue
    pr=try_emit5(p,nn,cands,10.0)
    if pr is None or len(pr)>len(prog): continue
    prog=pr; acc+=1
    print(it,'accepted',acc,'len',len(prog),'potential',pot(prog),'%.0fs'%(time.time()-t0),flush=True)

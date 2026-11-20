# hub gain: for each candidate node g, gain = sum_T max(0, best_T - best_T given g free) - cost(g)
import sys, time
from rlib import *
from prof import prof, P, st
pid=sys.argv[1]; K=int(sys.argv[2]); cmax=int(sys.argv[3])
p=P[pid]; L,n,rc,nodes,order,reach=prof(pid); T=set(p.targets)-set(p.sv)
core=[v for v in order if v not in T and len(reach[v])>=K]
pool=set(core)|set(p.sv)
D,R,err=enum(p.limit,sorted(pool),sorted(T),cmax,64,34)
def anc(g):
    s=set(); stk=[g]
    while stk:
        x=stk.pop()
        if x in pool or x in s: continue
        s.add(x); op,a,b=D[x]; stk+=[a,b]
    return s
best={}; withg=collections.defaultdict(dict)
for t in T:
    NS=[set(nodes_of(D,pool,t,rec).keys()) for rec in R[t]]
    best[t]=min(len(x) for x in NS)
    for ns in NS:
        for g in ns:
            if g==t or g not in D: continue
            a=anc(g); c=len(ns-a)
            if c<withg[g].get(t,99): withg[g][t]=c
gains=[]
for g,d in withg.items():
    gain=sum(max(0,best[t]-c) for t,c in d.items())-len(anc(g))
    gains.append((gain,len(anc(g)),sum(1 for t,c in d.items() if c<best[t]),g))
gains.sort(reverse=True)
print(pid,'K',K,'cmax',cmax,'sum best',sum(best.values()),'candidates',len(gains))
print('top hubs (gain, cost, #targets helped):',[(a,b,c) for a,b,c,g in gains[:15]])
print('hist of gain',sorted(collections.Counter(a for a,b,c,g in gains).items(),reverse=True)[:6])

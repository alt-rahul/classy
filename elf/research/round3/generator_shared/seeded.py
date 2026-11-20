"""Seeded solve prototype: mine small constants from target relations, build them first, then solve with them as free pool values."""
from lib import *
from rel import cfrels
import sys,json,copy,time,math
from solver import solve,Cfg,Solver,regalloc,simulate
from common import check,Puzzle
KW={"pool_cap":46,"evict_size":1,"regret":0.1}
def mine(p,thr=12,amax=10**5):
    T=p.targets;S={}
    def add(v,w):
        v=abs(v)
        if 1<v<=amax and v not in p.sv: S[v]=S.get(v,0)+w
    for i in range(len(T)):
        for j in range(i+1,len(T)):
            best=None
            for pp,q,r in cfrels(T[j],T[i]):
                s=math.log2(T[i])-math.log2(pp)-math.log2(q)-math.log2(abs(r)+1)
                if best is None or s>best[0]: best=(s,pp,q,r)
            if best and best[0]>thr:
                s,pp,q,r=best
                add(pp,s);add(q,s)
                if q==1:
                    a=round(r/pp);c=r-pp*a
                    add(a,s/2);add(c,s/2)
                elif pp==1:
                    a=round(r/q);c=r-q*a
                    add(a,s/2);add(c,s/2)
    return S
def clone(p,sv,targets):
    q=copy.copy(p);q.sv=list(sv);q.targets=sorted(set(targets));return q
def run_steps(p,kw=KW):
    s=Solver(p,Cfg(**kw));return s.run()
def seeded(p,smalls,extra_targets=(),kw=KW):
    smalls=[v for v in smalls if v not in p.sv]
    stA=run_steps(clone(p,p.sv,smalls),kw) if smalls else []
    pB=clone(p,list(p.sv)+list(smalls),list(p.targets)+list(extra_targets))
    stB=run_steps(pB,kw)
    steps=list(stA)+list(stB)
    simulate(p.sv,steps,p.limit)
    prog=regalloc(p.sv,steps,p.targets,p.nreg)
    return prog,len(stA),len(stB)
if __name__=='__main__':
    ps={p.id:p for p in P()}
    for pid in sys.argv[1].split(','):
        p=ps[pid]
        mode=sys.argv[2]
        if mode=='list': sm=eval(sys.argv[3])
        else:
            S=mine(p);K=int(sys.argv[3])
            sm=[v for v,w in sorted(S.items(),key=lambda x:-x[1])[:K]]
        t=time.time()
        prog,a,b=seeded(p,sm)
        print(pid,p.family,'smalls',len(sm),'A',a,'B',b,'final',len(prog) if prog else None,check(p,prog) if prog else 'regalloc fail','%.0fs'%(time.time()-t),sm,flush=True)

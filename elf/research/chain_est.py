import sys,json,math
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
import numpy as np
from collections import defaultdict
A=2048
def ccost(v,sv):
    v=abs(v)
    if v==0: return 0
    if v<=64 or v in sv: return 0.5
    if v<=4096: return 1.5
    return None
def mcost(M,sv):
    if M==1: return 0,0
    c=ccost(M,sv)
    if c is not None: return c,1
    # product of two <=4096
    best=None
    d=2
    while d*d<=M and d<=4096:
        if M%d==0 and M//d<=4096*16:
            c1=ccost(d,sv); c2=ccost(M//d,sv)
            if c2 is None and M//d<=4096*16: c2=2.5
            if c1 is not None and c2 is not None:
                if best is None or c1+c2<best: best=c1+c2
        d+=1
    if best is None: return None,None
    return best,2
def est_puzzle(p,prog):
    T=sorted(set(p.targets)-set(p.sv)); sv=set(p.sv)
    # marginal costs from program
    reg={i:v for i,v in enumerate(p.sv)}; ts=set(T); seen=set(); last=0; marg={}
    for k,(op,i,j,d) in enumerate(prog):
        a,b=reg[i],reg[j]
        v=a+b if op=='+' else a-b if op=='-' else a*b if op=='*' else a//b
        reg[d]=v
        if v in ts and v not in seen: seen.add(v); marg[v]=k+1-last; last=k+1
    avec=np.arange(-A,A+1,dtype=np.int64)
    out={}
    for j,t in enumerate(T):
        best=None
        for i in range(j):
            u=T[i]
            if u<(1<<24): continue
            d=u+avec
            r=t% d
            r=np.where(r>d//2,r-d,r)
            idx=np.nonzero(np.abs(r)<=A)[0]
            for ii in idx:
                a=int(avec[ii]); c=int(r[ii]); M=(t-c)//(u+a)
                mc,mops=mcost(M,sv)
                if mc is None: continue
                ca=ccost(a,sv); cc=ccost(c,sv)
                e=(a!=0)+(c!=0)+mops+mc+ca+cc
                if best is None or e<best[0]: best=(e,i,M,a,c)
        if best: out[t]=best
    return T,marg,out
if __name__=='__main__':
    which=sys.argv[1]; step=int(sys.argv[2])
    P=load_puzzles(which)
    if which=='scored':
        st=load_store(); ps=[p for p in P if pclass(p)=='other>=20'][::step]
    tot=defaultdict(float)
    for p in ps:
        T,marg,out=est_puzzle(p,st[p.id]['prog'])
        sav=sum(max(0,marg[t]-e[0]) for t,e in out.items())
        nl=len(out); ml=sum(marg[t] for t in out); el=sum(e[0] for e in out.values())
        tot['puz']+=1; tot['targets']+=len(T); tot['linked']+=nl; tot['marg_linked']+=ml; tot['est_linked']+=el; tot['saving']+=sav; tot['ops']+=len(st[p.id]['prog'])
    print(dict(tot))
    print('linked frac %.2f  marg/linked %.2f est/linked %.2f  saving frac of ops %.3f'%(tot['linked']/tot['targets'],tot['marg_linked']/max(1,tot['linked']),tot['est_linked']/max(1,tot['linked']),tot['saving']/tot['ops']))

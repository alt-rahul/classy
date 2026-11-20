"""Automatic hub miner: hub candidates B = (T + c)/m for |c|<=C, m<=M; a value hit by >=2 targets is a hub.
Returns hubs and the constants (m, |c|) of their relations."""
import sys,math,collections
import numpy as np
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
def mine(T,C=2500,M=2500,minB=10**7,minhits=2):
    cs=np.arange(-C,C+1,dtype=np.int64); ms=np.arange(1,M+1,dtype=np.int64)
    hit=collections.defaultdict(dict)
    for ti,t in enumerate(T):
        if t<minB: continue
        v=t+cs                              # (2C+1,)
        for lo in range(0,len(ms),256):
            mm=ms[lo:lo+256]
            R=v[:,None]%mm[None,:]
            ci,mi=np.nonzero(R==0)
            for a,b in zip(ci.tolist(),mi.tolist()):
                m=int(mm[b]); c=int(cs[a]); B=(t+c)//m
                if B<minB: continue
                d=hit[B]
                # keep the relation with the smallest |c|+m for this target
                if ti not in d or abs(c)+m<abs(d[ti][1])+d[ti][0]: d[ti]=(m,c)
    hubs={B:d for B,d in hit.items() if len(d)>=minhits}
    return hubs
def select(T,hubs,maxhubs=12):
    """greedy: repeatedly take the hub that explains most not-yet-explained targets with small constants"""
    Tset=set(T); explained=set(); out=[]
    def score(B,d):
        s=0
        for ti,(m,c) in d.items():
            if ti in explained: continue
            s+=1.0/(1+math.log10(1+abs(c))+math.log10(m))
        return s
    cand=dict(hubs)
    while cand and len(out)<maxhubs:
        B=max(cand,key=lambda b:score(b,cand[b])); d=cand.pop(B)
        new=[ti for ti in d if ti not in explained]
        if len(new)<2: break
        out.append((B,{ti:d[ti] for ti in new})); explained|=set(new)
    return out
if __name__=='__main__':
    which,pid=sys.argv[1],sys.argv[2]
    for p in load_puzzles(which):
        if p.id[-4:]==pid:
            hubs=mine(p.targets); sel=select(p.targets,hubs)
            print(len(hubs),'hub candidates')
            for B,d in sel: print(B,'target' if B in p.targets else 'hidden',{ti:mc for ti,mc in sorted(d.items())})
            hs=[B for B,d in sel]
            for i in range(len(hs)):
                for j in range(i):
                    g=math.gcd(hs[i],hs[j])
                    if g>=1000: print('  gcd',hs[i],hs[j],g)

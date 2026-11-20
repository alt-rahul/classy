from lib import *
from rel import cfrels
import math,collections,sys
def strip(t,f,B):
    x=1
    for q in f:
        if q>B: x*=q
    return x
def best_rel(a,b):
    best=None
    for pp,q,r in cfrels(a,b,10**4):
        s=math.log2(min(a,b))-math.log2(pp)-math.log2(q)-math.log2(abs(r)+1)
        if best is None or s>best[0]: best=(s,pp,q,r)
    return best
if __name__=='__main__':
    B=int(sys.argv[2]);show=sys.argv[3] if len(sys.argv)>3 else None
    agg=collections.defaultdict(lambda:[0,0,0])
    for p in P():
        if len(p.targets)<20 or p.family not in sys.argv[1].split(','): continue
        T=[t for t in p.targets if t<10**16] if p.family=='independent' else p.targets
        F=facs(T)
        X=[strip(t,f,B) for t,f in zip(T,F)]
        big=[i for i in range(len(T)) if X[i]>10**5]
        linked=set();linkedT=set()
        for a in big:
            for b in big:
                if a<b:
                    if X[a]==X[b]: linked|={a,b};continue
                    r=best_rel(X[a],X[b])
                    if r and r[0]>10:
                        linked|={a,b}
                        if show==p.id: print(' X%d=%d X%d=%d  %d*Xa = %d*Xb %+d (%.0f)'%(a,X[a],b,X[b],r[2],r[1],r[3],r[0]))
                    r=best_rel(T[a],T[b])
                    if r and r[0]>10: linkedT|={a,b}
        g=agg[p.family];g[0]+=len(big);g[1]+=len(linked);g[2]+=len(linkedT)
        if show==p.id:
            for i in range(len(T)): print(i,T[i],T[i]//X[i],X[i],i in linked)
    for k,g in agg.items(): print(k,g,'cofactor-linked %.2f target-linked %.2f'%(g[1]/g[0],g[2]/g[0]))

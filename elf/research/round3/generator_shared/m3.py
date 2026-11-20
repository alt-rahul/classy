from lib import *
import math,collections,random
random.seed(5)
def stat(Xs,C):
    cs=[c for c in range(-C,C+1) if c]
    q=[x+c for x in Xs for c in cs]
    F=facs(q);n=len(cs);hit=0;tot=0;per=collections.Counter()
    for i,x in enumerate(Xs):
        best=min((math.log(max(F[i*n+k]))/math.log(x),cs[k]) for k in range(n))
        tot+=1
        if best[0]<0.4: hit+=1;per[best[1]]+=1
    return hit,tot,sorted(per.items())
for fam in ('multiplicative','shared'):
    Xs=[];Ts=[]
    for p in P():
        if len(p.targets)<20 or p.family!=fam: continue
        T=[t for t in p.targets if t<10**13]
        for t,f in zip(T,facs(T)):
            if 10**4<max(f)<10**12: Xs.append(max(f))
            if max(f)>t**0.5 and t>10**5: Ts.append(t)
    for C in (1,3):
        print(fam,'cofactor X = largest prime: X+c smooth(u<0.4), |c|<=',C,stat(Xs,C))
        print(fam,'  control random same size',stat([random.randrange(x//2,x*2)|1 for x in Xs],C))
        print(fam,'  non-smooth targets T themselves: T+c smooth',stat(Ts,C))

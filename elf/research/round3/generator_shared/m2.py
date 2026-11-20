from lib import *
import math,collections,sys
tot=collections.Counter()
for p in P():
    if len(p.targets)<20 or p.family!='multiplicative': continue
    T=p.targets;F=facs(T)
    big=collections.Counter()
    for f in F:
        for q in set(f):
            if q>10**4: big[q]+=1
    nshare=sum(1 for f in F if any(big[q]>1 for q in f))
    npr=sum(1 for f in F if len(f)==1)
    nsm=sum(1 for f in F if max(f)<=10**4)
    sq=sum(1 for f in F if any(f.count(q)>=2 and q>50 for q in set(f)))
    print(p.id,p.sv,len(T),'bits %d-%d/%d'%(T[0].bit_length(),T[-1].bit_length(),p.limit.bit_length()),'share-bigprime',nshare,'prime',npr,'smooth(<=1e4)',nsm,'sq',sq,[ (q,c) for q,c in big.items() if c>1][:6])
    tot['n']+=len(T);tot['share']+=nshare;tot['prime']+=npr;tot['smooth']+=nsm;tot['sq']+=sq
print(tot)

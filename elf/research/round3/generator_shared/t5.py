from lib import *
from rel import cfrels
import math,collections
cnt=collections.Counter();rc=collections.Counter();per=collections.defaultdict(set)
for p in P():
    if p.family!='shared' or len(p.targets)<20: continue
    T=p.targets
    for i in range(len(T)):
        for j in range(i+1,len(T)):
            best=None
            for pp,q,r in cfrels(T[j],T[i]):
                s=math.log2(T[i])-math.log2(pp)-math.log2(q)-math.log2(abs(r)+1)
                if best is None or s>best[0]: best=(s,pp,q,r)
            if best and best[0]>14:
                s,pp,q,r=best
                cnt[pp]+=1;cnt[q]+=1;per[pp].add(p.id);per[q].add(p.id)
                if pp==1 and q==1: rc[abs(r)]+=1
print(sorted(cnt.items(),key=lambda x:-x[1])[:80])
print({k:len(v) for k,v in sorted(per.items(),key=lambda x:-len(x[1]))[:60]})
print(sorted(rc.items(),key=lambda x:-x[1])[:60])

from lib import *
import math,collections
agg=collections.defaultdict(lambda:[0,0,0,0,0])
for p in P():
    if len(p.targets)<20: continue
    T=[t for t in p.targets if t>10**9]
    F=facs(T)
    for t,f in zip(T,F):
        u=math.log(max(f))/math.log(t)
        a=agg[p.family];a[0]+=1;a[1]+=u<0.5;a[2]+=u<1/3;a[3]+=u<0.25; a[4]+=(len(f)==1)
for k,a in agg.items(): print(k,a[0],'P<t^.5 %.3f  P<t^.33 %.3f P<t^.25 %.3f prime %.3f'%(a[1]/a[0],a[2]/a[0],a[3]/a[0],a[4]/a[0]))

from lib import *
from rel import cfrels
import math,collections,sys
fam=sys.argv[1]
cls=collections.Counter();ex=[]
for p in P():
    if p.family!=fam or len(p.targets)<20: continue
    T=p.targets;S=set(p.sv)
    for i in range(len(T)):
        for j in range(i+1,len(T)):
            m,r=divmod(T[j],T[i])
            if r>T[i]//2: m+=1;r-=T[i]
            if m>5000: continue
            s=math.log2(T[i])-math.log2(m)-math.log2(abs(r)+1)
            if s<14: continue
            a=round(r/m);c=r-m*a
            k=('x%s'%('1' if m==1 else 'S' if m in S else 'm'), 'r0' if r==0 else ('a' if c==0 else ('c' if abs(r)<m else 'ac')))
            cls[k]+=1
            ex.append((p.id,p.sv,m,r,a,c))
print(sorted(cls.items(),key=lambda x:-x[1]))
import random
random.seed(1)
for e in random.sample(ex,60): print(e)

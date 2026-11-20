from lib import *
import sys,math,collections
ps={p.id:p for p in P()}
p=ps[sys.argv[1]];T=p.targets
print(p.family,p.sv,p.limit)
F=facs(T)
for i,t in enumerate(T):
    gs=sorted(((math.gcd(t,u),j) for j,u in enumerate(T) if j!=i),reverse=True)[:4]
    print(i,t,F[i],[(g,j) for g,j in gs if g>100])

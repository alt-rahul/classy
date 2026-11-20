from lib import *
import sys
ps={p.id:p for p in P()}
p=ps[sys.argv[1]];K=int(sys.argv[2])
F,seen=formula_levels(p.sv,p.limit,K)
print(p.sv,p.limit,[len(f) for f in F])
for t in p.targets:
    k=next((k for k in range(K+1) if t in F[k]),None)
    print(t,factor(t),k,expr(t,seen,p.sv) if k is not None else '')

from lib import *
import sys
sv=eval(sys.argv[1]);K=int(sys.argv[2]);q=eval(sys.argv[3])
F,seen=formula_levels(sv,10**7,K)
print([len(f) for f in F])
for t in q:
    k=next((k for k in range(K+1) if t in F[k]),None)
    print(t,k,expr(t,seen,sv) if k is not None else '')

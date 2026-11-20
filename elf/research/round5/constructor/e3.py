# E3: core = stored values with reach >= K; rebuild ALL targets over the core with recipe-union selection
import sys, time
from rlib import *
from prof import prof, P, st
pid=sys.argv[1]; K=int(sys.argv[2]); cmax=int(sys.argv[3])
p=P[pid]; L,n,rc,nodes,order,reach=prof(pid); T=set(p.targets)-set(p.sv)
core=[v for v in order if v not in T and len(reach[v])>=K]
# targets used as operands by others stay as they are (not in the pool)
pool=sorted(set(core)|set(p.sv))
t0=time.time(); D,R,err=enum(p.limit,pool,sorted(T),cmax,64,34)
print(err.split('\n')[0],'enum %.1fs'%(time.time()-t0))
have=[t for t in T if R[t]]
mins=[min(r[0] for r in R[t]) for t in have]
print(pid,'len',L,'core',len(core),'store non-core',L-len(core),'targets',len(T),'with recipes',len(have))
print('min tree cost hist',sorted(collections.Counter(mins).items()),'sum',sum(mins),'nrec median',sorted(len(R[t]) for t in have)[len(have)//2])
t0=time.time(); tot,ch,hist=select(D,pool,R,have,sweeps=8)
print('union',tot,'hist',hist,'sel %.1fs'%(time.time()-t0),'=> total',len(core)+tot,'(+ missing targets %d)'%(len(T)-len(have)))

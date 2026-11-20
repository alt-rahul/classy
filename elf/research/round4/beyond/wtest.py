import sys, time, json
from wlib import *
P={p.id:p for p in load_puzzles()}; st=load_store()
KP=int(sys.argv[5]) if len(sys.argv)>5 else 0
HV=int(sys.argv[6]) if len(sys.argv)>6 else 1
pid=sys.argv[1]; kmin=int(sys.argv[2]); cmin=int(sys.argv[3]); tl=float(sys.argv[4]) if len(sys.argv)>4 else 120
p=P[pid]; prog=st[pid]['prog']
nodes,order=build_dag(p,prog)
print(pid,len(prog),len(order))
# sanity: dfs reschedule of unchanged program
T=[v for v in order if v in set(p.targets)]
print('dfs resched ok:', try_emit(p,nodes,[dfs_order(p,nodes,T)]) is not None, 'orig order ok:', try_emit(p,nodes,[order]) is not None)
cones=all_cones(p,nodes)
for t,(cone,desc,users) in sorted(cones.items(), key=lambda x:-len(x[1][0])):
    k=len(cone)
    if k<kmin: break
    pool=[v for v in list(p.sv)+order if v not in cone and v not in desc]
    t0=time.time()
    ops,err=run_wide(p.limit,t,pool,cmin,k-1+KP,HV,tl)
    print('target',t,t.bit_length(),'cone',k,'pool',len(pool),'->',None if ops is None else len(ops),'%.0fs'%(time.time()-t0)); print('   ',err.strip().replace('\n',' | '))
    if ops:
        nn,cands=splice(p,nodes,order,cone,ops,t)
        pr=try_emit(p,nn,cands)
        print('   new size',len(nn),'emit',None if pr is None else len(pr))

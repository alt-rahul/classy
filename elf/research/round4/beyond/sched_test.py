import sys
from wlib import *
P={p.id:p for p in load_puzzles()}; st=load_store()
p=P['scored_0002']; nodes,order=build_dag(p,st[p.id]['prog'])
cones=all_cones(p,nodes)
for t,(cone,desc,users) in sorted(cones.items(), key=lambda x:-len(x[1][0]))[:2]:
    pool=[v for v in list(p.sv)+order if v not in cone and v not in desc]
    os.environ['X']='1'
    import wlib; wlib.WIDE=os.path.join(HERE,'wide2')
    ops,err=run_wide(p.limit,t,pool,len(cone)-1,len(cone)-1,1,25)
    nn,cands=splice(p,nodes,order,cone,ops,t)
    import time; t0=time.time()
    pr=try_emit4(p,nn,cands,order,60)
    print(len(cone),len(ops),len(nn),'emit',None if pr is None else len(pr),'%.1fs'%(time.time()-t0))
    sv=set(p.sv); hint=make_hint(nn,order,sv); rng=random.Random(1)
    print('orig pressure',pressure(p,nodes,order),'cands',[ (len(c),pressure(p,nn,c)) for c in cands if len(c)==len(nn)])
    ps=[]
    for k in range(50):
        od=list_sched(p,nn,hint,rng,0.0 if k==0 else 0.5)
        ps.append(pressure(p,nn,od) if len(od)==len(nn) else -1)
    print(sorted(ps)[:10])

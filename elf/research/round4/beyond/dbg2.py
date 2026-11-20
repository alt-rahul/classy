import sys, pickle, time
from wlib import *
P={p.id:p for p in load_puzzles()}; st=load_store()
pid,i,j,jc,si,sj=pickle.load(open('/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad/dbg_%s.pkl'%sys.argv[1],'rb'))
p=P[pid]; nodes,order=build_dag(p,st[pid]['prog'])
for a in si[:2]:
    for b in sj[:1]:
        nn,c=splice_blocks(p,nodes,order,jc,[a,b])
        for T0 in (0.5,2.0,8.0):
            t0=time.time(); E,od=sa_sched2(p,nn,c[0],p.nreg,iters=30000,T0=T0*50)
            print(len(nn),'start',pressure(p,nn,c[0]),'T0',T0,'-> E',E,'pressure',pressure(p,nn,od),'%.1fs'%(time.time()-t0),'emit',try_emit(p,nn,[od]) is not None,flush=True)

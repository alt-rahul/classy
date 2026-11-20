import sys, pickle, time
from wlib import *
P={p.id:p for p in load_puzzles()}; st=load_store()
pid,i,j,jc,si,sj=pickle.load(open('/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad/dbg_%s.pkl'%sys.argv[1],'rb'))
p=P[pid]; nodes,order=build_dag(p,st[pid]['prog'])
for a in si[:3]:
    nn,c=splice_blocks(p,nodes,order,jc,[a,sj[0]])
    t0=time.time(); pr=try_emit5(p,nn,c,20)
    print(len(nn),'start',pressure(p,nn,c[0]),'emit',None if pr is None else len(pr),'%.1fs'%(time.time()-t0),flush=True)

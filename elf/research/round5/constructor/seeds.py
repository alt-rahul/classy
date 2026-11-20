# distribution over seeds / palettes; offers wins to the store
import sys, time, json, random
from crun import *
from pal import palette
from prof import P, st
from multiprocessing import Pool
PALS=[[],[2],[2,3],[2,3,4],[2,4,8],[2,3,5],[2,4,8,16],[3],[2,5],[2,3,4,6],[2,4],[2,3,7]]
def one(a):
    pid,seed,args,binary=a; p=P[pid]; ar=list(args); ar[3]=seed
    init=palette(p,PALS[seed%len(PALS)]) if len(set(p.sv))>=3 else []
    t0=time.time()
    try: prog,log=cons(p,init,ar,binary=binary,tl=1500)
    except Exception as e: prog=None
    return pid,seed,prog,time.time()-t0
if __name__=='__main__':
    ids=sys.argv[1].split(','); n=int(sys.argv[2]); args=sys.argv[3].split(','); binary=sys.argv[4]; nw=int(sys.argv[5]); tag=sys.argv[6]
    seedl=[int(x) for x in sys.argv[7].split(',')] if len(sys.argv)>7 else list(range(1,n+1))
    jobs=[(i,s,args,binary) for s in seedl for i in ids]; res={}
    with Pool(nw) as pool:
        for pid,seed,prog,dt in pool.imap_unordered(one,jobs):
            L=len(prog) if prog else None; res.setdefault(pid,[]).append(L)
            print(pid,'seed',seed,'len',L,'store',st[pid]['len'],'%.0fs'%dt,flush=True)
            if prog:
                best=min(x for x in res[pid] if x)
                if L==best: json.dump(prog,open('best_%s_%s.json'%(tag,pid),'w'))
                s2=load_store()
                if L<s2[pid]['len'] and offer(s2,P[pid],prog,'r5-cons'): save_store(s2); print('  STORED',pid,L,flush=True)
    for pid in ids:
        v=sorted(x for x in res[pid] if x); print(pid,'store',st[pid]['len'],'min',v[0] if v else None,'median',v[len(v)//2] if v else None,'max',v[-1] if v else None,v)

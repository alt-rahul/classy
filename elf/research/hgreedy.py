import sys,json,math,random,time
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
import engine
def l1(pool,limit):
    s=set()
    for a in pool:
        for b in pool:
            if a+b<=limit: s.add(a+b)
            if a>b: s.add(a-b)
            if a*b<=limit: s.add(a*b)
            if a%b==0: s.add(a//b)
    return s-set(pool)
def H(pool,limit,unc,mc=5):
    engine.build(pool,limit,2,0)
    h=0
    for t in unc:
        r=engine.find_pair(t,mc)
        h+= r[0] if r else mc+1.5
    return h
def solve(p,mc=5,seed=0,maxc=10**9):
    rng=random.Random(seed)
    pool=list(dict.fromkeys(p.sv)); unc=set(p.targets)-set(pool); n=0
    while unc:
        cands=l1(pool,p.limit)
        best=None
        h0=H(pool,p.limit,unc,mc)
        for v in cands:
            u2=unc-{v}
            h=H(pool+[v],p.limit,u2,mc)
            key=(h,rng.random())
            if best is None or key<best[0]: best=(key,v)
        v=best[1]; pool.append(v); unc.discard(v); n+=1
        #print(n,v,h0,'->',best[0][0],len(unc))
    return n,pool
if __name__=='__main__':
    P={p.id:p for p in load_puzzles('practice')}
    tot=0
    for pid in sys.argv[1].split(','):
        t0=time.time(); n,pool=solve(P[pid]); tot+=n
        print(pid,P[pid].family,len(P[pid].targets),n,'%.1fs'%(time.time()-t0),flush=True)
    print('total',tot)

import sys,json,math,random,time,pickle
sys.path.insert(0,'/home/user/classy/elf/src')
import engine
from collections import Counter
def inv(t,s,limit):
    out=[]
    if t>s: out.append(t-s)
    if t+s<=limit: out.append(t+s)
    if s>t: out.append(s-t)
    if s>1:
        if t%s==0: out.append(t//s)
        if t*s<=limit: out.append(t*s)
        if s%t==0 and s//t>1: out.append(s//t)
    return out
def l1(pool,limit):
    s=set()
    for a in pool:
        for b in pool:
            if a+b<=limit: s.add(a+b)
            if a>b: s.add(a-b)
            if a*b<=limit: s.add(a*b)
            if a%b==0: s.add(a//b)
    return sorted(s-set(pool))
def exact_le(t,c,pool,L1,limit,stats):
    """True if tree cost of t <= c (c<=6)."""
    r=engine.find_pair(t,min(c,4))
    if r: return True
    if c<=4: return False
    if engine.find_pair(t,c): return True      # (3,1),(2,2) / (3,2)
    if c==5:
        for p in pool:
            for v in inv(t,p,limit):
                stats['fp']+=1
                if engine.find_pair(v,4): return True
        return False
    if c==6:
        for w in L1:
            for v in inv(t,w,limit):
                stats['fp']+=1
                if engine.find_pair(v,4): return True
        for p in pool:
            for v in inv(t,p,limit):
                if exact_le(v,5,pool,L1,limit,stats): return True
        return False
if __name__=='__main__':
    d=pickle.load(open(sys.argv[1],'rb')); nt=int(sys.argv[2])
    pool,limit=d['pool'],d['limit']
    L1=l1(pool,limit)
    print('pool bits',sorted(v.bit_length() for v in pool))
    print(engine.build(pool,limit,3,0))
    res=[]
    for t in d['ts'][:nt]:
        t0=time.time(); stats=Counter()
        r=engine.find_pair(t,11); c0=r[0] if r else 99
        w=engine.find_window(t,min(c0,12),1.5e6); c1=w[0] if w else 99
        e=min(c0,c1)
        ex=e
        for c in (5,6):
            if c<e and exact_le(t,c,pool,L1,limit,stats): ex=c; break
        res.append((e,ex))
        print(t.bit_length(),'engine',e,'exact<=6:',ex,'%.1fs'%(time.time()-t0),dict(stats),flush=True)
    import numpy as np
    print('engine mean %.2f exact mean %.2f'%(np.mean([a for a,b in res]),np.mean([b for a,b in res])), Counter(res))

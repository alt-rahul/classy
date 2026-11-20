import sys,time,pickle
sys.path.insert(0,'/tmp/fable_scratch'); sys.path.insert(0,'/home/user/classy/elf/src')
from exact import *
from exh import levels
import numpy as np
def exact7(t,pool,L1,L2,limit,stats):
    # (4,2)
    for w in L2:
        for v in inv(t,w,limit):
            stats['fp42']+=1
            if engine.find_pair(v,4): return '42'
    for w in L1:
        for v in inv(t,w,limit):
            stats['e5']+=1
            if exact_le(v,5,pool,L1,limit,stats): return '51'
    for p in pool:
        for v in inv(t,p,limit):
            if exact_le(v,6,pool,L1,limit,stats): return '60'
    return None
d=pickle.load(open(sys.argv[1],'rb')); nt=int(sys.argv[2])
pool,limit=d['pool'],d['limit']
L0,L1s,L2s=levels(pool,limit); L1=sorted(L1s); L2=sorted(L2s)
print('pool bits',sorted(v.bit_length() for v in pool)); print(engine.build(pool,limit,3,0),flush=True)
res=[]
for t in d['ts'][:nt]:
    t0=time.time(); stats=Counter()
    r=engine.find_pair(t,11); c0=r[0] if r else 99
    w=engine.find_window(t,min(c0,12),1.5e6); c1=w[0] if w else 99
    e=min(c0,c1); ex=e
    if e>7:
        if exact_le(t,6,pool,L1,limit,stats): ex=6
        else:
            s=exact7(t,pool,L1,L2,limit,stats)
            if s: ex=7
    res.append((e,ex))
    print(t.bit_length(),'engine',e,'exact<=7:',ex,'%.0fs'%(time.time()-t0),dict(stats),flush=True)
print('engine mean %.2f exact mean %.2f'%(np.mean([a for a,b in res]),np.mean([b for a,b in res])),Counter(res))

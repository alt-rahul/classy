import sys,json,math,random,time
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from solver import *
class SW(Solver):
    lo=-3; hi=1; nsmall=6
    def evict(self,pool,last_use,created,cap,protect):
        if len(pool)<=cap: return
        B=max(self.p.targets).bit_length()
        c=B/3.0
        cand=[v for v in pool if v>self.cfg.keep_small and v not in protect]
        smalls=sorted([v for v in cand if v.bit_length()<c+self.lo])
        keep_s=set(smalls[:self.nsmall])
        def key(v):
            b=v.bit_length()
            d=0
            if b>c+self.hi: d=b-(c+self.hi)+100      # too big: evict first, biggest first
            elif b<c+self.lo and v not in keep_s: d=50  # surplus small
            return (-d,last_use.get(v,0),created.get(v,0))
        cand.sort(key=key)
        k=len(pool)-cap
        drop=set(cand[:k])
        if drop: self.evicted=True
        pool[:]=[v for v in pool if v not in drop]
if __name__=='__main__':
    P={p.id:p for p in load_puzzles('practice')}
    ids=sys.argv[1].split(',')
    tb=0; tw={}
    for pid in ids:
        p=P[pid]; t0=time.time()
        s=Solver(p,Cfg(pool_cap=46,evict_size=1)); base=len(regalloc(p.sv,s.run(),p.targets)); tb+=base
        line={}
        for name,(lo,hi,ns) in {'w(-3,1,6)':(-3,1,6),'w(-4,2,10)':(-4,2,10),'w(-3,1,99)':(-3,1,99)}.items():
            SW.lo,SW.hi,SW.nsmall=lo,hi,ns
            s=SW(p,Cfg(pool_cap=46)); st=s.run(); prog=regalloc(p.sv,st,p.targets); assert check(p,prog) is None
            line[name]=len(prog); tw[name]=tw.get(name,0)+len(prog)
        print(pid,math.log2(p.limit)//1,'evict_size',base,line,'%.0fs'%(time.time()-t0),flush=True)
    print('evict_size',tb,tw)

import sys,json,math,random,time
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from solver import *
class S3(Solver):
    alpha=0.15; maxc=7
    def search_round(self,ts,order,pool):
        limit=self.p.limit
        engine.build(order,limit,3,0)
        self.deep_age=None
        best=None
        for t in ts:
            r=engine.find_pair(t,self.maxc)
            if r:
                sc=r[0]-self.alpha*t.bit_length()
                if best is None or (sc,r[0],t)<best[0]: best=((sc,r[0],t),r)
        if best is not None:
            return (best[1][0],best[0][2],best[1][1],'reg')
        return Solver.search_round(self,ts,order,pool)
if __name__=='__main__':
    P={p.id:p for p in load_puzzles('practice')}
    ids=sys.argv[1].split(','); alphas=[float(x) for x in sys.argv[2].split(',')]
    tot={a:0 for a in alphas}; tb=0
    for pid in ids:
        p=P[pid]; line=[]
        t0=time.time()
        s=Solver(p,Cfg(pool_cap=46,evict_big=4)); base=len(regalloc(p.sv,s.run(),p.targets)); tb+=base
        for a in alphas:
            S3.alpha=a
            s=S3(p,Cfg(pool_cap=46,evict_big=4)); st=s.run(); simulate(p.sv,st,p.limit)
            prog=regalloc(p.sv,st,p.targets); assert check(p,prog) is None
            line.append(len(prog)); tot[a]+=len(prog)
        print(pid,p.family,len(p.targets),'base',base,dict(zip(alphas,line)),'%.0fs'%(time.time()-t0),flush=True)
    print('base',tb,tot)

import sys,json,math,random,time,pickle
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from solver import *
pid=sys.argv[1]; stop=int(sys.argv[2]); which=sys.argv[3] if len(sys.argv)>3 else 'practice'
P={p.id:p for p in load_puzzles(which)}
p=P[pid]
class S2(Solver):
    def search_round(self,ts,order,pool):
        self.rn=getattr(self,'rn',0)+1
        if self.rn==stop:
            pickle.dump({'pool':list(order),'ts':list(ts),'limit':p.limit,'sv':p.sv},open('/tmp/fable_scratch/pool_%s.pkl'%pid,'wb'))
            raise SystemExit
        r=Solver.search_round(self,ts,order,pool)
        print(self.rn,r[0],r[3],flush=True)
        return r
S2(p,Cfg(pool_cap=46,evict_size=1)).run()

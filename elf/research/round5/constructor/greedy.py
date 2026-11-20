# xyr rolling-pool greedy constructor (steady-state test): init pool = stored values with reach >= K (or given list)
import sys, time, random
from rlib import *
sys.path.insert(0,'/home/user/classy/elf/src')
import solver
import wlib
WIDE2='/home/user/classy/elf/research/round4/beyond/wide2'
def wide(limit,t,pool,cmin,cmax,tl=20):
    inp='%d %d %d %d %d\n%d\n%s\n'%(limit,t,cmin,cmax,1,len(pool),' '.join(map(str,pool)))
    r=subprocess.run([WIDE2,str(tl)],input=inp,capture_output=True,text=True)
    lines=r.stdout.strip().split('\n')
    if not lines or not lines[0].startswith('SOL'): return None
    return [(ln.split()[1],int(ln.split()[0]),int(ln.split()[2]),int(ln.split()[3])) for ln in lines[1:]]
def run(p, init_steps, cap=42, cmax=8, seed=0, verbose=False, keep_small=1<<14, policy='fifo'):
    rnd=random.Random(seed)
    steps=list(init_steps); pool=list(p.sv)+[s[3] for s in init_steps]
    pool=list(dict.fromkeys(pool))
    T=set(p.targets)-set(p.sv); rem=[t for t in p.targets if t in T and t not in pool]
    big=min(p.targets)>>4
    pool=[v for v in pool if v<big or v in p.sv]
    costs=[]; lastuse={v:0 for v in pool}; step=0
    def evict():
        while len(pool)>cap:
            cand=[v for v in pool if v>=keep_small and v not in p.sv] or [v for v in pool if v not in p.sv]
            if policy=='fifo': v=cand[0]
            else: v=min(cand,key=lambda v:lastuse.get(v,0))
            pool.remove(v)
    evict()
    while rem:
        step+=1
        D,R,err=enum(p.limit,pool,rem,cmax)
        best=None
        for t in rem:
            for rec in R.get(t,()):
                key=(rec[0],rnd.random())
                if best is None or key<best[0]: best=(key,t,rec)
        if best is None:
            # fallback: exact search for the smallest remaining target sets
            got=None
            for c in (9,10,11):
                for t in rnd.sample(rem,min(len(rem),4)):
                    ops=wide(p.limit,t,sorted(pool),c,c,15)
                    if ops: got=(t,ops); break
                if got: break
            if not got: return None,costs
            t,ops=got; new=[(op,a,b,v) for (op,a,b,v) in ops]
        else:
            _,t,rec=best; nd=nodes_of(D,set(pool),t,rec); new=[(op,a,b,v) for v,(op,a,b) in nd.items()]
        ps=set(pool)
        for (op,a,b,v) in new:
            steps.append((op,a,b,v))
            for u in (a,b): lastuse[u]=step
            if v<big and v not in ps: pool.append(v); ps.add(v); lastuse[v]=step
        costs.append(len(new)); rem=[x for x in rem if x!=t and x not in set(s[3] for s in new)]
        if verbose: print(step,'target',t,'cost',len(new),'pool',len(pool),'rem',len(rem),flush=True)
        evict()
    return steps,costs
if __name__=='__main__':
    from prof import prof, P, st
    pid=sys.argv[1]; K=int(sys.argv[2]); cap=int(sys.argv[3]); pol=sys.argv[4] if len(sys.argv)>4 else 'fifo'
    p=P[pid]; L,n,rc,nodes,order,reach=prof(pid); T=set(p.targets)
    core=[v for v in order if v not in T and len(reach[v])>=K]
    init=[(nodes[v][0],nodes[v][1],nodes[v][2],v) for v in core]
    t0=time.time(); steps,costs=run(p,init,cap=cap,verbose=False,policy=pol)
    print(pid,'store',L,'core',len(core),'costs',costs,'sum',sum(costs),'time %.0fs'%(time.time()-t0))
    if steps:
        prog=solver.regalloc(p.sv,steps,p.targets,p.nreg)
        print('total',len(prog) if prog else None, check(p,prog) if prog else 'regalloc-fail')

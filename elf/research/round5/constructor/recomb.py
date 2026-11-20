# Whole-program recombination: every target gets all xyr recipes (<= cmax new nodes) over ALL mid values of the program;
# CP-SAT then picks one derivation per target minimising the number of values that must exist. Scheduling: wlib (sched SA).
import sys, time, json, collections
from rlib import *
sys.path.insert(0,'/home/user/classy/elf/research/round4/beyond')
import wlib
from ortools.sat.python import cp_model
def recomb(p, prog, cmax=7, tl=60, bigdiv=8, verbose=True, workers=1, nb=8, rbits=36):
    nodes, order = wlib.build_dag(p, prog); sv=set(p.sv); T=[t for t in dict.fromkeys(p.targets) if t not in sv]
    Tset=set(T); big=min(p.targets)//bigdiv
    pool=sorted(set(v for v in order if v<big and v not in Tset)|sv)
    t0=time.time(); D,R,err=enum(p.limit,pool,T,cmax,nb,rbits)
    if verbose: print('  pool',len(pool),err.split('\n')[0],'enum %.0fs'%(time.time()-t0),'recipes',sum(len(R[t]) for t in T),flush=True)
    poolset=set(pool)
    m=cp_model.CpModel(); b={}
    def B(v):
        if v not in b: b[v]=m.NewBoolVar('b%d'%len(b))
        return b[v]
    newdef={}   # new node value -> (op,a,b)
    for v in order:
        op,a,c=nodes[v]
        for u in (a,c):
            if u not in sv: m.AddImplication(B(v),B(u))
    nrec=0
    for t in T:
        alts=[]
        o=m.NewBoolVar('o'); alts.append(o)       # original derivation
        op,a,c=nodes[t]
        for u in (a,c):
            if u not in sv: m.AddImplication(o,B(u))
        seen=set()
        for rec in R.get(t,()):
            nd=nodes_of(D,poolset,t,rec)
            key=frozenset(nd.keys())
            if key in seen: continue
            seen.add(key)
            # skip recipes identical to existing structure only if they add nothing (still fine to add)
            y=m.NewBoolVar('y'); alts.append(y); nrec+=1
            for w,(op2,a2,c2) in nd.items():
                if w==t:
                    for u in (a2,c2):
                        if u not in sv: m.AddImplication(y,B(u))
                    continue
                m.AddImplication(y,B(w))
                if w in nodes: continue          # already a program value: keeps its original derivation
                if w not in newdef:
                    newdef[w]=(op2,a2,c2)
                    for u in (a2,c2):
                        if u not in sv: m.AddImplication(B(w),B(u))
        m.AddExactlyOne(alts); m.Add(B(t)==1)
        b[('alt',t)]=alts; b[('recs',t)]=None
    R2={t:[] for t in T}
    # rebuild alt -> recipe mapping (same iteration order)
    obj=[var for k,var in b.items() if not isinstance(k,tuple)]
    m.Minimize(sum(obj))
    # hint: current program
    for v in order: m.AddHint(B(v),1)
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=tl; s.parameters.num_search_workers=workers
    st=s.Solve(m)
    if st not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return None,'nosol'
    val=int(round(s.ObjectiveValue()))
    if verbose: print('  cp status',s.StatusName(st),'objective',val,'current',len(order),'bound',s.BestObjectiveBound(),'%.0fs'%(time.time()-t0),flush=True)
    if val>=len(order): return None,'nogain'
    # extract: chosen derivations
    nn={}
    for t in T:
        alts=b[('alt',t)]; k=[i for i,a in enumerate(alts) if s.Value(a)][0]
        if k==0: nn[t]=nodes[t]
        else:
            seen=set(); idx=0
            for rec in R.get(t,()):
                nd=nodes_of(D,poolset,t,rec); key=frozenset(nd.keys())
                if key in seen: continue
                seen.add(key); idx+=1
                if idx==k: nn[t]=nd[t]; break
    for v,var in b.items():
        if isinstance(v,tuple) or v in Tset: continue
        if s.Value(var): nn[v]=nodes[v] if v in nodes else newdef[v]
    # DCE + topological order (keep original relative order, new nodes by dependency)
    need=set(); stk=list(T)
    while stk:
        x=stk.pop()
        if x in need or x in sv: continue
        if x not in nn: return None,'incomplete %d'%x
        need.add(x); stk+=list(nn[x][1:])
    nn={v:nn[v] for v in need}
    pos={v:i for i,v in enumerate(order)}; out=[]; done=set()
    def emit(v):
        if v in done or v in sv: return
        stack=[(v,0)]
        while stack:
            x,stt=stack.pop()
            if x in done or x in sv: continue
            if stt==0:
                stack.append((x,1)); stack.append((nn[x][2],0)); stack.append((nn[x][1],0))
            else: done.add(x); out.append(x)
    # order targets by original position; emit dependencies first
    for v in sorted(nn.keys(), key=lambda v: pos.get(v, 10**9)):
        if v in Tset or v not in pos: continue
    for t in sorted(T,key=lambda t:pos[t]): emit(t)
    return (nn,out),'ok %d'%len(nn)
def emit_prog(p, nn, od, tl=40.0):
    import solver
    steps=[(nn[v][0],nn[v][1],nn[v][2],v) for v in od]
    prog=solver.regalloc(p.sv,steps,p.targets,p.nreg)
    if prog is not None and check(p,prog) is None: return prog,'direct'
    pr=wlib.try_emit5(p,nn,[od],tl)
    if pr is not None and check(p,pr) is None: return pr,'sched'
    return None,'noemit%d'%wlib.pressure(p,nn,od)
def improve(p, prog, cmax=7, tl=60, rounds=3, verbose=True, **kw):
    for r in range(rounds):
        res,msg=recomb(p,prog,cmax,tl,verbose=verbose,**kw)
        if res is None:
            if verbose: print('  round',r,msg)
            break
        nn,od=res; pr,how=emit_prog(p,nn,od)
        if verbose: print('  round',r,msg,'emit',how,len(pr) if pr else None,flush=True)
        if pr is None or len(pr)>=len(prog): break
        prog=pr
    return prog
if __name__=='__main__':
    P={p.id:p for p in load_puzzles('scored')}; st=load_store()
    pid=sys.argv[1]; src=sys.argv[2]; cmax=int(sys.argv[3]); tl=float(sys.argv[4])
    p=P[pid]; prog=st[pid]['prog'] if src=='store' else json.load(open(src))
    print(pid,'start',len(prog),'store',st[pid]['len'],flush=True)
    new=improve(p,prog,cmax,tl)
    print(pid,'final',len(new),check(p,new))
    if len(new)<len(prog): json.dump(new,open('recomb_%s.json'%pid,'w'))
    if len(new)<st[pid]['len']:
        s2=load_store()
        if offer(s2,p,new,'r5-recomb'): save_store(s2); print('STORED',pid,len(new))

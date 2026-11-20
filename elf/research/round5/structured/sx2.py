"""SX2: same universe as sx.py, pure Boolean model + lazy loop cuts (no level variables)."""
from lib import *
import numpy as np, time, random, wlib
from ortools.sat.python import cp_model
import sx
TAG=os.environ.get('SX_TAG','sx2'); CAP=int(os.environ.get('SX_CAP','1500'))

def solve(p,prog,tl,verbose=1,seed=1):
    t0=time.time()
    nodes,order=wlib.build_dag(p,prog)
    n0=len(order); sv=set(p.sv)
    cur,c1=sx.universe(p,nodes); curset=set(cur)
    U=np.array(sorted(curset|set(c1)),dtype=np.int64)
    rng=random.Random(seed)
    rec={}; usednew=set()
    K2=int(os.environ.get('SX_K2',str(CAP//3)))
    for x in order:
        rs=sx.recipes_for(x,U,None,p.limit)
        k0=[r for r in rs if r[1] in curset and r[2] in curset]
        k1=[r for r in rs if (r[1] in curset)!=(r[2] in curset)]
        k2=[r for r in rs if r[1] not in curset and r[2] not in curset]
        if len(k1)>CAP: k1=rng.sample(k1,CAP)
        if len(k2)>K2: k2=rng.sample(k2,K2)
        rs=k0+k1+k2
        cr=nodes[x]
        if cr not in rs and (cr[0],cr[2],cr[1]) not in rs: rs.append(cr)
        rec[x]=rs
        for r in rs:
            for o in (r[1],r[2]):
                if o not in curset: usednew.add(o)
    for v in usednew: rec[v]=c1[v][:4]
    vals=list(rec); nrec=sum(len(r) for r in rec.values())
    if verbose: print(p.id,'n0',n0,'U',len(U),'usednew',len(usednew),'recipes',nrec,flush=True)
    m=cp_model.CpModel()
    X={v:m.NewBoolVar('') for v in vals}
    T=set(p.targets)-sv
    ys={}; byval={}
    for v in vals:
        lits=[]
        for r in rec[v]:
            y=m.NewBoolVar(''); ys[(v,r)]=y; lits.append(y)
            a,b=r[1],r[2]
            if a not in sv: m.AddImplication(y,X[a])
            if b not in sv and b!=a: m.AddImplication(y,X[b])
        byval[v]=lits
        if v in T: m.AddBoolOr(lits); m.Add(X[v]==1)
        else: m.AddBoolOr(lits+[X[v].Not()])
    # 2-cycle clauses
    nc=0
    for v in vals:
        for r in rec[v]:
            for o in (r[1],r[2]):
                if o in sv or o not in rec: continue
                for r2 in rec[o]:
                    if v in (r2[1],r2[2]) and (o,r2)<(v,r):
                        m.AddBoolOr([ys[(v,r)].Not(),ys[(o,r2)].Not()]); nc+=1
    obj=sum(X[v] for v in vals)
    m.Add(obj<=n0-1)
    for v in vals:
        inc=v in nodes
        m.AddHint(X[v],1 if inc else 0)
        for r in rec[v]: m.AddHint(ys[(v,r)],1 if inc and r==nodes[v] else 0)
    rounds=0; status='UNKNOWN'; best=None
    while time.time()-t0<tl:
        s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=max(1.0,tl-(time.time()-t0)); s.parameters.num_workers=1; s.parameters.random_seed=seed+rounds
        st_=s.Solve(m); rounds+=1
        if st_==cp_model.INFEASIBLE: status='INFEASIBLE'; break
        if st_ not in (cp_model.OPTIMAL,cp_model.FEASIBLE): status='UNKNOWN'; break
        sel={v for v in vals if s.Value(X[v])}
        act={v:[r for r in rec[v] if s.Value(ys[(v,r)])] for v in sel}
        # derivability
        der=set(sv); ch=True; nn={}
        while ch:
            ch=False
            for v in sel:
                if v in der: continue
                for r in act[v]:
                    if r[1] in der and r[2] in der: der.add(v); nn[v]=r; ch=True; break
        if T<=der:
            need=set(); stk=list(T)
            while stk:
                x=stk.pop()
                if x in need or x in sv: continue
                need.add(x); stk+=[nn[x][1],nn[x][2]]
            best={v:nn[v] for v in need}; status='FEASIBLE'; break
        S=sel-der
        ext=[ys[(u,r)] for u in S for r in rec[u] if r[1] not in S and r[2] not in S]
        # any selected v in S needs an external entry into S
        for v in S:
            if v in T: m.AddBoolOr(ext); break
        else:
            for v in S: m.AddBoolOr(ext+[X[v].Not()])
        # also per-SCC-ish: cut for underivable targets' ancestor sets is covered above
    if verbose: print(p.id,'status',status,'rounds',rounds,'2cyc',nc,'t %.1f'%(time.time()-t0),flush=True)
    if best is None: return None,status
    lv={}
    def depth(v):
        if v in sv: return 0
        if v not in lv: lv[v]=1+max(depth(best[v][1]),depth(best[v][2]))
        return lv[v]
    sys.setrecursionlimit(10000)
    od=wlib.dfs_order(p,best,[t for t in order if t in T and t in best]+[t for t in T if t not in order])
    od2=sorted(best,key=lambda v:(depth(v),v))
    pr=wlib.try_emit5(p,best,[od,od2])
    return pr,status

if __name__=='__main__':
    tl=float(sys.argv[1]); ids=sys.argv[2:]
    P={p.id:p for p in load_puzzles('scored')}
    for pid in ids:
        st=load_store(); p=P[pid]; prog=st[pid]['prog']
        pr,status=solve(p,prog,tl)
        if pr is not None and check(p,pr) is None and len(pr)<len(prog):
            st=load_store()
            if offer(st,p,pr,TAG): save_store(st)
            print('GAIN',pid,len(prog),'->',len(pr),flush=True)
        else: print('none',pid,len(prog),status,'emitfail' if status=='FEASIBLE' else '',flush=True)
        with open(R+'/sx_runs.jsonl','a') as f: f.write(json.dumps({'id':pid,'old':len(prog),'new':len(pr) if pr else None,'status':status,'tl':tl,'tag':TAG,'env':{k:v for k,v in os.environ.items() if k.startswith('SX_')}})+'\n')

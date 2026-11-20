"""Structure-aware exchange (SX): ILP recombination over the full 1-op neighbourhood universe of a program's values.
usage: sx.py <tl_seconds> id [id...]    (env SX_TAG, SX_CAP)"""
from lib import *
import numpy as np, time, random, wlib
from ortools.sat.python import cp_model
TAG=os.environ.get('SX_TAG','sx1'); CAP=int(os.environ.get('SX_CAP','1500'))

def universe(p,nodes):
    L=p.limit; sv=list(dict.fromkeys(p.sv)); cur=sv+[v for v in nodes]
    curset=set(cur); c1={}
    n=len(cur)
    for i in range(n):
        a=cur[i]
        for j in range(i,n):
            b=cur[j]
            s=a+b
            if s<=L and s not in curset: c1.setdefault(s,[]).append(('+',a,b))
            m=a*b
            if m<=L and m not in curset and a>1 and b>1: c1.setdefault(m,[]).append(('*',a,b))
            if a!=b:
                hi,lo=(a,b) if a>b else (b,a)
                d=hi-lo
                if d not in curset: c1.setdefault(d,[]).append(('-',hi,lo))
                if lo>1 and hi%lo==0:
                    q=hi//lo
                    if q not in curset: c1.setdefault(q,[]).append(('/',hi,lo))
    return cur,c1

def recipes_for(x,U,Uset,L):
    """all (op,a,b) with a,b in U giving x. U sorted np.int64"""
    out=[]
    # add: x=u+w, u<=w
    u=U[U<=x//2]; w=x-u; ok=np.isin(w,U,assume_unique=True); 
    for a,b in zip(u[ok].tolist(),w[ok].tolist()): out.append(('+',b,a))
    # sub: x=w-u
    u=U[U<=L-x]; w=x+u; ok=np.isin(w,U,assume_unique=True)
    for a,b in zip(u[ok].tolist(),w[ok].tolist()): out.append(('-',b,a))
    # mul: x=u*w, 1<u<=w
    u=U[(U>1)&(U*U<=x)] if x<3*10**18 else U[U>1]
    u=u[x%u==0]; w=x//u; ok=np.isin(w,U,assume_unique=True)&(w>=u)
    for a,b in zip(u[ok].tolist(),w[ok].tolist()): out.append(('*',b,a))
    # div: x=w/u
    u=U[(U>1)&(U<=L//x)]; w=x*u; ok=np.isin(w,U,assume_unique=True)
    for a,b in zip(u[ok].tolist(),w[ok].tolist()): out.append(('/',b,a))
    return out

def solve(p,prog,tl,verbose=1,seed=1):
    t0=time.time()
    nodes,order=wlib.build_dag(p,prog)
    n0=len(order); sv=set(p.sv)
    cur,c1=universe(p,nodes); curset=set(cur)
    U=np.array(sorted(curset|set(c1)),dtype=np.int64)
    rng=random.Random(seed)
    rec={}   # value -> list of (op,a,b)
    usednew=set()
    for x in order:
        rs=recipes_for(x,U,None,p.limit)
        # classify by number of new operands
        k0=[r for r in rs if r[1] in curset and r[2] in curset]
        k1=[r for r in rs if (r[1] in curset)!=(r[2] in curset)]
        k2=[r for r in rs if r[1] not in curset and r[2] not in curset]
        if len(k1)>CAP: k1=rng.sample(k1,CAP)
        K2=int(os.environ.get('SX_K2',str(CAP//3)))
        if len(k2)>K2: k2=rng.sample(k2,K2)
        rs=k0+k1+k2
        cr=nodes[x]
        if cr not in rs and (cr[0],cr[2],cr[1]) not in rs: rs.append(cr)
        rec[x]=rs
        for r in rs:
            for o in (r[1],r[2]):
                if o not in curset: usednew.add(o)
    for v in usednew: rec[v]=c1[v][:4]
    vals=list(rec)
    nrec=sum(len(r) for r in rec.values())
    if verbose: print(p.id,'n0',n0,'U',len(U),'usednew',len(usednew),'recipes',nrec,'gen %.1fs'%(time.time()-t0),flush=True)
    m=cp_model.CpModel()
    X={v:m.NewBoolVar('') for v in vals}
    for s in sv: X[s]=1

    T=set(p.targets)-sv
    ys={}
    # depth hint from current program
    dep={s:0 for s in sv}
    for v in order: dep[v]=1+max(dep[nodes[v][1]],dep[nodes[v][2]])
    D=min(n0,max(dep.values())+int(os.environ.get('SX_D','2')))
    Lv={v:m.NewIntVar(1,D,'') for v in vals}
    for s in sv: Lv[s]=0
    for v in vals:
        lits=[]
        for r in rec[v]:
            y=m.NewBoolVar(''); ys[(v,r)]=y; lits.append(y)
            a,b=r[1],r[2]
            if a not in sv: m.AddImplication(y,X[a]); 
            if b not in sv and b!=a: m.AddImplication(y,X[b])
            if a not in sv: m.Add(Lv[v]>=Lv[a]+1).OnlyEnforceIf(y)
            if b not in sv and b!=a: m.Add(Lv[v]>=Lv[b]+1).OnlyEnforceIf(y)
        if v in T: m.AddBoolOr(lits)
        else: m.AddBoolOr(lits+[X[v].Not()])
        if v in T: m.Add(X[v]==1)
    obj=sum(X[v] for v in vals)
    m.Add(obj<=n0-1)
    m.Minimize(obj)
    # hint
    for v in vals:
        inc = v in nodes
        m.AddHint(X[v],1 if inc else 0)
        if inc: m.AddHint(Lv[v],min(dep[v],D))
        for r in rec[v]:
            m.AddHint(ys[(v,r)],1 if inc and (r==nodes[v]) else 0)
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=tl; s.parameters.num_workers=1; s.parameters.log_search_progress=bool(os.environ.get('SX_LOG')); s.parameters.random_seed=seed
    st_=s.Solve(m)
    if verbose: print(p.id,'status',s.StatusName(st_),'obj',s.ObjectiveValue() if st_ in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None,'bound',s.BestObjectiveBound(),'t %.1f'%(time.time()-t0),flush=True)
    if st_ not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return None,s.StatusName(st_)
    nn={}
    for v in vals:
        if s.Value(X[v]):
            for r in rec[v]:
                if s.Value(ys[(v,r)]): nn[v]=r; break
    # prune to needed + order by level
    need=set(); stk=list(T)
    while stk:
        x=stk.pop()
        if x in need or x in sv: continue
        need.add(x); stk+=[nn[x][1],nn[x][2]]
    nn={v:nn[v] for v in need}
    od=wlib.dfs_order(p,nn,[t for t in order if t in T and t in nn]+[t for t in T if t not in order])
    od2=sorted(nn,key=lambda v:(s.Value(Lv[v]),v))
    pr=wlib.try_emit5(p,nn,[od,od2])
    return pr,s.StatusName(st_)

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
        else: print('none',pid,len(prog),status, 'emitfail' if status in('OPTIMAL','FEASIBLE') else '',flush=True)
        with open(R+'/sx_runs.jsonl','a') as f: f.write(json.dumps({'id':pid,'old':len(prog),'new':len(pr) if pr else None,'status':status,'tl':tl,'tag':TAG})+'\n')

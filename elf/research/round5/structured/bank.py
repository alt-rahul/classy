"""BANK: diverse structure-seeded restarts (public constructor + short LNS, true mids/hubs as extra targets) recombined
with the stored program by CP-SAT over ALL cross recipes between the pooled values.
usage: bank.py nseeds lns_tl solve_tl id...   (env BANK_TAG)"""
from lib import *
import subprocess, restart, copy, wlib, sx, time, numpy as np, pickle
from ortools.sat.python import cp_model
TH=100000; TAG=os.environ.get('BANK_TAG','bank1')
BDIR=os.path.join(R,'bank'); os.makedirs(BDIR,exist_ok=True)

def gen(p,prog,nseeds,tl):
    nodes,order=wlib.build_dag(p,prog)
    hubs=[v for v in order if v>=TH and nodes[v][1]<TH and nodes[v][2]<TH]
    mids=sorted({x for h in hubs for x in nodes[h][1:] if x not in p.sv})
    out=[]
    pad=prog+[prog[-1]]*150
    for sd in range(1,nseeds+1):
        for name,extra in (('hm',hubs+mids),('h',hubs),('plain',[])):
            if name=='plain' and sd>max(1,nseeds//3): continue
            q=copy.copy(p); q.targets=sorted(set(p.targets)|set(extra))
            r=subprocess.run(['nice','-n','10',restart.EXE,'--construct','2','--crand','3','--time',str(tl),'--seed',str(sd),'--kicks','1000','--kslack','2'],input=restart.enc(q,pad),capture_output=True,text=True)
            lines=r.stdout.strip().split('\n')[1:]
            if not lines: continue
            pr=[[l.split()[0]]+list(map(int,l.split()[1:])) for l in lines]
            if check(p,pr) is None:
                nn,od=wlib.build_dag(p,pr); out.append((name,sd,len(od),pr))
    return out

def recombine(p,prog,progs,tl,verbose=1):
    t0=time.time(); sv=set(p.sv)
    nodes,order=wlib.build_dag(p,prog); n0=len(order)
    U=set(nodes)
    for pr in progs:
        nn,od=wlib.build_dag(p,pr); U|=set(nn)
    Uarr=np.array(sorted(U|sv),dtype=np.int64)
    rec={v:sx.recipes_for(v,Uarr,None,p.limit) for v in U}
    for v in nodes:
        cr=nodes[v]
        if cr not in rec[v] and (cr[0],cr[2],cr[1]) not in rec[v]: rec[v].append(cr)
    vals=list(U); T=set(p.targets)-sv
    m=cp_model.CpModel(); X={v:m.NewBoolVar('') for v in vals}
    dep={s:0 for s in sv}
    for v in order: dep[v]=1+max(dep[nodes[v][1]],dep[nodes[v][2]])
    D=max(dep.values())+6
    Lv={v:m.NewIntVar(1,D,'') for v in vals}
    for s in sv: X[s]=1; Lv[s]=0
    ys={}
    for v in vals:
        lits=[]
        for r in rec[v]:
            y=m.NewBoolVar(''); ys[(v,r)]=y; lits.append(y); a,b=r[1],r[2]
            if a not in sv: m.AddImplication(y,X[a]); m.Add(Lv[v]>=Lv[a]+1).OnlyEnforceIf(y)
            if b not in sv and b!=a: m.AddImplication(y,X[b]); m.Add(Lv[v]>=Lv[b]+1).OnlyEnforceIf(y)
        if v in T: m.AddBoolOr(lits); m.Add(X[v]==1)
        else: m.AddBoolOr(lits+[X[v].Not()])
    obj=sum(X[v] for v in vals); m.Add(obj<=n0-1); m.Minimize(obj)
    for v in vals:
        inc=v in nodes; m.AddHint(X[v],1 if inc else 0)
        if inc: m.AddHint(Lv[v],dep[v])
        for r in rec[v]: m.AddHint(ys[(v,r)],1 if inc and r==nodes[v] else 0)
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=tl; s.parameters.num_workers=1
    st_=s.Solve(m); name=s.StatusName(st_)
    nrec=sum(len(r) for r in rec.values())
    if verbose: print(p.id,'recombine: values',len(vals),'recipes',nrec,'status',name,'obj',s.ObjectiveValue() if st_ in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None,'t %.1f'%(time.time()-t0),flush=True)
    if st_ not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return None,name
    nn={}
    for v in vals:
        if s.Value(X[v]):
            for r in rec[v]:
                if s.Value(ys[(v,r)]): nn[v]=r; break
    need=set(); stk=list(T)
    while stk:
        x=stk.pop()
        if x in need or x in sv: continue
        need.add(x); stk+=[nn[x][1],nn[x][2]]
    nn={v:nn[v] for v in need}
    od=sorted(nn,key=lambda v:(s.Value(Lv[v]),v))
    od1=wlib.dfs_order(p,nn,[t for t in order if t in T]+[t for t in T if t not in nodes])
    return wlib.try_emit5(p,nn,[od1,od]),name

if __name__=='__main__':
    ns=int(sys.argv[1]); ltl=float(sys.argv[2]); stl=float(sys.argv[3]); ids=sys.argv[4:]
    P={p.id:p for p in load_puzzles('scored')}
    for pid in ids:
        p=P[pid]; prog=load_store()[pid]['prog']; t0=time.time()
        bf=os.path.join(BDIR,pid+'.pkl')
        bank=pickle.load(open(bf,'rb')) if os.path.exists(bf) else []
        have={(b[0],b[1]) for b in bank}
        new=[b for b in gen(p,prog,ns,ltl) if (b[0],b[1]) not in have] if len(bank)<ns*2 else []
        bank+=new; pickle.dump(bank,open(bf,'wb'))
        tg=time.time()-t0
        pr,status=recombine(p,prog,[b[3] for b in bank],stl)
        res=len(pr) if pr is not None and check(p,pr) is None else None
        print(pid,'stored',len(prog),'bank',sorted(b[2] for b in bank),'gen %.0fs'%tg,'recomb',status,res,flush=True)
        if res is not None and res<len(prog):
            st=load_store()
            if offer(st,p,pr,TAG): save_store(st)
            print('GAIN',pid,len(prog),'->',res,flush=True)
        with open(R+'/bank_runs.jsonl','a') as f: f.write(json.dumps({'id':pid,'old':len(prog),'new':res,'bank':sorted(b[2] for b in bank),'status':status,'t':time.time()-t0})+'\n')

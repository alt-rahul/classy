from seeded import *
ps=[p for p in P() if len(p.targets)>=20 and p.family=='multiplicative']
sel=ps[0:38:8]
tot=[0,0,0]
for p in sel:
    t=time.time()
    base=regalloc(p.sv,run_steps(p),p.targets,p.nreg)
    F=facs(p.targets);xs=[];cnt={}
    for tt,f in zip(p.targets,F):
        x=max(f)
        if x>1000 and tt//x>1 and x not in p.targets: xs.append(x)
        for q in set(f): cnt[q]=cnt.get(q,0)+1
    sm=[q for q,c in cnt.items() if c>=2 and max(p.sv)<q<=10**4][:20]
    pB=clone(p,p.sv,list(p.targets)+xs)
    st=run_steps(pB);prog1=regalloc(p.sv,st,p.targets,p.nreg)
    prog2,a,b=seeded(p,sm,xs)
    r=[len(base),len(prog1) if prog1 else 9999,len(prog2) if prog2 else 9999]
    for i in range(3): tot[i]+=r[i]
    print(p.id,p.sv,len(p.targets),'base',r[0],'cofactors-as-targets',r[1],check(p,prog1) if prog1 else 'fail','+shared primes as starts',r[2],check(p,prog2) if prog2 else 'fail','nx',len(xs),'nsm',len(sm),'%.0fs'%(time.time()-t),flush=True)
print('TOTAL',tot)

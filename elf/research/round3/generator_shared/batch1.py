from seeded import *
ps=[p for p in P() if len(p.targets)>=20 and p.family in ('shared','multiplicative','mixed')]
sel=[p for p in ps if p.family=='shared'][1:40:5]+[p for p in ps if p.family=='multiplicative'][0:38:7]+[p for p in ps if p.family=='mixed'][0:44:11]
K=int(sys.argv[1])
tot=[0,0]
for p in sel:
    t=time.time()
    base=regalloc(p.sv,run_steps(p),p.targets,p.nreg)
    S=mine(p);sm=[v for v,w in sorted(S.items(),key=lambda x:-x[1])[:K]]
    prog,a,b=seeded(p,sm)
    ok=check(p,prog) if prog else 'fail'
    tot[0]+=len(base);tot[1]+=len(prog) if prog else len(base)
    print(p.id,p.family,p.sv,len(p.targets),'base',len(base),'seeded',len(prog) if prog else None,ok,'nsm',len(sm),'A',a,'%.0fs'%(time.time()-t),flush=True)
print('TOTAL',tot)

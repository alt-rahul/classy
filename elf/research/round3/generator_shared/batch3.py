import sys
sys.path.insert(0,'/home/user/classy/elf/research/round3/generator_meta')
from seeded import *
import seed_proto as SP
base={'practice_0015':214,'practice_0071':172,'practice_0084':151,'practice_0091':211}
ps={p.id:p for p in P()}
tot=[0,0,0]
for pid,b in base.items():
    p=ps[pid];t=time.time()
    H,G,Cs=SP.seeds(p)
    hubs=[x for x in H+G if x<=p.limit]
    st=run_steps(clone(p,p.sv,list(p.targets)+hubs));prog1=regalloc(p.sv,st,p.targets,p.nreg)
    S=mine(p);sm=[v for v,w in sorted(S.items(),key=lambda x:-x[1])[:20]]
    prog2,a,bb=seeded(p,sm,hubs)
    r=[b,len(prog1),len(prog2)]
    for i in range(3): tot[i]+=r[i]
    print(pid,'base',b,'hubs as targets + DCE',r[1],check(p,prog1),'hubs + mined constants as built starts',r[2],check(p,prog2),'nhubs',len(hubs),'%.0fs'%(time.time()-t),flush=True)
print('TOTAL',tot)

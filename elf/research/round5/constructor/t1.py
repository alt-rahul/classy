import sys, time
from crun import *
from prof import P, st
pid=sys.argv[1]; binary=sys.argv[2]
for a in sys.argv[3:]:
    p=P[pid]; t0=time.time(); prog,log=cons(p,[],a.split(','),binary=binary)
    costs=[int(l.split()[2]) for l in log.split('\n') if l.startswith('step cost')]
    print(pid,binary,a,'store',st[pid]['len'],'->',len(prog) if prog else None,'regs',(max(max(x[1:]) for x in prog)+1) if prog else None,costs,'%.0fs'%(time.time()-t0),flush=True)
    if not prog: print(log[-500:])
    elif len(prog)<st[pid]['len']:
        s2=load_store()
        if offer(s2,p,prog,'r5-cons'): save_store(s2); print('  STORED',pid,len(prog),flush=True)

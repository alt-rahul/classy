import sys, time
from crun import *
from prof import P, st
def cons_any(p, args, nreg=4000):
    inp='%d %d %s\n%d %s\n0\n'%(p.limit,len(p.sv),' '.join(map(str,p.sv)),len(p.targets),' '.join(map(str,p.targets)))
    r=subprocess.run(['nice',os.path.join(HERE,'cons')]+[str(a) for a in args],input=inp,capture_output=True,text=True)
    steps=[(f[1],int(f[2]),int(f[3]),int(f[4])) for f in (ln.split() for ln in r.stdout.split('\n')) if f and f[0]=='S']
    costs=[int(l.split()[2]) for l in r.stderr.split('\n') if l.startswith('step cost')]
    if not steps: return None,costs,None,r.stderr
    prog=solver.regalloc(p.sv,steps,p.targets,nreg)
    peak=max(max(x[1:]) for x in prog)+1 if prog else None
    return (len(prog) if prog else None),costs,peak,r.stderr
if __name__=='__main__':
    pid=sys.argv[1]; p=P[pid]
    for cap in sys.argv[2:]:
        t0=time.time(); L,costs,peak,log=cons_any(p,[42,0,0,1,1,0,6,34,1]+cap.split(','))
        if L is None: print(log[-300:])
        print(pid,'cap',cap,'len',L,'regs used',peak,'costs',costs,'time %.0fs'%(time.time()-t0),flush=True)

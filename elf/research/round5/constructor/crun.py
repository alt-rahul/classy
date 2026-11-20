import sys, time, subprocess
from rlib import *
sys.path.insert(0,'/home/user/classy/elf/src')
import solver
def cons(p, init, args, tl=3600, binary="cons"):
    if not init and len(set(p.sv))<4:
        sv=sorted(set(p.sv)); s0=sv[0]; init=[]; have=set(sv)
        def add(op,a,b):
            v={'+':a+b,'*':a*b,'/':a//b,'-':a-b}[op]
            if v not in have and 1<=v<=p.limit: init.append((op,a,b,v)); have.add(v)
        add('/',s0,s0)
        if len(sv)<3: add('+',1,1)
        if len(sv)<2: add('+',2,1); add('*',s0,s0)
    inp='%d %d %s\n%d %s\n%d\n'%(p.limit,len(p.sv),' '.join(map(str,p.sv)),len(p.targets),' '.join(map(str,p.targets)),len(init))
    inp+=''.join('%s %d %d %d\n'%s for s in init)
    r=subprocess.run(['nice',os.path.join(HERE,binary)]+[str(a) for a in args],input=inp,capture_output=True,text=True,timeout=tl)
    steps=[]
    for ln in r.stdout.split('\n'):
        f=ln.split()
        if f and f[0]=='S': steps.append((f[1],int(f[2]),int(f[3]),int(f[4])))
    if r.returncode!=0 or not steps: return None, r.stderr
    prog=solver.regalloc(p.sv,steps,p.targets,p.nreg)
    if prog is None: return None, r.stderr+'\nREGALLOC FAIL'
    err=check(p,prog)
    if err: return None, r.stderr+'\nINVALID '+err
    return prog, r.stderr
if __name__=='__main__':
    from prof import prof, P, st
    pid=sys.argv[1]; K=int(sys.argv[2]); args=sys.argv[3:]
    p=P[pid]; init=[]
    if K>0:
        L,n,rc,nodes,order,reach=prof(pid); T=set(p.targets)
        core=[v for v in order if v not in T and len(reach[v])>=K]
        init=[(nodes[v][0],nodes[v][1],nodes[v][2],v) for v in core]
    t0=time.time(); prog,log=cons(p,init,args)
    costs=[int(l.split()[2]) for l in log.split('\n') if l.startswith('step cost')]
    print(pid,'store',st[pid]['len'],'init',len(init),'costs',costs,'->',len(prog) if prog else None,'time %.0fs'%(time.time()-t0))
    if not prog: print(log[-600:])

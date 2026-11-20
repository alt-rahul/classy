import sys, subprocess, json, time
from rlib import *
SP='/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad'
def elf(p, prog, secs, seed=1, extra=()):
    inp='P %s %d %s %d %s %d %d %d\n'%(p.id,len(p.sv),' '.join(map(str,p.sv)),len(p.targets),' '.join(map(str,p.targets)),p.limit,p.nreg,len(prog))
    inp+=''.join('%s %d %d %d\n'%tuple(x) for x in prog)
    r=subprocess.run(['nice',SP+'/elfopt','--time',str(secs),'--seed',str(seed)]+list(extra),input=inp,capture_output=True,text=True)
    out=None; lines=r.stdout.split('\n')
    for k,ln in enumerate(lines):
        if ln.startswith('R '):
            n=int(ln.split()[3]); out=[]
            for l2 in lines[k+1:k+1+n]:
                f=l2.split(); out.append([f[0],int(f[1]),int(f[2]),int(f[3])])
    if out is not None and check(p,out) is None and len(out)<len(prog): return out
    return prog
if __name__=='__main__':
    import bench
    pid=sys.argv[1]; secs=float(sys.argv[2])
    res,dt,err=bench.run_beam(SP+'/beam_pub',[pid],[3,0.18,52,56,20,1])
    prog=res[pid][1]; p=bench.P[pid]
    print('raw',len(prog))
    for it in range(int(sys.argv[3])):
        prog=elf(p,prog,secs,seed=it+1); print('after',(it+1)*secs,'s:',len(prog),flush=True)

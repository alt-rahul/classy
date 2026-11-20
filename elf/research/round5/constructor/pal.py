import sys, time
from crun import *
from prof import P, st
def palette(p, smalls):
    s0=p.sv[0]; init=[]; have=set(p.sv)
    def add(op,a,b):
        v={'+':a+b,'*':a*b,'/':a//b,'-':a-b}[op]
        if v not in have: init.append((op,a,b,v)); have.add(v)
        return v
    if smalls: add('/',s0,s0)
    for v in smalls:
        if v in have: continue
        # build v as a+b from existing smalls
        ok=False
        for a in sorted(have):
            if v-a in have and a<=v-a: add('+',a,v-a); ok=True; break
        if not ok:
            for a in sorted(have):
                if a>1 and v%a==0 and v//a in have: add('*',a,v//a); ok=True; break
        assert ok,(v,sorted(have)[:10])
    return init
if __name__=='__main__':
    pid=sys.argv[1]; p=P[pid]; args='42,0,0,1,1,0,6,34,1,47,200'.split(',')
    for spec in sys.argv[2:]:
        sm=[int(x) for x in spec.split(',') if x]
        init=palette(p,sm); t0=time.time(); prog,log=cons(p,init,args,binary='cons')
        print(pid,'palette',sm,'init ops',len(init),'->',len(prog) if prog else None,'store',st[pid]['len'],'%.0fs'%(time.time()-t0),flush=True)

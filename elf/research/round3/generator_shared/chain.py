from lib import *
import sys
def fwd(sv,lim,D,selfop=True,cap=3_000_000):
    lev=[set(sv)];seen={v:None for v in sv}
    for d in range(D):
        new={}
        for x in lev[-1]:
            cands=[]
            for s in sv:
                cands+=[(x+s,'+',s),(x*s,'*',s)]
                if x>s: cands.append((x-s,'-',s))
                if s>x: cands.append((s-x,'r-',s))
                if x%s==0: cands.append((x//s,'/',s))
                if s%x==0: cands.append((s//x,'r/',s))
            if selfop: cands+=[(x+x,'+','x'),(x*x,'*','x')]
            for v,o,s in cands:
                if 0<v<=lim and v not in seen and v not in new: new[v]=(x,o,s)
        seen.update(new);lev.append(set(new))
        if len(seen)>cap: break
    return lev,seen
def path(v,seen):
    out=[]
    while seen[v] is not None:
        x,o,s=seen[v];out.append('%s%s'%(o,s));v=x
    return str(v)+' '+' '.join(reversed(out))
if __name__=='__main__':
    ps={p.id:p for p in P()}
    p=ps[sys.argv[1]];D=int(sys.argv[2])
    lev,seen=fwd(p.sv,p.limit,D)
    print(p.family,p.sv,p.limit,[len(l) for l in lev])
    for t in p.targets:
        d=next((i for i,l in enumerate(lev) if t in l),None)
        print(t,d,path(t,seen) if d is not None else '')

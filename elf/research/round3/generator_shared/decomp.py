from lib import *
from common import check
def decomp(p,prog,show=True):
    """print program as SSA with names; returns list of (value, op, a, b)"""
    reg={i:('s%d'%i,v) for i,v in enumerate(p.sv)}
    T=set(p.targets);lines=[];n=0
    uses={}
    ssa=[]
    for op,i,j,d in prog:
        (na,a),(nb,b)=reg[i],reg[j]
        v={'+':a+b,'-':a-b,'*':a*b,'/':a//b if b else 0}[op]
        nm='v%d'%n;n+=1
        uses[na]=uses.get(na,0)+1;uses[nb]=uses.get(nb,0)+1
        ssa.append((nm,op,na,nb,v,a,b))
        reg[d]=(nm,v)
    if show:
        for nm,op,na,nb,v,a,b in ssa:
            tg=('T%d'%p.targets.index(v)) if v in T else ''
            print('%s = %s(%d) %s %s(%d) = %d  %s uses=%d'%(nm,na,a,op,nb,b,v,tg,uses.get(nm,0)))
    return ssa
if __name__=='__main__':
    import sys,json
    sys.path.insert(0,'/home/user/classy/elf/src')
    from solver import solve,Cfg
    ps={p.id:p for p in P()}
    p=ps[sys.argv[1]]
    kw=json.loads(sys.argv[2]) if len(sys.argv)>2 else {"pool_cap":46,"evict_size":1,"regret":0.1}
    prog,st=solve(p,Cfg(**kw))
    print(len(prog),check(p,prog))
    json.dump(prog,open('prog_%s.json'%p.id,'w'))
    decomp(p,prog)

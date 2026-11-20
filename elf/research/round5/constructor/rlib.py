import sys, os, subprocess, collections, random
sys.path.insert(0,'/home/user/classy/elf/research/round4/beyond')
from dagutil import *
HERE=os.path.dirname(os.path.abspath(__file__))
def state_at(p, prog, t):
    regs=list(p.sv)+[None]*(p.nreg-len(p.sv)); done=set(p.sv)
    for op,i,j,d in prog[:t]:
        regs[d]=OPF[op](regs[i],regs[j]); done.add(regs[d])
    pool=sorted(set(r for r in regs if r is not None))
    rem=[x for x in p.targets if x not in done]
    return pool, rem
def enum(limit, pool, targets, cmax=8, nb=64, rbits=34):
    inp='%d %d %d %d %d\n%s\n%d\n%s\n'%(limit,cmax,nb,rbits,len(pool),' '.join(map(str,pool)),len(targets),' '.join(map(str,targets)))
    r=subprocess.run([os.path.join(HERE,'xyr')],input=inp,capture_output=True,text=True)
    D={}; R=collections.defaultdict(list)
    for ln in r.stdout.split('\n'):
        f=ln.split()
        if not f: continue
        if f[0]=='D': D[int(f[1])]=(f[4],int(f[3]),int(f[5]))
        else: R[int(f[1])].append((int(f[2]),int(f[3]),int(f[4]),int(f[5]),f[6]))
    return D,R,r.stderr
def nodes_of(D, pool, T, rec):
    """list of ops (op,a,b,v) for recipe, new values only"""
    cost,x,y,r,sg=rec; out={}
    def need(v):
        if v in pool or v in out or v==0: return
        op,a,b=D[v]; need(a); need(b); out[v]=(op,a,b)
    need(x); 
    if y: need(y)
    if r: need(r)
    if y==0:
        out[T]=('+',x,r) if sg=='+' else ('-',x,r)
    elif sg=='0': out[T]=('*',x,y)
    else:
        pr=x*y
        if pr not in pool and pr not in out: out[pr]=('*',x,y)
        out[T]=('+',pr,r) if sg=='+' else ('-',pr,r)
    return out
def select(D, pool, R, targets, sweeps=6, seed=0, verbose=False):
    """choose one recipe per target minimising the union of new nodes. returns (total, choice dict T->nodes dict)"""
    rnd=random.Random(seed); pool=set(pool)
    NS={T:[frozenset(nodes_of(D,pool,T,rec).keys()) for rec in R[T]] for T in targets}
    pop=collections.Counter()
    for T in targets:
        s=set()
        for ns in NS[T]: s|=ns
        pop.update(s)
    ref=collections.Counter(); ch={}
    order=sorted(targets,key=lambda T:min(len(ns) for ns in NS[T]))
    for T in order:
        best=min(range(len(NS[T])),key=lambda k:(sum(1 for v in NS[T][k] if ref[v]==0), -sum(pop[v] for v in NS[T][k])))
        ch[T]=best; ref.update(NS[T][best])
    def total(): return sum(1 for v in ref if ref[v]>0)
    hist=[total()]
    for sw in range(sweeps):
        ts=list(targets); rnd.shuffle(ts)
        for T in ts:
            ref.subtract(NS[T][ch[T]])
            best=min(range(len(NS[T])),key=lambda k:(sum(1 for v in NS[T][k] if ref[v]<=0), -sum(min(pop[v],3) for v in NS[T][k] if ref[v]<=0), rnd.random()))
            ch[T]=best; ref.update(NS[T][best])
        hist.append(total())
    return total(), {T:nodes_of(D,pool,T,R[T][ch[T]]) for T in targets}, hist

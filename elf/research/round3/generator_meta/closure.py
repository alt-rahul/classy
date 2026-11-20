"""How many targets are exactly one op / two ops from (targets U starts)?"""
import sys,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
def one_op(K,L):
    out={}
    K=sorted(K)
    for i,a in enumerate(K):
        for b in K[:i+1]:
            for v,op in ((a+b,'+'),(a-b,'-'),(a*b,'*'),(a//b if a%b==0 else 0,'/')):
                if 0<v<=L and v not in out: out[v]=(a,op,b)
    return out
res=collections.defaultdict(lambda:[0,0,0,0])
for p in load_puzzles('practice'):
    if p.family in('tiny',) or len(p.targets)<20: continue
    K=set(p.sv)|set(p.targets); T=set(p.targets)
    o1=one_op(K,p.limit)
    h1=[t for t in T if t in o1 and o1[t][0]!=t and o1[t][2]!=t]
    # two ops: t = x op k, x in o1
    h2=set()
    Ks=sorted(K)
    for t in T:
        if t in h1: continue
        for k in Ks:
            if k==t: continue
            for x in (t-k,t+k,k-t,t*k,(t//k if t%k==0 else 0),(k//t if k%t==0 else 0)):
                if x>0 and x<=p.limit and x in o1 and x not in T and t not in (o1[x][0],o1[x][2]): h2.add(t); 
    r=res[p.family]; r[0]+=len(T); r[1]+=len(h1); r[2]+=len(h2); r[3]+=1
    if len(h1)+len(h2)>=6: print(p.id,p.family,len(T),len(h1),len(h2),[ (t,o1[t]) for t in h1][:4])
for f,r in res.items(): print(f,r,'one-op frac %.3f two-op %.3f'%(r[1]/r[0],r[2]/r[0]))

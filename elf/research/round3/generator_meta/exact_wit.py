"""exact tree cost for all v<=L (small L) plus all top-level min decompositions of targets"""
import sys, numpy as np
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from exact_small import costs
def decomp(v,cost,L,S):
    """all (op,a,b) with cost[a]+cost[b]+1==cost[v]"""
    k=int(cost[v]); out=[]
    if k==0: return out
    for i in range(0,k):
        j=k-1-i
        A=S[i]
        if len(A)==0: continue
        # v=a+b
        b=v-A; m=(b>0)&(b>=A); bb=b[m]; aa=A[m]; ok=cost[bb]==j
        out+=[('+',int(x),int(y)) for x,y in zip(aa[ok],bb[ok])]
        b=A-v; m=b>0; bb=b[m]; aa=A[m]; ok=cost[bb]==j   # v=a-b
        out+=[('-',int(x),int(y)) for x,y in zip(aa[ok],bb[ok])]
        a2=A+v; m=a2<=L; aa=a2[m]; bb=A[m]; ok=cost[aa]==j  # v=a2-A, A has cost i
        out+=[('-',int(x),int(y)) for x,y in zip(aa[ok],bb[ok])]
        m=(v%A==0); aa=A[m]; bb=v//aa; ok=(cost[bb]==j)&(bb>=aa)
        out+=[('*',int(x),int(y)) for x,y in zip(aa[ok],bb[ok])]
        m=(A<=L//v); bb=A[m]; aa=bb*v; ok=cost[aa]==j   # v = aa / bb, bb cost i
        out+=[('/',int(x),int(y)) for x,y in zip(aa[ok],bb[ok])]
        m=(A%v==0); aa=A[m]; bb=aa//v; ok=cost[bb]==j   # v = aa/bb, aa cost i
        out+=[('/',int(x),int(y)) for x,y in zip(aa[ok],bb[ok])]
    return sorted(set(out))
def expr(v,cost,L,S,memo):
    if cost[v]==0: return str(v)
    if v in memo: return memo[v]
    d=decomp(v,cost,L,S)
    op,a,b=d[0]
    memo[v]='(%s%s%s)'%(expr(a,cost,L,S,memo),op,expr(b,cost,L,S,memo)); return memo[v]
if __name__=='__main__':
    which=sys.argv[1]; pid=sys.argv[2]; maxk=int(sys.argv[3]) if len(sys.argv)>3 else 8
    for p in load_puzzles(which):
        if p.id[-4:]==pid:
            print(p.id,p.family,p.sv,p.limit)
            c=costs(p.sv,p.limit,maxk)
            S=[np.nonzero(c==k)[0].astype(np.int64) for k in range(maxk+1)]
            S[0]=S[0][S[0]>0]
            memo={}
            for t in p.targets:
                if c[t]==99: print(t,'cost>%d'%maxk); continue
                d=decomp(t,c,p.limit,S)
                print(t,'cost',c[t],'ndec',len(d),expr(t,c,p.limit,S,memo))
                for op,a,b in d[:12]: print('      ',a,op,b,' costs',c[a],c[b])

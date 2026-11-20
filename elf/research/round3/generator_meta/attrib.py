import sys,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from fam import fam,decade
def marginal(p,prog):
    """marginal op count per target in order of first appearance (ancestor ops not already counted)"""
    reg={i:('s',i) for i in range(len(p.sv))}; val={i:v for i,v in enumerate(p.sv)}
    par=[]; vals=[]
    for k,(op,i,j,d) in enumerate(prog):
        a,b=val[i],val[j]
        v={'+':a+b,'-':a-b,'*':a*b,'/':a//b}[op]
        par.append((reg[i],reg[j])); vals.append(v); reg[d]=('o',k); val[d]=v
    T=set(p.targets); seen=set(); counted=set(); res={}
    for t in p.targets:
        if t in p.sv: res[t]=0
    for k,v in enumerate(vals):
        if v in T and v not in res:
            st=[('o',k)]; c=0
            while st:
                x=st.pop()
                if x[0]=='s' or x[1] in counted: continue
                counted.add(x[1]); c+=1; st+=list(par[x[1]])
            res[v]=c
    return res,len(prog)-len(counted)
if __name__=='__main__':
    st=load_store(); A=collections.defaultdict(list)
    for p in load_puzzles('scored'):
        f=fam(p)
        if len(p.targets)<20: continue
        res,dead=marginal(p,st[p.id]['prog'])
        if f=='mixed':
            k,r=decade(p); r=set(r)
            for t,c in res.items(): A[('mixed%d'%k,'rand' if t in r else 'struct')].append(c)
        elif f=='indep':
            k,r=decade(p)
            for t,c in res.items(): A[('indep%d'%k,'rand')].append(c)
        else:
            for t,c in res.items(): A[(f,'struct')].append(c)
        A[(f,'dead')].append(dead)
    for k in sorted(A): print(k,len(A[k]),sum(A[k]),'%.2f'%(sum(A[k])/len(A[k])))

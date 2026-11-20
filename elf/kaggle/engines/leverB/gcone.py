import sys, json, itertools, collections
from wlib import *
def dsets(p, nodes, order):
    sv=set(p.sv); T=set(p.targets)-sv
    users=collections.defaultdict(set)
    for v,(op,a,b) in nodes.items(): users[a].add(v); users[b].add(v)
    D={}
    for v in reversed(order):
        s=set([v]) if v in T else set()
        for w in users[v]: s|=D[w]
        D[v]=frozenset(s)
    return D
def groups(p, nodes, order, thr, maxk=4):
    D=dsets(p,nodes,order)
    cnt=collections.Counter(D.values())
    cands=[A for A in cnt if 2<=len(A)<=maxk]
    # add unions of pairs of candidate sets (size<=maxk)
    ext=set(cands)
    for A,B in itertools.combinations(cands,2):
        U=A|B
        if len(U)<=maxk: ext.add(U)
    out=[]
    for A in ext:
        jc=sum(c for B,c in cnt.items() if B<=A)
        if jc>thr*len(A): out.append((jc-thr*len(A),jc,A))
    out.sort(key=lambda x:-x[0])
    return out, D
if __name__=='__main__':
    P={p.id:p for p in load_puzzles()}; st=load_store()
    for name,thr in [('i40',5),('i50',6),('i60',7),('i60',8),('m50',6),('m60',7),('m60',8),('s60',7),('s60',8)]:
        tot=0; bysize=collections.Counter()
        for pid in open('ids_%s.txt'%name).read().split(','):
            p=P[pid]; nodes,order=build_dag(p,st[pid]['prog'])
            g,D=groups(p,nodes,order,thr)
            used=set()
            for gain,jc,A in g:
                if A&used: continue
                used|=A; tot+=gain; bysize[len(A)]+=gain
        print(name,'thr',thr,'potential',tot,dict(bysize))

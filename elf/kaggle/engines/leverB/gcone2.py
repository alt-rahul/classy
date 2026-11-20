import sys, json, itertools, collections
from wlib import *
from gcone import dsets
def best_groups(p, nodes, order, thr, maxk=8):
    D=dsets(p,nodes,order)
    sv=set(p.sv); T=sorted(set(p.targets)-sv)
    cnt=collections.Counter(D.values())
    items=list(cnt.items())
    def jc(A): return sum(c for B,c in items if B and B<=A)
    res=[]
    seeds=[A for A in cnt if 1<=len(A)<=3]
    seen=set()
    for A in seeds:
        A=frozenset(A)
        while len(A)<=maxk:
            if A not in seen:
                seen.add(A); v=jc(A)
                if v>thr*len(A): res.append((v-thr*len(A),v,A))
            if len(A)==maxk: break
            # grow by best target
            best=None
            for t in T:
                if t in A: continue
                B=A|{t}; g=jc(B)-thr*len(B)
                if best is None or g>best[0]: best=(g,B)
            A=best[1]
    res.sort(key=lambda x:-x[0])
    return res
if __name__=='__main__':
    P={p.id:p for p in load_puzzles()}; st=load_store()
    for name,thr,mk in [('i50',6,6),('i50',6,10),('i50',7,10),('i60',7,10),('i60',8,10),('i40',5,10),('i40',6,10)]:
        tot=0; bysize=collections.Counter(); n=0
        for pid in open('ids_%s.txt'%name).read().split(',')[:20]:
            p=P[pid]; nodes,order=build_dag(p,st[pid]['prog'])
            g=best_groups(p,nodes,order,thr,mk)
            used=set(); n+=1
            for gain,jc,A in g:
                if A&used: continue
                used|=A; tot+=gain; bysize[len(A)]+=gain
        print(name,'thr',thr,'maxk',mk,'puzzles',n,'potential',tot,dict(sorted(bysize.items())),flush=True)

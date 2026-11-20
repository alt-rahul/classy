import sys, json, itertools, collections
from wlib import *
fam=json.load(open('fam.json'))
def joint_cone(nodes, sv, T, users, roots):
    cone=set(roots); stack=list(roots)
    while stack:
        x=stack.pop(); op,a,b=nodes[x]
        for u in (a,b):
            if u in cone or u in sv or u in T: continue
            if users[u]<=cone: cone.add(u); stack.append(u)
    # fixpoint: nodes whose users all in cone may appear after others get added
    changed=True
    while changed:
        changed=False
        for x in list(cone):
            op,a,b=nodes[x]
            for u in (a,b):
                if u in cone or u in sv or u in T: continue
                if users[u]<=cone: cone.add(u); changed=True
    return cone
if __name__=='__main__':
    P=load_puzzles(); st=load_store()
    name=sys.argv[1]; thr=int(sys.argv[2])
    ids=open('ids_%s.txt'%name).read().split(',')
    PP={p.id:p for p in P}
    tot=0; hist=collections.Counter(); npz=0
    for pid in ids:
        p=PP[pid]; nodes,order=build_dag(p,st[pid]['prog'])
        sv=set(p.sv); T=set(p.targets)-sv
        users=collections.defaultdict(set)
        for v,(op,a,b) in nodes.items(): users[a].add(v); users[b].add(v)
        single={t:len(joint_cone(nodes,sv,T,users,[t])) for t in T}
        best=[]
        for i,j in itertools.combinations(sorted(T),2):
            c=len(joint_cone(nodes,sv,T,users,[i,j]))
            ex=c-single[i]-single[j]
            if ex>0: hist[(c)]+=1
            if c>2*thr: best.append((c-2*thr,i,j))
        # greedy disjoint
        best.sort(reverse=True); used=set(); g=0
        for d,i,j in best:
            if i in used or j in used: continue
            used|={i,j}; g+=d
        tot+=g; npz+=1
    print(name,'puzzles',npz,'potential gain if every target rebuildable at',thr,':',tot,'pair joint-cone hist (pairs with shared-exclusive nodes):',sorted(hist.items()))

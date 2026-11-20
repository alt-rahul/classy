import sys, itertools, pickle
from wlib import *
from jcone import joint_cone
from jrun import desc_of
P={p.id:p for p in load_puzzles()}; st=load_store()
pid=sys.argv[1]; p=P[pid]; prog=st[pid]['prog']
nodes,order=build_dag(p,prog); sv=set(p.sv); T=set(p.targets)-sv
users=collections.defaultdict(set)
for v,(op,a,b) in nodes.items(): users[a].add(v); users[b].add(v)
best=None
for i,j in itertools.combinations(sorted(T),2):
    jc=joint_cone(nodes,sv,T,users,[i,j])
    if len(jc)>=13 and (best is None or len(jc)>len(best[2])): best=(i,j,jc)
i,j,jc=best
dsc=desc_of(users,jc)
pool=[v for v in list(dict.fromkeys(p.sv))+order if v not in jc and v not in dsc]
si,_=run_wide_enum(p.limit,i,pool,5,7,1,12,12); sj,_=run_wide_enum(p.limit,j,pool,5,7,1,12,12)
print('orig pressure',pressure(p,nodes,order),'sols',[len(s) for s in si],[len(s) for s in sj])
pickle.dump((pid,i,j,jc,si,sj),open('/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad/dbg_%s.pkl'%pid,'wb'))
for a in si:
    for b in sj:
        nn,c=splice_blocks(p,nodes,order,jc,[a,b]); nn2,c2=splice_blocks(p,nodes,order,jc,[b,a])
        print(len(nn),[pressure(p,nn,x) for x in c],[pressure(p,nn2,x) for x in c2])

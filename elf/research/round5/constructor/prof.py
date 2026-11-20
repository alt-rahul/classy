# H1 profile: reach (number of targets depending on a value), exclusive cones, magnitude of pool values
import sys, collections
from rlib import *
sys.path.insert(0,'/home/user/classy/elf/research/round4/beyond')
import wlib
st=load_store(); P={p.id:p for p in load_puzzles('scored')}
def prof(pid, verbose=False):
    p=P[pid]; prog=st[pid]['prog']; nodes,order=wlib.build_dag(p,prog); sv=set(p.sv); T=set(p.targets)-sv
    reach=collections.defaultdict(set)
    for t in T:
        stk=[t]; seen=set()
        while stk:
            x=stk.pop()
            if x in seen or x in sv: continue
            seen.add(x); reach[x].add(t); op,a,b=nodes[x]; stk+=[a,b]
    rc=collections.Counter()
    for v in order:
        if v in T: continue
        k=len(reach[v]); rc['1' if k==1 else '2' if k==2 else '3-5' if k<=5 else '6-12' if k<=12 else '13+']+=1
    return len(prog), len(T), rc, nodes, order, reach
if __name__=='__main__':
    for pid in sys.argv[1:]:
        L,n,rc,nodes,order,reach=prof(pid)
        print(pid,L,n,'mids by reach',dict(sorted(rc.items())), 'direct-use fanout:')

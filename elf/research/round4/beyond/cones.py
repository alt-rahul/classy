from dagutil import *
import json, collections
fam=json.load(open('fam.json'))
def excl_cones(p,prog):
    """returns dict target -> exclusive cone size (ops needed only by that target incl. itself), based on value DAG"""
    nodes=dag(p,prog)
    T=[t for t in set(p.targets) if nodes[t][0] is not None]
    # users
    users=collections.defaultdict(set)
    for v,(op,a,b,k) in nodes.items():
        if op: users[a].add(v); users[b].add(v)
    Ts=set(T); res={}
    for t in T:
        cone={t}; 
        # iteratively add operands whose all users are in cone and that are not targets/starts
        stack=[t]
        while stack:
            x=stack.pop(); op,a,b,k=nodes[x]
            for u in (a,b):
                if u in cone or nodes[u][0] is None or u in Ts: continue
                if users[u]<=cone: cone.add(u); stack.append(u)
        # if target itself is used by others, still count (it must exist anyway)
        res[t]=len(cone)
    return res
if __name__=='__main__':
    P=load_puzzles(); st=load_store()
    agg=collections.defaultdict(collections.Counter); tot=collections.Counter(); gain6=collections.Counter(); gain5=collections.Counter()
    for p in P:
        if len(p.targets)<20: continue
        k=(fam[p.id],min(p.limit.bit_length()//10*10,60))
        r=excl_cones(p,st[p.id]['prog'])
        for t,c in r.items():
            agg[k][min(c,12)]+=1
            gain6[k]+=max(0,c-6); gain5[k]+=max(0,c-5)
        tot[k]+=st[p.id]['len']
    for k in sorted(agg):
        n=sum(agg[k].values())
        print(k,'ops',tot[k],'targets',n,'cone hist',' '.join('%d:%d'%(c,agg[k][c]) for c in sorted(agg[k])),'| gain if all cones<=6:',gain6[k],' <=5:',gain5[k])
    print(sum(gain6.values()),sum(gain5.values()))

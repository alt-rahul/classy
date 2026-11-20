from lib import *
from collections import Counter,defaultdict
import statistics as S
st=load_store(); P=load_puzzles('scored')
rows=json.load(open(R+'/cone_rows.json')); ml={r[0]:r for r in rows if r[2]}
out=[]
def closure(d,ts):
    seen=set(); stack=list(ts)
    while stack:
        v=stack.pop()
        if v in seen or v not in d: continue
        seen.add(v); stack+=[d[v][1],d[v][2]]
    return seen
for p in P:
    if p.id not in ml: continue
    d=defs(p,st[p.id]['prog'])
    c=Counter(len(str(t)) for t in p.targets); k,n=c.most_common(1)[0]
    Rn=[t for t in p.targets if len(str(t))==k]; Sn=[t for t in p.targets if len(str(t))!=k]
    full=len(closure(d,p.targets)); cs=len(closure(d,Sn)); cr=len(closure(d,Rn))
    out.append((p.id,FAM[p.id],k,len(Sn),len(Rn),full,cs,cr))
agg=defaultdict(list)
for r in out: agg[(r[1],r[2])].append(r)
print('fam decade n | nS nR total | closure(S)/nS  marg(R)/nR | closure(R)/nR marg(S)/nS')
for k in sorted(agg):
    rs=agg[k]; nS=sum(r[3] for r in rs); nR=sum(r[4] for r in rs); F=sum(r[5] for r in rs); cs=sum(r[6] for r in rs); cr=sum(r[7] for r in rs)
    print(k,len(rs),'|',nS,nR,F,'| %.2f %.2f | %.2f %.2f'%(cs/nS,(F-cs)/nR,cr/nR,(F-cr)/nS))
json.dump(out,open(R+'/mixsplit.json','w'))

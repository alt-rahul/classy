from lib import *
from collections import Counter,defaultdict
import statistics as S
st=load_store(); P=load_puzzles('scored')
rows=json.load(open(R+'/cone_rows.json')); pure={r[0]:r for r in rows if not r[2]}
out=[]
for p in P:
    if p.id not in pure: continue
    d=defs(p,st[p.id]['prog']); T=set(p.targets)
    th=min(t for t in p.targets if t>=10**6)//50 if any(t>=10**6 for t in p.targets) else 10**6
    th=max(th,10**4)
    small={v for v in d if v<th}
    big={v for v in d if v>=th}
    need=set()   # smalls used directly by big ops
    for v in big:
        op,a,b,k=d[v]
        for x in (a,b):
            if x in small: need.add(x)
    need|= (small&T)
    bigT=big&T; bigN=big-T
    # hubs: big non-target built from two smalls
    hubs=[v for v in big if d[v][1] not in big and d[v][2] not in big]
    out.append((p.id,pure[p.id][1],len(p.targets),len(d),len(small),len(need),len(small)-len(need),len(hubs),len(bigN)-len([h for h in hubs if h not in T]),len(bigT)))
agg=defaultdict(list)
for r in out: agg[r[1]].append(r)
print('fam n  ops smalls needed aux hubs bigInter bigT')
for f,rs in agg.items():
    print(f,len(rs),[round(S.mean(r[i] for r in rs),1) for i in range(3,10)])
json.dump(out,open(R+'/infra_rows.json','w'))

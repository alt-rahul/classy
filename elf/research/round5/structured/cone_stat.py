from lib import *
from collections import Counter,defaultdict
st=load_store(); P=load_puzzles('scored')
H=defaultdict(Counter); rows=[]
for p in P:
    f=FAM[p.id]
    if f in('independent','tiny') or len(p.targets)<20: continue
    c,n=cones(p,st[p.id]['prog'])
    k,nd=mixed_like(p)
    ml = nd>=12 and k in (12,15,18)
    key=(f,'mixedlike' if ml else 'pure')
    for t,v in c.items(): H[key][min(v,9)]+=1
    rows.append((p.id,f,ml,len(p.targets),st[p.id]['len'],sum(1 for v in c.values() if v>=5),sum(max(0,v-4) for v in c.values()),k,nd,max(p.targets).bit_length(),len(p.sv)))
for k in sorted(H): print(k,sorted(H[k].items()), sum(H[k].values()))
json.dump(rows,open(R+'/cone_rows.json','w'))
agg=defaultdict(lambda:[0,0,0,0,0])
for r in rows:
    a=agg[(r[1],r[2])]; a[0]+=1;a[1]+=r[3];a[2]+=r[4];a[3]+=r[5];a[4]+=r[6]
for k in sorted(agg): print(k,agg[k],round(agg[k][2]/agg[k][1],2))

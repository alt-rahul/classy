from lib import *
import random, statistics as S
from mixsplit import closure
st=load_store(); P=load_puzzles('scored')
rows=json.load(open(R+'/cone_rows.json')); pure={r[0]:r for r in rows if not r[2]}
random.seed(1); res={}
for p in P:
    if p.id not in pure: continue
    d=defs(p,st[p.id]['prog']); f=FAM[p.id]; b=max(p.targets).bit_length()
    for k in (14,):
        vals=[]
        for _ in range(10):
            sub=random.sample(p.targets,k); vals.append(len(closure(d,sub))/k)
        res.setdefault((f,(b+5)//10*10),[]).append(S.mean(vals))
for k in sorted(res): print(k,len(res[k]),'closure(14 random targets)/14 = %.2f'%S.mean(res[k]))

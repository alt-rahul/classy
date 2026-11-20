from rlib import *
import collections
st=load_store(); PZ=load_puzzles('scored')
rows=collections.defaultdict(list)
for p in PZ:
    if max(p.sv)<=40 or p.limit<=16: continue
    b=max(p.targets).bit_length(); n=len(set(p.targets)-set(p.sv))
    bb=40 if 36<=b<=43 else 50 if 46<=b<=53 else 60 if b>=56 else 30 if 26<=b<=33 else 20 if 16<=b<=23 else None
    if bb: rows[bb].append((n,st[p.id]['len'],len(p.sv)))
for bb in sorted(rows):
    r=rows[bb]; by=collections.defaultdict(list)
    for n,L,s in r: by[n//4*4].append(L/n)
    print(bb,'bits', 'puzzles',len(r),' n-bucket: mean ops/target (count):',' '.join('%d:%.2f(%d)'%(k,sum(v)/len(v),len(v)) for k,v in sorted(by.items())))
    import numpy as np
    A=np.array([[1,n] for n,L,s in r],float); y=np.array([L for n,L,s in r],float)
    c=np.linalg.lstsq(A,y,rcond=None)[0]; print('   fit L = %.1f + %.2f n'%tuple(c), ' n range',min(x[0] for x in r),max(x[0] for x in r))

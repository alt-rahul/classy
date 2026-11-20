import sys,json
sys.path.insert(0,'src'); sys.path.insert(0,'research/round4/beyond')
from common import *
fam=json.load(open('research/round4/beyond/fam.json'))
st=load_store(); P=load_puzzles('scored')
from collections import defaultdict
agg=defaultdict(lambda:[0,0,0])
rows=[]
for p in P:
    f=fam[p.id]; b=max(p.targets).bit_length(); n=len(p.targets); L=st[p.id]['len']
    k=(f,(b+5)//10*10 if n>=20 else 0)
    a=agg[k];a[0]+=1;a[1]+=n;a[2]+=L
    rows.append((f,b,n,L,L/n,p.id,len(p.sv),st[p.id]['tag']))
for k in sorted(agg):
    a=agg[k];print(k,a,round(a[2]/a[1],2))
print(sum(st[p.id]['len'] for p in P))
if len(sys.argv)>1:
    for r in sorted(rows):
        if r[0]==sys.argv[1]: print(r)

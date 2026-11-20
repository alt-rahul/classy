import sys,json
sys.path.insert(0,'src'); sys.path.insert(0,'research/round4/beyond')
from common import *
from dagutil import values
st=load_store(); P={p.id:p for p in load_puzzles('scored')}
p=P[sys.argv[1]]; prog=st[p.id]['prog']
print(p.sv,p.limit,len(p.targets),len(prog))
T=set(p.targets)
name={}
for v in p.sv: name[v]=str(v)
for k,(op,a,b,v) in enumerate(values(p,prog)):
    def nm(x):
        return str(x) if x<10**7 else name.get(x,str(x))
    name.setdefault(v,'v%d'%k)
    print(k,'%s = %s %s %s'%(name[v],nm(a),op,nm(b)),v if v>=10**7 else '', 'T' if v in T else '')

import sys
sys.path.insert(0,'/home/user/classy/elf/research/round4/beyond')
from dagutil import *
st=load_store(); P={p.id:p for p in load_puzzles('scored')}
pid=sys.argv[1]; p=P[pid]; prog=st[pid]['prog']
print(p.sv, len(p.targets), p.limit, len(prog), st[pid]['tag'])
T=set(p.targets); name={v:'s%d'%i for i,v in enumerate(p.sv)}
regs=list(p.sv)+[None]*(p.nreg-len(p.sv))
for k,(op,i,j,d) in enumerate(prog):
    a,b=regs[i],regs[j]; v=OPF[op](a,b); regs[d]=v
    def nm(x):
        return name.get(x,str(x))
    if v not in name: name[v]=('T%d'%p.targets.index(v)) if v in T else 'v%d'%k
    live=sum(r is not None for r in regs)
    print(k,name[v],'=',nm(a),op,nm(b),'  ',v, 'bits',v.bit_length(),'live',live)

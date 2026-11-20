import sys,json
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from solver import solve,Cfg
ids=sys.argv[1].split(','); kw=json.loads(sys.argv[2]) if len(sys.argv)>2 else {"pool_cap":46,"regret":0.1,"evict_size":1}
for p in load_puzzles('practice'):
    if p.id[-4:] in ids:
        prog,st=solve(p,Cfg(**kw)); assert check(p,prog) is None
        print(p.id,p.family,len(p.targets),len(prog))
        reg={i:v for i,v in enumerate(p.sv)}; T=set(p.targets)
        for op,i,j,d in prog:
            a,b=reg[i],reg[j]; v={'+':a+b,'-':a-b,'*':a*b,'/':a//b}[op]; reg[d]=v
            print('   %d %s %d = %d %s'%(a,op,b,v,'  <== T' if v in T else ''))

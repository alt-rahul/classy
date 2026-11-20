from lib import *
import sys,json,copy,time
from solver import solve,Cfg
from common import check
ps={p.id:p for p in P()}
p=ps[sys.argv[1]];extra=eval(sys.argv[2]);xt=eval(sys.argv[3]) if len(sys.argv)>3 else []
kw={"pool_cap":46,"evict_size":1,"regret":0.1}
p.sv=p.sv+extra
p.targets=sorted(set(p.targets+xt))
t=time.time()
prog,st=solve(p,Cfg(**kw))
print(sys.argv[1],extra,xt,len(prog),check(p,prog),'%.0fs'%(time.time()-t))
if len(sys.argv)>4:
    from decomp import decomp
    decomp(p,prog)

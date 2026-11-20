import sys, json, time
from crun import *
from polish import elf
from prof import P, st
pid=sys.argv[1]; args=sys.argv[2].split(','); secs=float(sys.argv[3]); n=int(sys.argv[4])
p=P[pid]; prog,log=cons(p,[],args)
print(pid,'store',st[pid]['len'],'cons',len(prog),flush=True)
json.dump(prog,open('/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad/cons_%s.json'%pid,'w'))
for it in range(n):
    prog=elf(p,prog,secs,seed=it+1); print(' after',(it+1)*secs,'s:',len(prog),flush=True)
json.dump(prog,open('/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad/consp_%s.json'%pid,'w'))

"""Positive control for SX: a deliberately weak program (public constructor + 5 s LNS) must be improvable."""
os_env=__import__('os').environ
os_env.setdefault('SX_TH','100000'); os_env.setdefault('SX_K2','0')
from lib import *
import subprocess, restart, sx
P={p.id:p for p in load_puzzles('scored')}; st=load_store()
for pid in sys.argv[1:]:
    p=P[pid]; prog=st[pid]['prog']; pad=prog+[prog[-1]]*80
    r=subprocess.run(['nice','-n','10',restart.EXE,'--construct','1','--time','5','--seed','1'],input=restart.enc(p,pad),capture_output=True,text=True)
    lines=r.stdout.strip().split('\n')[1:]; weak=[[l.split()[0]]+list(map(int,l.split()[1:])) for l in lines]
    assert check(p,weak) is None
    cur=weak; hist=[len(weak)]
    for it in range(6):
        pr,status=sx.solve(p,cur,30,verbose=0)
        if pr is None or check(p,pr) is not None or len(pr)>=len(cur): hist.append(status); break
        cur=pr; hist.append(len(pr))
    print(pid,'stored',len(prog),'weak program -> SX rounds:',hist,flush=True)

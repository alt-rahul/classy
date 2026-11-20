"""Seeded restart: public constructor + LNS with the true mids/hubs (taken from the stored program) as extra targets."""
from lib import *
import subprocess, restart, copy, wlib
P={p.id:p for p in load_puzzles('scored')}; st=load_store()
TH=100000
tl=sys.argv[1]
for pid in sys.argv[2:]:
    p=P[pid]; prog=st[pid]['prog']
    nodes,order=wlib.build_dag(p,prog)
    hubs=[v for v in order if v>=TH and nodes[v][1]<TH and nodes[v][2]<TH]
    mids=sorted({x for h in hubs for x in nodes[h][1:] if x not in p.sv})
    for name,extra in (('hubs+mids',hubs+mids),):
        q=copy.copy(p); q.targets=sorted(set(p.targets)|set(extra))
        pad=prog+[prog[-1]]*80
        r=subprocess.run(['nice','-n','10',restart.EXE,'--construct','3','--time',tl,'--seed','1','--kicks','1000','--kslack','2'],input=restart.enc(q,pad),capture_output=True,text=True)
        cons=[l.split('cost')[1].split()[0] for l in r.stderr.split('\n') if 'construct rep' in l]
        lines=r.stdout.strip().split('\n')[1:]; pr=[[l.split()[0]]+list(map(int,l.split()[1:])) for l in lines]
        ok=check(p,pr) is None if pr else False
        # DCE to the real targets
        n=None
        if ok:
            nn,od=wlib.build_dag(p,pr); n=len(od)
            if n<len(prog):
                pr2=wlib.try_emit5(p,nn,[od])
                if pr2 and check(p,pr2) is None:
                    s2=load_store()
                    if offer(s2,p,pr2,'hubseed'): save_store(s2); print('GAIN',pid,len(prog),len(pr2))
        print(pid,'stored',len(prog),name,len(extra),'construct',cons,'final(with seeds)',len(pr) if pr else None,'after DCE to real targets',n,flush=True)

"""Plateau jump: SX finds an equal-length program that uses as many NEW one-op small values as possible (<= SX_DIV),
then the public LNS polishes it. usage: divx.py sx_tl lns_tl id..."""
from lib import *
import subprocess, sx, restart
if __name__=='__main__':
    stl=float(sys.argv[1]); ltl=sys.argv[2]; ids=sys.argv[3:]
    P={p.id:p for p in load_puzzles('scored')}
    for pid in ids:
        st=load_store(); p=P[pid]; prog=st[pid]['prog']
        for sd in (1,2):
            pr,status=sx.solve(p,prog,stl,verbose=0,seed=sd)
            if pr is None or check(p,pr) is not None: print(pid,len(prog),'sx',status,'no program',flush=True); continue
            d0=set(defs(p,prog)); d1=set(defs(p,pr))
            r=subprocess.run(['nice','-n','10',restart.EXE,'--time',ltl,'--seed',str(sd),'--kicks','1000','--kslack','2'],input=restart.enc(p,pr),capture_output=True,text=True)
            new=len(pr)
            if r.stdout.startswith('R'):
                lines=r.stdout.strip().split('\n')[1:]; pr2=[[l.split()[0]]+list(map(int,l.split()[1:])) for l in lines]
                if check(p,pr2) is None: new=len(pr2); pr=pr2
            print(pid,'stored',len(prog),'jump',len(d1),'newvals',len(d1-d0),'after LNS',new,flush=True)
            if new<len(prog):
                s2=load_store()
                if offer(s2,p,pr,'sx-jump'): save_store(s2); print('GAIN',pid,len(prog),new,flush=True)
                break

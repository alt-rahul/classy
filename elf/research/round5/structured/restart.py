"""Restart test: public constructor (elfopt --construct) + public LNS from scratch vs the stored program."""
from lib import *
import subprocess
EXE=os.path.join(R,'../../../kaggle/build/elfopt')
def enc(p,prog):
    return 'P %s %d %s %d %s %d %d %d\n'%(p.id,len(p.sv),' '.join(map(str,p.sv)),len(p.targets),' '.join(map(str,p.targets)),p.limit,p.nreg,len(prog))+''.join('%s %d %d %d\n'%tuple(x) for x in prog)
if __name__=='__main__':
    tl=sys.argv[1]; seeds=int(sys.argv[2]); ids=sys.argv[3:]
    P={p.id:p for p in load_puzzles('scored')}; st=load_store()
    for pid in ids:
        p=P[pid]; prog=st[pid]['prog']
        for sd in range(1,seeds+1):
            r=subprocess.run(['nice','-n','10',EXE,'--construct','3','--time',tl,'--seed',str(sd),'--kicks','1000','--kslack','2'],input=enc(p,prog),capture_output=True,text=True)
            last=[l for l in r.stderr.split('\n') if '->' in l]
            cons=[l.split('cost')[1].split()[0] for l in r.stderr.split('\n') if 'construct rep' in l]
            print(pid,'stored',len(prog),'seed',sd,'construct',cons,'final',last[-1].split('evals')[0] if last else r.stderr[-200:],flush=True)
            if r.stdout.startswith('R'):
                lines=r.stdout.strip().split('\n')[1:]; pr=[[l.split()[0]]+list(map(int,l.split()[1:])) for l in lines]
                if check(p,pr) is None:
                    s2=load_store()
                    if offer(s2,p,pr,'restart-pub'): save_store(s2); print('GAIN',pid,len(prog),len(pr),flush=True)

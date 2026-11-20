import sys, json, subprocess, time
from rlib import *
st=load_store(); PZ=load_puzzles('scored'); P={p.id:p for p in PZ}
def cls(p):
    b=max(p.targets).bit_length(); c=pclass(p)
    return c, b
def benchset():
    out=[]
    def pick(cond,k):
        s=[p.id for p in PZ if cond(p)][:k]; out.extend(s)
    pick(lambda p: pclass(p)=='indep>=20' and 46<=max(p.targets).bit_length()<=52, 8)
    pick(lambda p: pclass(p)=='indep>=20' and max(p.targets).bit_length()>=56, 8)
    pick(lambda p: pclass(p)=='other>=20' and max(p.targets).bit_length()>=56 and st[p.id]['len']/len(p.targets)>6.3, 6)
    pick(lambda p: pclass(p)=='indep>=20' and 36<=max(p.targets).bit_length()<=42, 6)
    return out
def run_beam(binary, ids, args, tl=600):
    inp=''.join('%s %d %d %d\n%s\n%s\n'%(i,P[i].limit,len(P[i].sv),len(P[i].targets),' '.join(map(str,P[i].sv)),' '.join(map(str,P[i].targets))) for i in ids)
    t0=time.time(); r=subprocess.run(['nice',binary]+[str(a) for a in args],input=inp,capture_output=True,text=True,timeout=tl)
    res={}
    for ln in r.stdout.split('\n'):
        if not ln.strip(): continue
        i,js=ln.split(' ',1); prog=json.loads(js); err=check(P[i],prog)
        res[i]=(len(prog) if not err else None, prog, err)
    return res, time.time()-t0, r.stderr
if __name__=='__main__':
    ids=benchset(); binary=sys.argv[1]; args=sys.argv[2:]
    res,dt,err=run_beam(binary,ids,args)
    tot=0; ts=0
    for i in ids:
        L=res.get(i,(None,None,'missing'))
        print(i,cls(P[i]),len(P[i].targets),'store',st[i]['len'],'raw',L[0],L[2] or '')
        if L[0]: tot+=L[0]; ts+=st[i]['len']
    print('TOTAL raw',tot,'store',ts,'ratio %.3f'%(tot/ts),'time %.0fs'%dt, err[:200])

# benchmark the constructor on the fixed set; offers any strictly shorter valid program to the store (tag r5-cons)
import sys, time, json
from crun import *
import bench
from multiprocessing import Pool
def one(a):
    pid,args=a; p=bench.P[pid]; t0=time.time()
    best=None; log=''
    for regcap in (47,46,45,44):
        ar=list(args); ar[9]=regcap
        prog,log=cons(p,[],ar)
        if prog: best=prog; break
    return pid,best,time.time()-t0,log[-200:] if best is None else ''
if __name__=='__main__':
    tag=sys.argv[1]; args=sys.argv[2].split(','); nw=int(sys.argv[3]); ids=bench.benchset()
    if len(sys.argv)>4: ids=sys.argv[4].split(',')
    st=load_store(); tot=0; ts=0; out={}
    with Pool(nw) as pool:
        for pid,prog,dt,log in pool.imap(one,[(i,args) for i in ids]):
            L=len(prog) if prog else None; p=bench.P[pid]
            print(pid,bench.cls(p),len(p.targets),'store',st[pid]['len'],'cons',L,'%.0fs'%dt,log.replace('\n',' | '),flush=True)
            if prog:
                tot+=L; ts+=st[pid]['len']; out[pid]=prog
                if L<st[pid]['len']:
                    s2=load_store()
                    if offer(s2,p,prog,'r5-cons'): save_store(s2); print('  STORED',pid,L,flush=True)
    json.dump(out,open('bench_%s.json'%tag,'w'))
    print('TOTAL cons',tot,'store',ts,'ratio %.4f'%(tot/ts if ts else 0),'n',len(out))

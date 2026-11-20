"""SX over hub families: new smalls (< TH) plus new big values f op s / f1 op f2 inside one hub family at a time.
usage: sxfam.py tl id...   (env SX_TH default 100000, SX_FMAX family cap 14)"""
os_env=__import__('os').environ
os_env.setdefault('SX_TH','100000'); os_env.setdefault('SX_K2','0')
from lib import *
import sx, wlib, collections, time
TAG='sx-fam'; FMAX=int(os_env.get('SX_FMAX','14')); TH=int(os_env['SX_TH'])
def families(p,prog):
    nodes,order=wlib.build_dag(p,prog)
    users=collections.defaultdict(list)
    for v,(op,a,b) in nodes.items(): users[a].append(v); users[b].append(v)
    hubs=[v for v in order if v>=TH and nodes[v][1]<TH and nodes[v][2]<TH]
    fams=[]
    for h in hubs:
        F=[h]; q=[h]; seen={h}
        while q and len(F)<FMAX:
            x=q.pop(0)
            for u in users[x]:
                if u not in seen and len(F)<FMAX: seen.add(u); F.append(u); q.append(u)
        if len(F)>=3: fams.append(set(F))
    return fams
if __name__=='__main__':
    tl=float(sys.argv[1]); ids=sys.argv[2:]
    P={p.id:p for p in load_puzzles('scored')}
    for pid in ids:
        p=P[pid]; t0=time.time(); res=collections.Counter(); gain=0
        prog=load_store()[pid]['prog']; n00=len(prog)
        fams=families(p,prog); k=0
        while k<len(fams):
            sx.FAMSET=fams[k]; k+=1
            pr,status=sx.solve(p,prog,tl,verbose=0)
            res[status]+=1
            if pr is not None and check(p,pr) is None and len(pr)<len(prog):
                st=load_store()
                if offer(st,p,pr,TAG): save_store(st)
                gain+=len(prog)-len(pr); prog=pr; fams=families(p,prog); k=0
                print('GAIN',pid,n00,'->',len(prog),flush=True)
        print('done',pid,n00,'->',len(prog),'families',len(fams),dict(res),'t %.0f'%(time.time()-t0),flush=True)
        with open(R+'/sxfam_runs.jsonl','a') as f: f.write(json.dumps({'id':pid,'old':n00,'new':len(prog),'fams':len(fams),'res':dict(res),'t':time.time()-t0,'tl':tl})+'\n')

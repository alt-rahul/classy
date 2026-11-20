# single-cone rebuild (previous agent's wide2 + best_splice) applied to an arbitrary program
import sys, time, json, collections
sys.path.insert(0,'/home/user/classy/elf/research/round4/beyond')
from wlib import *
def cone_hist(p, prog):
    nodes, order = build_dag(p, prog); cones = all_cones(p, nodes)
    return sorted(collections.Counter(len(c[0]) for c in cones.values()).items())
def lns(p, prog, kmin, tl, maxt=1e9, verbose=True):
    nodes, order = build_dag(p, prog); tried=set(); t00=time.time()
    while time.time()-t00<maxt:
        cones = all_cones(p, nodes)
        cand = [(len(c[0]), t) for t, c in cones.items() if len(c[0]) >= kmin and (t, len(c[0])) not in tried]
        if not cand: break
        k, t = max(cand); tried.add((t, k)); cone, desc, users = cones[t]
        pool = [v for v in list(dict.fromkeys(p.sv)) + order if v not in cone and v not in desc]
        t0=time.time(); sols, err = run_wide_enum(p.limit, t, pool, max(4, k - 3), k - 1, 1, tl, 300)
        res='none'
        if sols:
            pr, bp = best_splice(p, nodes, order, cone, sols, t, len(prog)); res='noemit%d'%bp
            if pr is not None and check(p,pr) is None and len(pr)<len(prog):
                prog = pr; nodes, order = build_dag(p, prog); res='ok'
        if verbose: print('  cone',k,'->',min(map(len,sols)) if sols else None,res,'len',len(prog),'%.0fs'%(time.time()-t0),flush=True)
    return prog
if __name__=='__main__':
    P={p.id:p for p in load_puzzles('scored')}; st=load_store()
    pid=sys.argv[1]; prog=json.load(open(sys.argv[2])); 
    if isinstance(prog,dict): prog=prog[pid]
    p=P[pid]; print(pid,'len',len(prog),'store',st[pid]['len'],'cones',cone_hist(p,prog),' store cones',cone_hist(p,st[pid]['prog']))
    if len(sys.argv)>3:
        prog=lns(p,prog,int(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]))
        print('final',len(prog),cone_hist(p,prog))
        json.dump(prog,open(sys.argv[2]+'.lns','w'))
        if len(prog)<st[pid]['len']:
            s2=load_store()
            if offer(s2,p,prog,'r5-cons-lns'): save_store(s2); print('STORED',pid,len(prog))

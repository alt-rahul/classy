"""Prototype: mine hidden hubs (+ shared gcd, + constants) and hand them to the existing solver as extra targets.
usage: seed_proto.py practice|scored id1,id2,... [variants]   variants: plain,hubs,hubsc"""
import sys,math,time,json
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from solver import solve,Cfg
from mine2 import mine
KW={"pool_cap":46,"regret":0.1,"evict_size":1}
def gain(B,d,T):
    g=-B.bit_length()
    for ti,(m,c) in d.items():
        g+=max(0,T[ti].bit_length()-m.bit_length()-abs(c).bit_length()-2)
    return g
import os
STRICT=int(os.environ.get('STRICT','0'))
def pick(T,hubs,maxhubs=10,ming=12):
    if STRICT: ming=20
    sc={B:gain(B,d,T) for B,d in hubs.items()}
    rank=sorted((B for B in hubs if sc[B]>=ming),key=lambda B:-sc[B])
    cov=set(); out=[]
    for B in rank:
        d=hubs[B]
        new=[ti for ti in d if ti not in cov]
        if any(abs(B-b2)<=6000 for b2,_ in out): continue
        if len(new)>=2 or (not STRICT and len(new)>=1 and len(d)>=3):
            out.append((B,d)); cov|=set(d)
        if len(out)>=maxhubs: break
    return out
def seeds(p):
    hubs=mine(p.targets); sel=pick(p.targets,hubs)
    Tset=set(p.targets)|set(p.sv)
    H=[B for B,d in sel if B not in Tset]
    G=set()
    allh=[B for B,d in sel]
    for i in range(len(allh)):
        for j in range(i):
            g=math.gcd(allh[i],allh[j])
            if g>=10**4 and g not in Tset and g not in allh: G.add(g)
    consts=set()
    for B,d in sel:
        for ti,(m,c) in d.items():
            for x in (m,abs(c)):
                if 2<=x<=10**5 and x not in Tset: consts.add(x)
    return H,sorted(G),sorted(consts)
def run(p,extra):
    q=Puzzle.__new__(Puzzle); q.id=p.id; q.sv=p.sv; q.limit=p.limit; q.nreg=p.nreg; q.family=p.family
    q.targets=sorted(set(p.targets)|set(x for x in extra if x<=p.limit))
    prog,st=solve(q,Cfg(**KW))
    assert check(p,prog) is None
    return prog
if __name__=='__main__':
    which=sys.argv[1]; ids=sys.argv[2].split(','); variants=(sys.argv[3] if len(sys.argv)>3 else 'plain,hubs,hubsc').split(',')
    for p in load_puzzles(which):
        if p.id[-4:] not in ids: continue
        H,G,Cs=seeds(p)
        res={}
        for v in variants:
            t0=time.time()
            extra={'plain':[],'hubs':H+G,'hubsc':H+G+Cs}[v]
            prog=run(p,extra); res[v]=len(prog)
            json.dump(prog,open('progs/%s_%s.json'%(p.id,v),'w'))
        print(p.id,p.family,len(p.targets),'H',H,'hubs',len(H),'G',len(G),'consts',len(Cs),res,flush=True)

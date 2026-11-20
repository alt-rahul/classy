import sys, pickle
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
def ann(pid, steps, show=True):
    P={p.id:p for p in load_puzzles('practice')}; p=P[pid]; T={t:i for i,t in enumerate(p.targets)}
    # keep only needed steps
    prod={s[3]:k for k,s in enumerate(steps)}
    need=set(); stack=[t for t in p.targets if t in prod]
    while stack:
        v=stack.pop()
        if v in need or v not in prod: continue
        need.add(v); s=steps[prod[v]]; stack+= [s[1],s[2]]
    uses={}
    for s in steps:
        if s[3] in need:
            uses[s[1]]=uses.get(s[1],0)+1; uses[s[2]]=uses.get(s[2],0)+1
    n=0; last=0
    for s in steps:
        if s[3] not in need: continue
        n+=1
        tag=''
        if s[3] in T: tag='  <== T%d (+%d)'%(T[s[3]],n-last); last=n
        if show: print('%3d  %d = %d %s %d  [uses %d]%s'%(n,s[3],s[1],s[0],s[2],uses.get(s[3],0),tag))
    return n
if __name__=='__main__':
    d=pickle.load(open(sys.argv[2],'rb')); ann(sys.argv[1], d[1])

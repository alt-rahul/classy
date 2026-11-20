import sys,json,os
R=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(R,'../../../src')); sys.path.insert(0,os.path.join(R,'../../round4/beyond'))
from common import *
OPF = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b, '/': lambda a, b: a // b}
FAM=json.load(open(os.path.join(R,'../../round4/beyond/fam.json')))
def vprog(p,prog):
    """value-level SSA: list of (op,a,b,v), dedupe not applied"""
    regs=list(p.sv)+[None]*(p.nreg-len(p.sv)); out=[]
    for op,i,j,d in prog:
        a,b=regs[i],regs[j]; v=OPF[op](a,b); regs[d]=v; out.append((op,a,b,v))
    return out
def defs(p,prog):
    d={}
    for k,(op,a,b,v) in enumerate(vprog(p,prog)):
        if v not in d and v not in p.sv: d[v]=(op,a,b,k)
    return d
def cones(p,prog):
    """exclusive cone size per target: ops that die if target t is no longer required"""
    d=defs(p,prog); T=[t for t in p.targets if t in d]
    def closure(ts):
        seen=set(); stack=list(ts)
        while stack:
            v=stack.pop()
            if v in seen or v not in d: continue
            seen.add(v); stack+= [d[v][1],d[v][2]]
        return seen
    full=closure(T); res={}
    for t in T:
        res[t]=len(full)-len(closure([x for x in T if x!=t]))
    return res,len(full)
def mixed_like(p):
    """fraction of targets in the top decade"""
    from collections import Counter
    c=Counter(len(str(t)) for t in p.targets); k,n=c.most_common(1)[0]
    return k,n

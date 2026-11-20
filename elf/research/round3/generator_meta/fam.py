import sys,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
def decade(p):
    """returns (k, list of targets in the modal decade among 6,9,12,15,18 digits)"""
    dg=collections.Counter(len(str(t)) for t in p.targets)
    k=max((6,9,12,15,18),key=lambda k:dg[k])
    return k,[t for t in p.targets if len(str(t))==k]
def fam(p):
    if p.limit<=16: return 'tiny'
    if max(p.sv)>40: return 'indep'
    k,r=decade(p)
    if len(r)>=max(4,len(p.targets)//2-1) and (len(p.targets)>=20 or True): return 'mixed'
    return 'sm'
if __name__=='__main__':
    for w in ('practice',):
        c=collections.Counter()
        for p in load_puzzles(w):
            f=fam(p); c[(p.family,f)]+=1
            if (p.family=='mixed')!=(f=='mixed'): print(p.id,p.family,f,decade(p)[0],len(decade(p)[1]),len(p.targets))
        print(c)
    st=load_store(); c=collections.Counter(); n=collections.Counter(); nt=collections.Counter(); nr=collections.Counter()
    for p in load_puzzles('scored'):
        f=fam(p)+('<20' if len(p.targets)<20 else ''); c[f]+=st[p.id]['len']; n[f]+=1; nt[f]+=len(p.targets)
        if f.startswith('mixed'): nr[f+str(decade(p)[0])]+=len(decade(p)[1]); n[f+str(decade(p)[0])]+=1; c[f+str(decade(p)[0])]+=st[p.id]['len']; nt[f+str(decade(p)[0])]+=len(p.targets)
    for f in sorted(n): print(f,n[f],nt[f],c[f],'%.2f'%(c[f]/nt[f]),nr.get(f))

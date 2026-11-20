import sys,math,collections,random
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from fam import fam,decade
from nt import factor
random.seed(2)
def stat(X,name):
    r=[];om=[]
    for x in X:
        f=factor(x); r.append(math.log(max(f))/math.log(x)); om.append(sum(f.values()))
    r.sort(); n=len(r)
    print(name,n,'P(maxp<n^0.3)=%.3f P(<0.4)=%.3f P(<0.5)=%.3f median=%.3f omega=%.2f'%(sum(x<.3 for x in r)/n,sum(x<.4 for x in r)/n,sum(x<.5 for x in r)/n,r[n//2],sum(om)/n))
P=load_puzzles('scored')+load_puzzles('practice')
for k in (12,15,18):
    M=[t for p in P if fam(p)=='mixed' and len(p.targets)>=20 and decade(p)[0]==k for t in decade(p)[1]]
    I=[t for p in P if fam(p)=='indep' and len(p.targets)>=20 and decade(p)[0]==k for t in decade(p)[1]]
    random.shuffle(M); random.shuffle(I)
    stat(M[:600],'mixed%d'%k); stat(I[:600],'indep%d'%k)

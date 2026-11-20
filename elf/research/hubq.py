import sys,math
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from collections import defaultdict
def hubs(p,Mmax=4096,S=64,minhub=3000,exact=False):
    T=sorted(set(p.targets)-set(p.sv))
    Q=[]
    for i,t in enumerate(T):
        for M in range(1,Mmax+1):
            q,r=divmod(t,M)
            if q<minhub: break
            if exact:
                if r==0: Q.append((q,i,M,0))
            else:
                if r<=S: Q.append((q,i,M,r))
                if M-r<=S and r: Q.append((q+1,i,M,r-M))
    Q.sort()
    linked=defaultdict(list)
    j=0
    for k in range(len(Q)):
        while Q[k][0]-Q[j][0]>S: j+=1
        for m in range(j,k):
            if Q[m][1]!=Q[k][1]:
                # significance: need hub big vs window
                if Q[k][0]> 200*S*Q[k][2]*Q[m][2]:
                    linked[Q[k][1]].append((Q[m][1],Q[k][0],Q[k][2],Q[m][2])); linked[Q[m][1]].append((Q[k][1],Q[m][0],Q[m][2],Q[k][2]))
    return T,linked
P=load_puzzles('practice')
import collections
agg=collections.defaultdict(lambda:[0,0])
for p in P:
    if p.family=='tiny' or len(p.targets)<20: continue
    T,l=hubs(p)
    agg[p.family][0]+=len(l); agg[p.family][1]+=len(T)
    if len(sys.argv)>1 and p.id==sys.argv[1]:
        for i,t in enumerate(T): print(i,t,sorted(set(l.get(i,[])))[:4])
print({k:(v,round(v[0]/v[1],2)) for k,v in agg.items()})

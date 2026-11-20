"""Exact tree (formula) cost from the starts for ALL values <= limit on small-limit independent puzzles.
Compare target costs with the cost distribution of all values in the target decade."""
import sys, numpy as np, collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
def costs(sv, L, maxk=7, budget=3e9):
    cost=np.full(L+1,99,np.int8)
    S=[np.unique(np.array([v for v in sv if v<=L],np.int64))]
    cost[S[0]]=0
    for k in range(1,maxk+1):
        new=np.zeros(L+1,bool)
        for i in range(0,(k-1)//2+1):
            j=k-1-i
            A,B=S[i],S[j]
            if len(A)>len(B): A,B=B,A
            if len(A)*len(B)>budget: print('skip',k,i,j,len(A),len(B)); continue
            for a in A.tolist():
                s=B+a; new[s[s<=L]]=True
                d=B-a; new[d[d>0]]=True
                d=a-B; new[d[d>0]]=True
                m=B[B<=L//a]*a; new[m]=True
                q=B[B%a==0]//a; new[q]=True
                q=B[(B<=a)]; q=q[a%q==0]; new[a//q]=True
        new&=(cost==99)
        idx=np.nonzero(new)[0]; idx=idx[idx>0]
        cost[idx]=k; S.append(idx.astype(np.int64))
        print('  level',k,len(idx),flush=True)
        if (cost[1:]==99).sum()==0: break
    return cost
if __name__=='__main__':
    which=sys.argv[1]; ids=sys.argv[2:]
    for p in load_puzzles(which):
        if p.id[-4:] in ids:
            print(p.id,p.sv,p.limit,p.targets)
            c=costs(p.sv,p.limit)
            lo=10**(len(str(min(p.targets)))-1)
            allc=collections.Counter(c[lo:10*lo].tolist()); n=sum(allc.values())
            tc=collections.Counter(c[p.targets].tolist())
            print(' decade dist',{k:round(v/n,4) for k,v in sorted(allc.items())})
            print(' targets   ',sorted(tc.items()), 'mean dec %.3f tgt %.3f'%(sum(k*v for k,v in allc.items())/n, np.mean(c[p.targets])))

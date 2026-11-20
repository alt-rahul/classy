import sys,json,math,random,time
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
A=4096
def mine(p):
    T=sorted(set(p.targets)-set(p.sv)); n=len(T); par={}
    for j in range(n):
        t2=T[j]
        for i in range(n):
            t1=T[i]
            if i==j or t1<1e6: continue
            if t1<t2:
                b=(t2+t1//2)//t1
                r=t2-b*t1
                fp=2*(b*A)/t1*max(1,b)**0  # false-positive prob
                if abs(r)<=b*A and fp<1e-3:
                    par.setdefault(t2,[]).append((fp,'%d*T%d%+d'%(b,i,r)))
            else:
                # t2 = (t1 + a)/b  -> t1 = b*t2 - a
                b=(t1+t2//2)//t2
                r=t1-b*t2
                if b>1 and abs(r)<=A and 2*A/t2<1e-3 and t2>1e6:
                    par.setdefault(t2,[]).append((2*A/t2,'(T%d%+d)/%d'%(i,-r,b)))
        for i in range(n):
            for k in range(i,n):
                for nm,v in (('+',T[i]+T[k]),('*',T[i]*T[k]),('-',T[k]-T[i])):
                    if j in (i,k) or v<1e6: continue
                    r=t2-v
                    if abs(r)<=A and min(T[i],T[k])>1e6*(nm!='*'): par.setdefault(t2,[]).append((1e-9,'T%d%sT%d%+d'%(k,nm,i,r)))
    return T,par
if __name__=='__main__':
    P=load_puzzles('practice')
    fam=sys.argv[1]; k=int(sys.argv[2]); verbose=len(sys.argv)>3
    agg=[0,0]
    for p in P:
        if p.family!=fam or len(p.targets)<20: continue
        if k==0: break
        k-=1
        T,par=mine(p)
        big=[t for t in T if t>=1e6]
        agg[0]+=sum(1 for t in big if t in par); agg[1]+=len(big)
        print(p.id,p.sv,'targets',len(T),'big',len(big),'with parent',sum(1 for t in big if t in par))
        if verbose:
            for i,t in enumerate(T): print('   T%d'%i,t,[x[1] for x in sorted(par.get(t,[]))[:4]])
    print(agg)

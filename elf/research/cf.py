import sys,math
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
def cf_rel(t1,t2,A=4096,qmax=10**7):
    # find p/q ~ t2/t1 : |t2*q - t1*p| <= A*(p+q)   (t2 = p*(B+a2)+c2, t1 = q*(B+a1)+c1)
    out=[]
    h0,h1,k0,k1=0,1,1,0
    x,y=t2,t1
    while y and k1<=qmax:
        a=x//y; x,y=y,x-a*y
        h0,h1=h1,a*h1+h0; k0,k1=k1,a*k1+k0
        if k1>qmax or h1>qmax: break
        r=t2*k1-t1*h1
        B=t1//k1
        if B>1e6 and abs(r)<=A*(h1+k1) and 4*A*(h1+k1)*max(h1,k1)<B*1e-2:   # significance
            out.append((h1,k1,r))
    return out
if __name__=='__main__':
    P=load_puzzles('practice')
    fam=sys.argv[1]; k=int(sys.argv[2]); verbose=len(sys.argv)>3
    agg=[0,0]
    for p in P:
        if p.family!=fam or len(p.targets)<20: continue
        if k==0: break
        k-=1
        T=sorted(set(p.targets)); n=len(T); link={}
        for i in range(n):
            for j in range(i+1,n):
                rr=cf_rel(T[i],T[j])
                if rr:
                    link.setdefault(j,[]).append((i,rr[0])); link.setdefault(i,[]).append((j,rr[0]))
        big=[i for i in range(n) if T[i]>1e7]
        agg[0]+=sum(1 for i in big if i in link); agg[1]+=len(big)
        print(p.id,p.sv,'big',len(big),'linked',sum(1 for i in big if i in link))
        if verbose:
            for i in range(n): print('   T%d'%i,T[i],[(j,)+r for j,r in link.get(i,[]) if j<i][:5])
    print(agg)

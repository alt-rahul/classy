"""Hidden-base mining: pairs (Ti,Tj) with mj*Ti - mi*Tj small -> common base H ~ Ti/mi. Cluster and print M, c."""
import sys,math,collections
from fractions import Fraction
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from nt import factor
def fs(n): return '*'.join('%d^%d'%(a,b) if b>1 else str(a) for a,b in sorted(factor(n).items())) if n>1 else str(n)
def convergents(a,b,maxden):
    h0,h1,k0,k1=0,1,1,0
    out=[]
    while b:
        q=a//b; a,b=b,a-q*b
        h0,h1=h1,q*h1+h0; k0,k1=k1,q*k1+k0
        if k1>maxden: break
        out.append((h1,k1))
    return out
TOL=1000
def links(T,maxm=5000,tol=None):
    L=[]
    for i in range(len(T)):
        for j in range(i):
            a,b=T[i],T[j]   # a>=b ; a/b ~ mi/mj
            for mi,mj in convergents(a,b,maxm):
                if mi>maxm*50: break
                r=mj*a-mi*b
                H=a//mi if mi else 0
                if H<10**6: continue
                # significance: |r| << b (random expectation for |r| at this denominator ~ b/mj.. ) require |r|*mi*mj < b * 1e-3
                if abs(r)*mi*mj*TOL < b* max(mi,mj): L.append((j,i,mj,mi,r,H))
    return L
if __name__=='__main__':
    which,pid=sys.argv[1],sys.argv[2]
    for p in load_puzzles(which):
        if p.id[-4:]==pid:
            T=p.targets
            for j,i,mj,mi,r,H in links(T):
                print('T%d*%d - T%d*%d = %d   H~%d  (%s)'%(i,mj,j,mi,r,H,fs(abs(r)) if r else 0))

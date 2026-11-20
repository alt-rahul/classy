from lib import *
import sys,math
def cfrels(a,b,pm=10**7):
    """yield (p,q,r): q*a - p*b = r, p/q convergents of a/b"""
    h0,h1,k0,k1=0,1,1,0
    x,y=a,b
    while y:
        c=x//y;x,y=y,x-c*y
        h0,h1=h1,c*h1+h0;k0,k1=k1,c*k1+k0
        if h1>pm or k1>pm: break
        if h1==0: continue
        yield h1,k1,k1*a-h1*b
if __name__=='__main__':
    ps={p.id:p for p in P()}
    p=ps[sys.argv[1]];T=p.targets
    print(p.family,p.sv,p.limit)
    for i in range(len(T)):
        out=[]
        for j in range(len(T)):
            if i==j: continue
            best=None
            for pp,q,r in cfrels(T[i],T[j]):
                s=math.log2(min(T[i],T[j]))-math.log2(pp)-math.log2(q)-math.log2(abs(r)+1)
                if best is None or s>best[0]: best=(s,pp,q,r)
            if best and best[0]>12: out.append((round(best[0],1),j)+best[1:])
        out.sort(reverse=True)
        print(i,T[i],['%d*T%d=%d*T%d%+d (%.0f)'%(q,i,pp,j,r,s) for s,j,pp,q,r in out[:6]])

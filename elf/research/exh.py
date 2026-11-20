import sys,json,math,random,time,pickle
sys.path.insert(0,'/home/user/classy/elf/src')
import engine, numpy as np
from collections import Counter
def levels(pool,limit):
    pool=sorted(set(pool))
    def comb(A,B):
        A=np.array(A,dtype=object); out=set()
        for b in B:
            for a in A.tolist():
                if a+b<=limit: out.add(a+b)
                if a*b<=limit: out.add(a*b)
                if a>b: 
                    out.add(a-b)
                    if a%b==0: out.add(a//b)
                elif b>a:
                    out.add(b-a)
                    if b%a==0: out.add(b//a)
        return out
    L0=set(pool); L1=comb(pool,pool)-L0; L2=comb(sorted(L1),pool)-L1-L0
    return L0,L1,L2
def search(t,pool,limit,L0,L1,L2,maxY=5,tmax=600,verbose=False):
    """t = X*Y +- Z ; X in L0|L1, Z in {0}|L0|L1|L2, Y by find_pair"""
    best=(99,None); t0=time.time()
    Zs=[(0,0)]+[(z,0) for z in L0]+[(z,1) for z in L1]+[(z,2) for z in L2]
    zarr=np.array([z for z,_ in Zs if z< 2**62],dtype=np.int64); zc=np.array([c for z,c in Zs if z<2**62],dtype=np.int64)
    Xs=[(x,0) for x in L0 if x>1]+[(x,1) for x in L1 if x>1]
    cands=[]
    for x,cx in Xs:
        if x>t//2: continue
        for sg in (1,-1):
            # t = x*y + sg*z  -> z ≡ sg*t mod x ... (t - sg*z) % x ==0
            m=((t - sg*zarr) % x)==0
            idx=np.nonzero(m)[0]
            for i in idx:
                z=int(zarr[i]); cz=int(zc[i])
                y=(t-sg*z)//x
                if y<1 or (sg==-1 and x*y>limit): continue
                base=cx+cz+(2 if z else 1)
                cands.append((base,x,z,sg,y))
    cands.sort()
    nfp=0
    for base,x,z,sg,y in cands:
        mc=min(maxY,best[0]-1-base)
        if mc<0: break
        r=engine.find_pair(y,mc); nfp+=1
        if r is not None and base+r[0]<best[0]:
            best=(base+r[0],(x,z,sg,y,r[0]))
        if time.time()-t0>tmax: break
    return best,len(cands),nfp
if __name__=='__main__':
    d=pickle.load(open(sys.argv[1],'rb')); nt=int(sys.argv[2]); maxY=int(sys.argv[3])
    pool,limit=d['pool'],d['limit']
    L0,L1,L2=levels(pool,limit); print('levels',len(L0),len(L1),len(L2))
    sz=engine.build(pool,limit,3,0); print(sz)
    res=[]
    for t in d['ts'][:nt]:
        t0=time.time()
        r=engine.find_pair(t,11); c0=r[0] if r else 99
        w=engine.find_window(t,min(c0,12),1.5e6); c1=w[0] if w else 99
        b,nc,nfp=search(t,pool,limit,L0,L1,L2,maxY)
        res.append((min(c0,c1),b[0]))
        print(t.bit_length(),'engine pair',c0,'win',c1,'| exhaustive',b,'cands',nc,'fp',nfp,'%.0fs'%(time.time()-t0),flush=True)
    print('engine mean %.2f exhaustive mean %.2f best-of-both %.2f'%(np.mean([a for a,b in res]),np.mean([b for a,b in res]),np.mean([min(a,b) for a,b in res])))

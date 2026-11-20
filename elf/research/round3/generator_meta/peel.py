"""G2 test: backward BFS from each target using inverse ops with START values only; does it hit another target/start?"""
import sys,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
def back(t,S,L,depth,K):
    """return (d, path) of the nearest hit in K (excluding t) or None"""
    seen={t:None}; fr=[t]
    for d in range(1,depth+1):
        nf=[]
        for v in fr:
            for s in S:
                cand=[(v+s,'-'),(v*s,'/')]            # v = u - s ; v = u / s
                if v>s: cand.append((v-s,'+'))         # v = u + s
                if v%s==0: cand.append((v//s,'*'))     # v = u * s
                if s>v: cand.append((s-v,'r-'))        # v = s - u
                for u,op in cand:
                    if u<=0 or u>L or u in seen: continue
                    seen[u]=(v,op,s)
                    if u in K and u!=t:
                        path=[];x=u
                        while seen[x] is not None:
                            v2,op2,s2=seen[x]; path.append('%s%d'%(op2,s2)); x=v2
                        return d,u,path
                    nf.append(u)
        fr=nf
    return None
if __name__=='__main__':
    depth=int(sys.argv[1]); fams=sys.argv[2].split(','); verbose=len(sys.argv)>3
    res=collections.defaultdict(lambda:collections.Counter())
    for p in load_puzzles('practice'):
        if p.family not in fams or len(p.targets)<20: continue
        K=set(p.targets)|set(p.sv)
        for t in p.targets:
            if t<10**7: continue
            r=back(t,p.sv,p.limit,depth,K)
            res[p.family][r[0] if r else 0]+=1
            if verbose and r: print(p.id[-4:],t,'<-',r[1],' '.join(r[2]))
    for f in res: print(f,sorted(res[f].items()),'hit frac %.3f'%(1-res[f][0]/sum(res[f].values())))

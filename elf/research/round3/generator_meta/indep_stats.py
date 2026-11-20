import sys,random,math,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from nt import factor
P=[p for w in('scored','practice') for p in load_puzzles(w) if max(p.sv)>40 and p.limit>16]
T=[t for p in P for t in p.targets]
print(len(P),len(T))
random.seed(1)
C=[random.randrange(10**(len(str(t))-1),10**len(str(t))) for t in T]
def stats(X,name):
    ld=collections.Counter(str(x)[0] for x in X)
    print(name,'leading',[ld[str(d)] for d in range(1,10)])
    print(name,'mod2..11 zero frac',[round(sum(x%m==0 for x in X)/len(X),3) for m in (2,3,5,7,11,13)])
    print(name,'last digit',sorted(collections.Counter(x%10 for x in X).items()))
    sub=X[::8]
    fs=[factor(x) for x in sub]
    print(name,'omega mean',sum(sum(f.values()) for f in fs)/len(fs),'log(maxp)/log(n) mean',sum(math.log(max(f))/math.log(x) for f,x in zip(fs,sub))/len(fs),'primes',sum(len(f)==1 and sum(f.values())==1 for f in fs)/len(fs))
stats(T,'T');stats(C,'C')
# within-puzzle: pair gcds, min gap relative
g=[];gc=[]
for p in P:
    t=p.targets
    for i in range(len(t)):
        for j in range(i):
            g.append(math.gcd(t[i],t[j]))
    lo=10**(len(str(t[0]))-1); c=[random.randrange(lo,10*lo) for _ in t]
    for i in range(len(c)):
        for j in range(i):
            gc.append(math.gcd(c[i],c[j]))
for X,n in((g,'T'),(gc,'C')):
    print(n,'gcd>1',sum(x>1 for x in X)/len(X),'gcd>100',sum(x>100 for x in X)/len(X),'max',max(X))
# spacing: normalized gaps (uniform -> exponential)
gaps=[]
for p in P:
    if len(p.targets)<20: continue
    t=p.targets; lo=10**(len(str(t[0]))-1)
    u=[(x-lo)/(9*lo) for x in t]
    gaps+= [ (b-a)*len(t) for a,b in zip(u,u[1:])]
print('gap mean',sum(gaps)/len(gaps),'frac<0.1',sum(x<0.1 for x in gaps)/len(gaps),'(exp: %.3f)'%(1-math.exp(-0.1)),'frac<0.01',sum(x<0.01 for x in gaps)/len(gaps),'min',min(gaps))
# uniform in decade? position histogram
h=collections.Counter(int((t-10**(len(str(t))-1))/(9*10**(len(str(t))-1))*10) for t in T); print('pos hist',sorted(h.items()))
# targets vs limit: any target > limit? count of n targets
print(collections.Counter(len(p.targets) for p in P))
print(collections.Counter(len(p.sv) for p in P if len(p.targets)>=20),collections.Counter(len(p.sv) for p in P if len(p.targets)<20))

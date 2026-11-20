import sys,math
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from nt import factor
from fam import decade
which,pid=sys.argv[1],sys.argv[2]
def fs(n): return ' * '.join('%d^%d'%(a,b) if b>1 else str(a) for a,b in sorted(factor(n).items()))
for p in load_puzzles(which):
    if p.id[-4:]==pid:
        T=p.targets; print(p.sv,p.limit)
        for i,t in enumerate(T):
            print(i,t,t.bit_length(),fs(t))
        print('pair gcd > 1000:')
        for i in range(len(T)):
            for j in range(i):
                g=math.gcd(T[i],T[j])
                if g>1000: print('  ',j,i,'gcd',g,fs(g),' cof',T[j]//g,T[i]//g)
        print('near ratio / small remainders:')
        for i in range(len(T)):
            for j in range(len(T)):
                if i==j or T[j]>T[i]: continue
                q,r=divmod(T[i],T[j])
                if r>T[j]//2: q+=1; r-=T[j]
                if abs(r)<T[j]**0.5 and q<10**6: print('  T%d = %d*T%d %+d'%(i,q,j,r))

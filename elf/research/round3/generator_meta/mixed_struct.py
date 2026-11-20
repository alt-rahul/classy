"""composition of the structured half of mixed puzzles: base-linked (shared-like), gcd-linked (mult-like), unlinked"""
import sys,math,collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from fam import fam,decade
import bases
from bases import links
bases.TOL=100000
def analyse(p,T):
    link=set()
    for j,i,mj,mi,r,H in links(T,maxm=3000):
        link.add(i); link.add(j)
    g=set()
    for i in range(len(T)):
        for j in range(i):
            gg=math.gcd(T[i],T[j])
            if gg>=1000 and gg*gg*gg>min(T[i],T[j]) or gg>10**5: g.add(i); g.add(j)
    return link,g
tot=collections.Counter()
for p in load_puzzles('practice'):
    if len(p.targets)<20 or p.family in ('tiny','independent'): continue
    if p.family=='mixed':
        k,r=decade(p); T=p.targets
    else: T=p.targets; r=[]
    link,g=analyse(p,T)
    for i,t in enumerate(T):
        key=p.family+('-R' if t in r else '')
        tot[(key,'n')]+=1
        tot[(key,'base')]+= i in link
        tot[(key,'gcd')]+= i in g
        tot[(key,'none')]+= (i not in g and i not in link)
for k in sorted(set(a for a,b in tot)):
    n=tot[(k,'n')]; print(k,n,'base-linked %.2f gcd-linked %.2f none %.2f'%(tot[(k,'base')]/n,tot[(k,'gcd')]/n,tot[(k,'none')]/n))

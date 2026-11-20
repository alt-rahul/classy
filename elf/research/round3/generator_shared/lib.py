import sys,csv,json
sys.path.insert(0,'/home/user/classy/elf/src')
from common import load_puzzles
def rows():
    return list(csv.DictReader(open('/home/user/classy/elf/data/practice_puzzles.csv')))
def P(which='practice'):
    ps=load_puzzles(which)
    if which=='practice':
        for p,r in zip(ps,rows()): p_tier=r['design_tier']
    return ps
def tiers():
    return {r['id']:r['design_tier'] for r in rows()}
def ops(a,b,lim):
    r=[]
    if a+b<=lim: r.append(('+',a+b))
    if a*b<=lim: r.append(('*',a*b))
    if a>b: r.append(('-',a-b))
    if b>a: r.append(('-r',b-a))
    if a%b==0: r.append(('/',a//b))
    if b%a==0: r.append(('/r',b//a))
    return r
def formula_levels(base,lim,K):
    """F[k]: dict value->(op,a,b) for values with min formula size k"""
    F=[{v:None for v in base}]
    seen=dict(F[0])
    for k in range(1,K+1):
        new={}
        for i in range(0,(k-1)//2+1):
            j=k-1-i
            for a in F[i]:
                for b in F[j]:
                    for o,v in ops(a,b,lim):
                        if v not in seen and v not in new: new[v]=(o,a,b)
        seen.update(new);F.append(new)
    return F,seen
def expr(v,seen,base):
    if seen[v] is None: return str(v)
    o,a,b=seen[v]
    if o.endswith('r'): o=o[0];a,b=b,a
    return '('+expr(a,seen,base)+o+expr(b,seen,base)+')'
def factor(n):
    f=[];d=2
    while d*d<=n and d<10**6:
        while n%d==0: f.append(d);n//=d
        d+=1
    if n>1:f.append(n)
    return f
import random,math
def is_prime(n):
    if n<2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n%p==0: return n==p
    d=n-1;s=0
    while d%2==0: d//=2;s+=1
    for a in (2,3,5,7,11,13,17,19,23,29,31,37):
        x=pow(a,d,n)
        if x in(1,n-1): continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1: break
        else: return False
    return True
def rho(n):
    if n%2==0: return 2
    while True:
        c=random.randrange(1,n);x=y=random.randrange(n);d=1
        while d==1:
            q=1
            for _ in range(128):
                x=(x*x+c)%n;y=(y*y+c)%n;y=(y*y+c)%n;q=q*abs(x-y)%n
            d=math.gcd(q,n)
        if d!=n: return d
def fac(n):
    r=[]
    def f(n):
        if n==1: return
        if is_prime(n): r.append(n);return
        d=rho(n);f(d);f(n//d)
    for p in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47):
        while n%p==0: r.append(p);n//=p
    f(n);return sorted(r)
import subprocess
def facs(ns):
    out=subprocess.run(['factor']+[str(n) for n in ns],capture_output=True,text=True).stdout
    return [[int(x) for x in l.split(':')[1].split()] for l in out.strip().split('\n')]
def fac(n): return facs([n])[0]

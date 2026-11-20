import random, math
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
    if n%3==0: return 3
    while True:
        c=random.randrange(1,n);y=x=random.randrange(n);g=1;q=1;r=1
        f=lambda v:(v*v+c)%n
        while g==1:
            x=y
            for _ in range(r): y=f(y)
            k=0
            while k<r and g==1:
                ys=y
                for _ in range(min(128,r-k)):
                    y=f(y);q=q*abs(x-y)%n
                g=math.gcd(q,n);k+=128
            r*=2
        if g==n:
            g=1
            while g==1:
                ys=f(ys);g=math.gcd(abs(x-ys),n)
        if g!=n: return g
def factor(n):
    f={}
    def rec(n):
        if n==1: return
        if is_prime(n): f[n]=f.get(n,0)+1;return
        for p in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47):
            if n%p==0:
                f[p]=f.get(p,0)+1; return rec(n//p)
        d=rho(n); rec(d); rec(n//d)
    rec(n); return f

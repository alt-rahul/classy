import sys,subprocess
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
P={p.id:p for p in load_puzzles('practice')}
T=P['practice_0015'].targets
def fac(ns):
    out=subprocess.run(['factor'],input='\n'.join(map(str,ns)),capture_output=True,text=True).stdout.strip().split('\n')
    return [[int(x) for x in l.split(':')[1].split()] for l in out]
def mine(i,p,j,q,A=200000,top=12):
    # Ti = p*H + a, Tj = q*H + b
    res=[]
    a0=T[i]%p
    for a in range(a0-(A//p)*p, A, p):
        H=(T[i]-a)//p
        b=T[j]-q*H
        res.append((abs(a)+abs(b),a,b,H))
    res.sort()
    res=res[:top]
    fs=fac([r[3] for r in res])
    print('T%d=%d*H%+d, T%d=%d*H%+d'%(i,p,0,j,q,0))
    for r,f in zip(res,fs): print('   a=%d b=%d H=%d'%(r[1],r[2],r[3]),f)
mine(5,1,10,30); mine(3,1,17,350); mine(17,25,21,127); mine(17,25,24,279); mine(9,7,15,31); mine(23,14,30,1019); mine(18,1,28,99); mine(0,1,1,1,top=3)

import sys,random
sys.path.insert(0,'/tmp/fable_scratch')
from poolprof import *
rng=random.Random(3)
for B in (16,20,24,28,32,36):
    limit=2**(B+2)
    targets=[rng.randrange(2**(B-1),2**B) for _ in range(30)]
    def rb(lo,hi):
        b=rng.uniform(lo,hi); return max(2,int(2**b))
    pool=[rb(1,5) for _ in range(6)]+[rb(max(3,B/3-3),B/3+1) for _ in range(28)]+[rb(B/2,B) for _ in range(12)]
    test('B=%d mixed-pool'%B,pool,limit,targets)

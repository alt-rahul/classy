import sys,random
sys.path.insert(0,'/tmp/fable_scratch')
from poolprof import *
rng=random.Random(11)
B=int(sys.argv[1]); limit=2**B; nt=int(sys.argv[2])
targets=[rng.randrange(limit//8,limit) for _ in range(nt)]
def rb(lo,hi):
    b=rng.uniform(lo,hi); return max(2,int(2**b))
def mk(spec):
    out=[]
    for n,lo,hi in spec: out+=[rb(lo,hi) for _ in range(n)]
    return out
tests={
 '46 @ B/3-3..B/3+1':[(46,B/3-3,B/3+1)],
 '46 @ B/3-1..B/3':[(46,B/3-1,B/3)],
 '46 @ B/3-5..B/3':[(46,B/3-5,B/3)],
 '46 @ B/4..B/3':[(46,B/4,B/3)],
 '46 @ B/5..B/3':[(46,B/5,B/3)],
 '36 @ B/3-3..B/3+1 + 10 small(2..10b)':[(36,B/3-3,B/3+1),(10,1,10)],
 '30 @ B/3-3..B/3+1 + 16 @ 2B/3-3..2B/3':[(30,B/3-3,B/3+1),(16,2*B/3-3,2*B/3)],
 '23 @ B/3 + 23 @ B/6..B/4':[(23,B/3-3,B/3+1),(23,B/6,B/4)],
 '30 only @ B/3-3..B/3+1':[(30,B/3-3,B/3+1)],
 '20 only @ B/3-3..B/3+1':[(20,B/3-3,B/3+1)],
}
sel=sys.argv[3:]
for k,v in tests.items():
    if sel and not any(s in k for s in sel): continue
    test(k,mk(v),limit,targets)

import sys,json,math,random,time
sys.path.insert(0,'/home/user/classy/elf/src')
import engine
from collections import Counter
def cost(t,budget=1.5e6):
    r=engine.find_pair(t,11); c=r[0] if r else 99
    w=engine.find_window(t,c,budget)
    if w and w[0]<c: c=w[0]
    return c
def test(name,pool,limit,targets,cap=20_000_000,budget=1.5e6):
    pool=sorted(set(pool))
    t0=time.time()
    sz=engine.build(pool,limit,7,0,cap)
    cs=[cost(t,budget) for t in targets]
    print('%-28s n=%d mean %.2f'%(name,len(pool),sum(cs)/len(cs)),sorted(Counter(cs).items()),'%.0fs'%(time.time()-t0),'levels',[('%.0e'%s) for s in sz],flush=True)
    return sum(cs)/len(cs)
if __name__=='__main__':
    rng=random.Random(5)
    B=int(sys.argv[1]); limit=2**B; nt=int(sys.argv[2])
    targets=[rng.randrange(limit//8,limit) for _ in range(nt)]
    def rb(lo,hi): 
        b=rng.uniform(lo,hi); return max(2,int(2**b))
    profs={
     'loguniform 1..B':[rb(1,B) for _ in range(46)],
     'small <=12b':[rb(1,12) for _ in range(46)],
     'all ~B/3':[rb(B/3-3,B/3+1) for _ in range(46)],
     'all ~B/2':[rb(B/2-3,B/2+1) for _ in range(46)],
     'mix 8 small,19 B/3,19 B/2':[rb(1,8) for _ in range(8)]+[rb(B/3-3,B/3+1) for _ in range(19)]+[rb(B/2-3,B/2+1) for _ in range(19)],
     'mix 8 small,38 B/4..B/2':[rb(1,8) for _ in range(8)]+[rb(B/4,B/2) for _ in range(38)],
     'big B-10..B':[rb(1,8) for _ in range(6)]+[rb(B-10,B) for _ in range(40)],
     'mix 6s, 20 B/3..B/2, 20 big':[rb(1,8) for _ in range(6)]+[rb(B/3,B/2) for _ in range(20)]+[rb(B-8,B) for _ in range(20)],
    }
    sel=sys.argv[3:] 
    for k,v in profs.items():
        if sel and not any(s in k for s in sel): continue
        test(k,v,limit,targets)

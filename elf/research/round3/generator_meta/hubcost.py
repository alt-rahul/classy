import sys, numpy as np, io, contextlib, collections
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from exact_small import costs
from exact_wit import decomp
def allexpr(v,cost,L,S,memo,cap=8):
    if cost[v]==0: return [str(v)]
    if v in memo: return memo[v]
    out=[]
    for op,a,b in decomp(v,cost,L,S):
        for ea in allexpr(a,cost,L,S,memo,cap)[:2]:
            for eb in allexpr(b,cost,L,S,memo,cap)[:2]:
                out.append('(%s%s%s)'%(ea,op,eb))
        if len(out)>=cap: break
    memo[v]=out[:cap]; return memo[v]
pid=sys.argv[1]; LIM=int(float(sys.argv[2])); maxk=int(sys.argv[3]); vals=[int(x) for x in sys.argv[4:]]
for p in load_puzzles('practice'):
    if p.id[-4:]==pid:
        with contextlib.redirect_stdout(io.StringIO()): c=costs(p.sv,min(LIM,p.limit),maxk)
        L=min(LIM,p.limit)
        S=[np.nonzero(c==k)[0].astype(np.int64) for k in range(maxk+1)]; S[0]=S[0][S[0]>0]
        memo={}
        for v in vals:
            lo=max(1,v-2000); nb=collections.Counter(c[lo:v+2000].tolist())
            print(v,'cost',c[v],'neighbourhood',sorted(nb.items()))
            if c[v]<99: print('    ',' | '.join(allexpr(v,c,L,S,memo)))

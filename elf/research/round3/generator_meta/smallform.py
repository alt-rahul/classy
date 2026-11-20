"""min tree formulas (all of them up to a cap) for small targets, from starts only"""
import sys, numpy as np
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
from exact_small import costs
from exact_wit import decomp
import io,contextlib
def allexpr(v,cost,L,S,memo,cap=6):
    if cost[v]==0: return [str(v)]
    if v in memo: return memo[v]
    out=[]
    for op,a,b in decomp(v,cost,L,S):
        for ea in allexpr(a,cost,L,S,memo,cap)[:2]:
            for eb in allexpr(b,cost,L,S,memo,cap)[:2]:
                out.append('(%s%s%s)'%(ea,op,eb))
        if len(out)>=cap: break
    memo[v]=out[:cap]; return memo[v]

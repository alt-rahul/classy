import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../src'))
from common import *
OPF = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b, '/': lambda a, b: a // b}
def values(p, prog):
    """list of (op,a,b,v) in program order (values, not registers)"""
    regs = list(p.sv) + [None] * (p.nreg - len(p.sv))
    out = []
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]; v = OPF[op](a, b); regs[d] = v
        out.append((op, a, b, v))
    return out
def dag(p, prog):
    """value -> (op,a,b,step) first definition"""
    nodes = {v: (None, None, None, -1) for v in p.sv}
    for k, (op, a, b, v) in enumerate(values(p, prog)):
        nodes.setdefault(v, (op, a, b, k))
    return nodes

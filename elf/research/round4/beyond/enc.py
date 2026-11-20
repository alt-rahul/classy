"""encode store programs for the public elfopt engine: enc.py REGS id1,id2,... > file"""
import sys
from dagutil import *
P={p.id:p for p in load_puzzles()}; st=load_store()
regs=int(sys.argv[1])
for pid in sys.argv[2].split(','):
    p=P[pid]; prog=st[pid]['prog']
    s=['P',p.id,str(len(p.sv))]+[str(x) for x in p.sv]+[str(len(p.targets))]+[str(x) for x in p.targets]+[str(p.limit),str(regs if regs else p.nreg),str(len(prog))]
    print(' '.join(s)); 
    for x in prog: print('%s %d %d %d'%tuple(x))

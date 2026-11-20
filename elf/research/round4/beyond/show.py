from dagutil import *
import json,sys
fam=json.load(open('fam.json'))
P={p.id:p for p in load_puzzles()}; st=load_store()
def show(pid):
    p=P[pid]; prog=st[pid]['prog']
    print(pid,fam[pid],p.sv,len(p.targets),p.limit.bit_length(),len(prog), round(len(prog)/len(p.targets),2))
    T=set(p.targets); vals=values(p,prog); fan={}
    for op,a,b,v in vals:
        fan[a]=fan.get(a,0)+1; fan[b]=fan.get(b,0)+1
    for k,(op,a,b,v) in enumerate(vals):
        print(k,a,op,b,'->',v,'b%d'%v.bit_length(),'f%d'%fan.get(v,0),'T' if v in T else '')
if __name__=='__main__':
    for x in sys.argv[1:]: show(x)

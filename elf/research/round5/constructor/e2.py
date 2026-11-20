# E2: at step t of the stored program, one-shot recipe selection over the static live pool vs what the store spends
import sys, time
from rlib import *
st=load_store(); P={p.id:p for p in load_puzzles('scored')}
pid=sys.argv[1]; t=int(sys.argv[2]); cmax=int(sys.argv[3]) if len(sys.argv)>3 else 8
p=P[pid]; prog=st[pid]['prog']
pool,rem=state_at(p,prog,t)
t0=time.time(); D,R,err=enum(p.limit,pool,rem,cmax)
print(err.split('\n')[0], 'enum %.1fs'%(time.time()-t0))
have=[T for T in rem if R[T]]
mins=[min(r[0] for r in R[T]) for T in have]
print(pid,'t',t,'pool',len(pool),'remaining',len(rem),'with recipes',len(have),'store spends',len(prog)-t,'=%.2f/target'%((len(prog)-t)/len(rem)))
print('min tree cost hist',sorted(collections.Counter(mins).items()),'sum',sum(mins), 'recipes/target median',sorted(len(R[T]) for T in have)[len(have)//2])
t0=time.time(); tot,ch,hist=select(D,pool,R,have)
print('union after selection',tot,'hist',hist,'= %.2f/target'%(tot/len(have)),'sel %.1fs'%(time.time()-t0))

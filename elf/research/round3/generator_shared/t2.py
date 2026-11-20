from lib import *
import sys
ps={p.id:p for p in P()}
p=ps[sys.argv[1]]
print(p.family,p.sv,p.limit,p.limit.bit_length())
for i,(t,f) in enumerate(zip(p.targets,facs(p.targets))): print(i,t,t.bit_length(),f)

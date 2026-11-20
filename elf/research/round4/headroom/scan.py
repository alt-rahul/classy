import sys,subprocess
sys.path.insert(0,'/home/user/classy/elf/src')
from common import *
P={p.id:p for p in load_puzzles('practice')}
p=P[sys.argv[1]]; C=int(sys.argv[2]); B=int(sys.argv[3])
nums=[]
for i,t in enumerate(p.targets):
    for c in range(-C,C+1):
        if t+c>1: nums.append((i,c,t+c))
out=subprocess.run(['factor'],input='\n'.join(str(n) for _,_,n in nums),capture_output=True,text=True).stdout.split('\n')
for (i,c,n),l in zip(nums,out):
    f=[int(x) for x in l.split(':')[1].split()]
    if max(f)<=B: print(i,c,n,f)

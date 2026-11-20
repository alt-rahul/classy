from chain import *
sv=eval(sys.argv[1]);D=int(sys.argv[2]);q=eval(sys.argv[3]);lim=int(sys.argv[4])
lev,seen=fwd(sv,lim,D)
print([len(l) for l in lev])
for t in q:
    d=next((i for i,l in enumerate(lev) if t in l),None)
    print(t,d,path(t,seen) if d is not None else '')

import csv,json,sys,statistics as S
csv.field_size_limit(10**9)
K='/tmp/claude-0/-home-user-classy/1d160dbb-3e03-591e-a40c-f5c8c1649727/scratchpad/kg/'
files={'asterios':'ds_asteriospanoras_santa-2026-elf-programs-and-experiments/best_programs.csv','rudolph':'ds_alexchilton_santa-2026-rudolph-programs/rudolph_programs.csv','gp_front':'ds_goodpjw2008_santa-2026-dag-lns-programs/public_frontier.csv','gp_ours':'ds_goodpjw2008_santa-2026-dag-lns-programs/our_programs.csv','v61':'ds_royalacecat_santa-2026-retained-replay-inputs-133974/baseline_v61_133998.csv','ast992':'ds_royalacecat_santa-2026-retained-replay-inputs-133974/asterios_133992.csv'}
L={}
for k,f in files.items():
    rows=list(csv.reader(open(K+f)))
    hdr=rows[0]; d={}
    for r in rows[1:]:
        try:
            pid=r[0]; cell=[c for c in r[1:] if c.startswith('[[')]
            if cell: d[pid]=len(json.loads(cell[0]))
        except Exception as e: pass
    L[k]=d; print(k,hdr[:4],len(d),sum(d.values()) if len(d)>=1000 else '')
rows=json.load(open('cone_rows.json')); pure={r[0]:r for r in rows if not r[2]}
store=json.load(open('../../../store/best.json'))
for a in ('asterios','rudolph','gp_ours'):
    ex=[]; 
    for pid in pure:
        if pid in L[a]: ex.append(L[a][pid]-store[pid]['len'])
    if ex: print(a,'pure puzzles',len(ex),'mean excess over store %.2f'%S.mean(ex),'ties',sum(1 for e in ex if e==0),'>=3:',sum(1 for e in ex if e>=3),'>=6:',sum(1 for e in ex if e>=6), 'sum',sum(ex))

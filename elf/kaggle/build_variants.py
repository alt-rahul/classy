"""Build kaggle/build/variants.csv: per puzzle, up to K distinct valid programs from public banks whose length is
within SLACK of the store program (crossover seeds for the *_seeded WJ strategies; the runner also adds its own).
usage: python3 kaggle/build_variants.py [--k 4] [--slack 3] bank1.csv bank2.csv ...
Banks are CSVs with columns id,program (others ignored). Every program is validated before it is written.
"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
from common import load_puzzles, load_store, check  # noqa: E402
csv.field_size_limit(10 ** 9)

args = sys.argv[1:]
K, SLACK = 4, 3
if '--k' in args:
    i = args.index('--k'); K = int(args[i + 1]); del args[i:i + 2]
if '--slack' in args:
    i = args.index('--slack'); SLACK = int(args[i + 1]); del args[i:i + 2]
P = {p.id: p for p in load_puzzles()}
st = load_store()
seen = {k: {json.dumps(v['prog'], separators=(',', ':'))} for k, v in st.items()}
cand = {k: [] for k in P}
for path in args:
    try:
        rows = csv.DictReader(open(path, newline='', encoding='utf-8-sig'))
        n = 0
        for r in rows:
            k = r.get('id')
            if k not in P or not r.get('program'):
                continue
            try:
                prog = json.loads(r['program'])
            except ValueError:
                continue
            if not isinstance(prog, list) or len(prog) > st[k]['len'] + SLACK:
                continue
            key = json.dumps(prog, separators=(',', ':'))
            if key in seen[k]:
                continue
            seen[k].add(key)
            if check(P[k], prog) is None:
                cand[k].append((len(prog), key)); n += 1
        print('%s: %d new variants' % (os.path.basename(path), n))
    except Exception as e:
        print('skip', path, e)
out = os.path.join(HERE, 'build', 'variants.csv')
tot = 0
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['id', 'program'])
    for k in P:
        for _, key in sorted(cand[k])[:K]:
            w.writerow([k, key]); tot += 1
print('wrote', out, tot, 'variants for', sum(1 for k in P if cand[k]), 'puzzles')

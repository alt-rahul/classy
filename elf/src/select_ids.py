"""Print a comma list of scored puzzle ids from the store, filtered by class and base tag.
usage: select_ids.py [--classes c1,c2] [--tags t1,t2] [--not-tags t1,t2]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_puzzles, load_store, pclass
ap = argparse.ArgumentParser()
ap.add_argument('--classes', default='')
ap.add_argument('--tags', default='')
ap.add_argument('--not-tags', default='')
a = ap.parse_args()
st = load_store()
cl = set(a.classes.split(',')) if a.classes else None
tg = set(a.tags.split(',')) if a.tags else None
nt = set(a.not_tags.split(',')) if a.not_tags else set()
out = []
for p in load_puzzles('scored'):
    base = st[p.id]['tag'].split('+')[0]
    if cl and pclass(p) not in cl:
        continue
    if tg and base not in tg:
        continue
    if base in nt:
        continue
    out.append(p.id)
print(','.join(out))

"""Merge external submission CSVs (id,program) into the store, keeping the shortest valid program per puzzle.

Usage: python3 src/merge_public.py [--dry] name=path.csv [name=path.csv ...]
Every program is re-validated with common.check (via offer) before it enters the store.
Prints, per source: valid count, its own total, and how many puzzles it improves.
"""
import csv, json, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from common import load_puzzles, load_store, save_store, offer, check

csv.field_size_limit(10**9)


def read_csv(path):
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out[row['id']] = json.loads(row['program'])
            except Exception:
                pass
    return out


def main(argv):
    dry = '--dry' in argv
    srcs = [a.split('=', 1) for a in argv if '=' in a]
    puzzles = load_puzzles('scored')
    st = load_store()
    before = sum(v['len'] for v in st.values())
    print(f'store before {before}')
    for name, path in srcs:
        progs = read_csv(path)
        valid = tot = wins = gain = 0
        bad = []
        for p in puzzles:
            prog = progs.get(p.id)
            if prog is None:
                bad.append(p.id); continue
            if check(p, prog):
                bad.append(p.id); continue
            valid += 1
            tot += len(prog)
            cur = st[p.id]['len']
            if len(prog) < cur:
                if offer(st, p, prog, 'pub:' + name):
                    wins += 1
                    gain += cur - len(prog)
        print(f'{name}: valid {valid}/1000 total {tot if valid == 1000 else "n/a"} '
              f'improves {wins} puzzles by {gain}; invalid/missing {len(bad)} {bad[:5]}')
    after = sum(v['len'] for v in st.values())
    print(f'store after {after} (gain {before - after})')
    if not dry:
        save_store(st)
        print('saved')


if __name__ == '__main__':
    main(sys.argv[1:])

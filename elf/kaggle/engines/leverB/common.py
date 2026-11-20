"""Shared helpers: puzzle loading, classification, best-solution store, validation, CSV writer."""
import csv, json, os, sys, time
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from validate import run as vrun  # independent validator (do not edit validate.py)

DATA = os.path.join(ROOT, 'data')
STORE = os.path.join(ROOT, 'store', 'best.json')


class Puzzle:
    __slots__ = ('id', 'sv', 'targets', 'limit', 'nreg', 'family')

    def __init__(self, row):
        self.id = row['id']
        self.sv = json.loads(row['starting_values'])
        self.targets = json.loads(row['targets'])
        self.limit = int(row['value_limit'])
        self.nreg = int(row['registers'])
        self.family = row.get('family')


def load_puzzles(which='scored'):
    fn = {'scored': 'puzzles.csv', 'practice': 'practice_puzzles.csv'}[which]
    return [Puzzle(r) for r in csv.DictReader(open(os.path.join(DATA, fn)))]


def pclass(p):
    if p.limit <= 16:
        return 'tiny'
    ind = max(p.sv) > 40
    return ('indep' if ind else 'other') + ('>=20' if len(p.targets) >= 20 else '<20')


def check(p, prog):
    """None if valid, else error string."""
    return vrun(p.sv, p.targets, p.limit, prog, p.nreg)


def load_store():
    if os.path.exists(STORE):
        return json.load(open(STORE))
    return {}


def save_store(st):
    """Merge-safe save: under a lock, re-read the file and keep the shorter program per id
    (both sides only ever hold validated programs). Updates st in place with the merge."""
    import fcntl
    lockf = open(STORE + '.lock', 'w')
    fcntl.flock(lockf, fcntl.LOCK_EX)
    try:
        if os.path.exists(STORE):
            disk = json.load(open(STORE))
            for k, rec in disk.items():
                if k not in st or rec['len'] < st[k]['len']:
                    st[k] = rec
        tmp = STORE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(st, f, separators=(',', ':'))
        os.replace(tmp, STORE)
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def offer(st, p, prog, tag):
    """Insert prog into store if valid and strictly shorter. Returns True if improved."""
    cur = st.get(p.id)
    if cur is not None and len(prog) >= cur['len']:
        return False
    err = check(p, prog)
    if err:
        return False
    cell = json.dumps(prog, separators=(',', ':'))
    if len(cell.encode()) > 131072:
        return False
    st[p.id] = {'len': len(prog), 'prog': prog, 'tag': tag}
    return True


def seed_from_sample(st, puzzles):
    rows = list(csv.reader(open(os.path.join(DATA, 'sample_submission.csv'), encoding='utf-8')))[1:]
    P = {p.id: p for p in puzzles}
    n = 0
    for pid, cell in rows:
        if offer(st, P[pid], json.loads(cell), 'sample'):
            n += 1
    return n


def breakdown(st, puzzles):
    agg = {}
    for p in puzzles:
        c = pclass(p)
        a = agg.setdefault(c, [0, 0, 0])
        a[0] += 1
        a[1] += len(p.targets)
        a[2] += st[p.id]['len'] if p.id in st else 0
    return agg


def write_submission(st, puzzles, path):
    tot = 0
    bad = 0
    rows = []
    for p in puzzles:
        rec = st[p.id]
        prog = rec['prog']
        err = check(p, prog)
        cell = json.dumps(prog, separators=(',', ':'))
        if err or len(cell.encode()) > 131072:
            bad += 1
            print('INVALID', p.id, err)
        tot += len(prog)
        rows.append((p.id, cell))
    if bad:
        raise SystemExit('refusing to write: %d invalid' % bad)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'program'])
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)
    return tot


def print_breakdown(st, puzzles):
    agg = breakdown(st, puzzles)
    tot = 0
    for c in sorted(agg):
        n, nt, ops = agg[c]
        tot += ops
        print('  %-9s puzzles %4d targets %6d ops %8d ops/target %.2f' % (c, n, nt, ops, ops / max(nt, 1)))
    print('  total', tot)
    return tot

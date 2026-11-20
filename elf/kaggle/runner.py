"""Elf compute runner: improve a shard of Santa 2026 programs with native engines for a time budget.

Runs inside a private Kaggle script kernel (fleet.py prepends `CONFIG = {...}` to this file and pushes it),
or locally for tests:  python3 runner.py --config cfg.json --input DIR [DIR ...] --out OUTDIR
Inputs: puzzles.csv (competition data), programs.csv and engine sources (our private dataset); CONFIG may
also embed fresh programs for its shard. Every program an engine returns is replayed by an exact referee
(same rules as src/validate.py) and kept only if strictly shorter. Outputs, rewritten every few minutes:
  improved.csv  id,len,base_len,engine,program   (only puzzles that got shorter)
  stats.json    per-engine runs / cpu seconds / gains, per-puzzle time spent
Engine protocols:
  elf  (goodpjw elfopt and compatible): stdin "P id k s.. m t.. limit regs n" + n lines "op i j d";
       stdout "R id n_old n_new" + n_new lines, only when strictly shorter. Gets --time and --seed.
  wj   (Wang Jakie strategies): stdin count, then per puzzle "limit regs rounds 0 seed", starts, targets,
       ntapes, tapes; stdout one line "n op a b d ..." per puzzle.
"""
import csv, glob, json, os, random, shutil, subprocess, sys, threading, time

csv.field_size_limit(10 ** 9)
T_START = time.time()


def log(*a):
    print('[%7.1f min]' % ((time.time() - T_START) / 60), *a, flush=True)


# ------------------------------------------------------------------ referee (rules of src/validate.py)
def referee(sv, targets, limit, prog, nreg=48):
    if not isinstance(prog, list): return 'not list'
    if len(prog) > 4096: return 'too long'
    reg = [None] * nreg
    for i, v in enumerate(sv): reg[i] = v
    need = set(targets) - set(sv)
    for k, ins in enumerate(prog):
        if not (isinstance(ins, list) and len(ins) == 4): return 'bad ins %d' % k
        op, i, j, d = ins
        if not isinstance(op, str) or len(op) != 1 or op not in '+-*/': return 'bad op %d' % k
        for x in (i, j, d):
            if type(x) is not int or not 0 <= x < nreg: return 'bad reg %d' % k
        a, b = reg[i], reg[j]
        if a is None or b is None: return 'undef %d' % k
        if op == '+': r = a + b
        elif op == '-':
            if a <= b: return 'neg %d' % k
            r = a - b
        elif op == '*': r = a * b
        else:
            if a % b: return 'inexact %d' % k
            r = a // b
        if r > limit or r < 1: return 'limit %d' % k
        reg[d] = r
        need.discard(r)
    if need: return 'missing %d' % len(need)
    if len(json.dumps(prog, separators=(',', ':')).encode()) > 131072: return 'cell too big'
    return None


class Puz:
    __slots__ = ('id', 'sv', 'targets', 'limit', 'nreg')

    def __init__(self, r):
        self.id = r['id']; self.sv = json.loads(r['starting_values']); self.targets = json.loads(r['targets'])
        self.limit = int(r['value_limit']); self.nreg = int(r['registers'])

    def check(self, prog):
        return referee(self.sv, self.targets, self.limit, prog, self.nreg)


def find_file(dirs, name):
    for d in dirs:
        hits = sorted(glob.glob(os.path.join(d, '**', name), recursive=True))
        if hits: return hits[0]
    return None


def read_programs(path):
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            out[row['id']] = json.loads(row['program'])
    return out


# ------------------------------------------------------------------ engines
def build_engines(cfg, dirs, bindir):
    src = os.path.join(bindir, 'src'); os.makedirs(src, exist_ok=True)
    built = {}
    jobs = []
    for e in cfg['engines']:
        name = e['name']
        if name in built: continue
        files = e.get('sources') or [e['main']]
        for fn in files + e.get('includes', []):
            p = find_file(dirs, fn)
            if p is None: raise SystemExit('engine source %s not found' % fn)
            shutil.copy(p, os.path.join(src, fn))
        exe = os.path.join(bindir, name)
        cmd = e.get('compile') or ['g++', '-std=c++17', '-O3', '-march=native', '-o', exe, os.path.join(src, e['main'])]
        cmd = [c.replace('{exe}', exe).replace('{src}', src) for c in cmd]
        jobs.append((name, exe, subprocess.Popen(cmd, cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        built[name] = exe
    for name, exe, pr in jobs:
        out, _ = pr.communicate()
        if pr.returncode != 0 or not os.path.exists(exe):
            raise SystemExit('compile %s failed: %s' % (name, out[-2000:]))
    log('built engines', sorted(built))
    return built


def enc_elf(p, prog):
    s = ['P', p.id, str(len(p.sv))] + [str(x) for x in p.sv] + [str(len(p.targets))] + [str(x) for x in p.targets]
    s += [str(p.limit), str(p.nreg), str(len(prog))]
    return '\n'.join([' '.join(s)] + ['%s %d %d %d' % tuple(x) for x in prog]) + '\n'


def parse_elf(text):
    res, lines, k = {}, text.split('\n'), 0
    while k < len(lines):
        ln = lines[k].split()
        if ln and ln[0] == 'R' and len(ln) >= 4:
            pid, n = ln[1], int(ln[3])
            try:
                res[pid] = [[o, int(i), int(j), int(d)] for o, i, j, d in (lines[k + 1 + t].split() for t in range(n))]
            except (ValueError, IndexError):
                pass
            k += n + 1
        else:
            k += 1
    return res


def enc_wj(p, tapes, rounds, seed):
    rows = ['%d %d %d 0 %d' % (p.limit, p.nreg, rounds, seed), ' '.join(map(str, (len(p.sv), *p.sv))),
            ' '.join(map(str, (len(set(p.targets)), *sorted(set(p.targets))))), str(len(tapes))]
    for t in tapes:
        rows.append(str(len(t))); rows += ['%s %d %d %d' % (op, x, y, d) for op, x, y, d in t]
    return '\n'.join(rows)


def parse_wj(line):
    tok = line.split()
    if not tok: return None
    try:
        return [[tok[i], int(tok[i + 1]), int(tok[i + 2]), int(tok[i + 3])] for i in range(1, len(tok), 4)]
    except (ValueError, IndexError):
        return None


def run_engine(e, exe, p, prog, variants, seed, deadline):
    """Returns (program or None, cpu seconds used)."""
    t0 = time.time()
    kind = e.get('kind', 'elf')
    if kind == 'elf':
        tl = min(float(e.get('time', 60)), max(5.0, deadline - time.time() - 30))
        cmd = [exe, '--time', '%.1f' % tl, '--seed', str(seed)] + list(e.get('args', []))
        inp = enc_elf(p, prog)
        timeout = tl * 2 + 120
    else:
        cmd = [exe] + list(e.get('args', []))
        tapes = [prog] + (variants[:int(e.get('max_seeds', 4))] if e.get('seeded') else [])
        inp = '1\n' + enc_wj(p, tapes, int(e.get('rounds', 700)), seed % (2 ** 32)) + '\n'
        timeout = max(60.0, min(float(e.get('timeout', 900)), deadline - time.time()))
    try:
        r = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=timeout)
        out = r.stdout
    except subprocess.TimeoutExpired:
        return None, time.time() - t0
    new = parse_elf(out).get(p.id) if kind == 'elf' else parse_wj(out.split('\n')[0] if out else '')
    return new, time.time() - t0


# ------------------------------------------------------------------ main loop
class State:
    def __init__(self, P, ids, start, outdir, cfg):
        self.P, self.ids, self.outdir, self.cfg = P, ids, outdir, cfg
        self.base = {k: start[k] for k in ids}
        self.best = dict(self.base)
        self.who = {}
        self.variants = {k: [] for k in ids}
        self.spent = {k: 0.0 for k in ids}
        self.busy = set()
        self.passes = {k: 0 for k in ids}
        self.lock = threading.Lock()
        self.stats = {e['name']: {'runs': 0, 'cpu_s': 0.0, 'gain': 0, 'hits': 0, 'invalid': 0} for e in cfg['engines']}
        self.timeline = []

    def pick(self):
        with self.lock:
            free = [k for k in self.ids if k not in self.busy]
            if not free: return None
            # least-visited first, longer programs first within a pass
            k = min(free, key=lambda x: (self.passes[x], -len(self.best[x]), x))
            self.busy.add(k); self.passes[k] += 1
            return k, self.passes[k], [list(map(list, v)) for v in self.variants[k]], self.best[k]

    def report(self, k, ename, new, cpu):
        with self.lock:
            self.busy.discard(k)
            st = self.stats[ename]; st['runs'] += 1; st['cpu_s'] += cpu; self.spent[k] += cpu
            if new is None: return
            cur = self.best[k]
            if len(new) >= len(cur):
                return
            err = self.P[k].check(new)
            if err:
                st['invalid'] += 1; log('INVALID from', ename, k, err); return
            st['gain'] += len(cur) - len(new); st['hits'] += 1
            self.variants[k] = ([cur] + self.variants[k])[:6]
            self.best[k] = new; self.who[k] = ename
            self.timeline.append([round(time.time() - T_START), k, ename, len(cur), len(new)])
            log('GAIN %s %s %d -> %d (base %d)' % (k, ename, len(cur), len(new), len(self.base[k])))

    def write(self, final=False):
        with self.lock:
            rows = [(k, len(self.best[k]), len(self.base[k]), self.who.get(k, ''), self.best[k])
                    for k in self.ids if len(self.best[k]) < len(self.base[k])]
            stats = {'engines': self.stats, 'elapsed_s': time.time() - T_START, 'final': final,
                     'shard': self.ids, 'spent_s': self.spent, 'timeline': self.timeline,
                     'gain': sum(r[2] - r[1] for r in rows), 'tag': self.cfg.get('tag', '')}
        tmp = os.path.join(self.outdir, '.improved.csv.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['id', 'len', 'base_len', 'engine', 'program'])
            for k, n, b, e, prog in rows:
                w.writerow([k, n, b, e, json.dumps(prog, separators=(',', ':'))])
        os.replace(tmp, os.path.join(self.outdir, 'improved.csv'))
        tmp = os.path.join(self.outdir, '.stats.json.tmp')
        with open(tmp, 'w') as f: json.dump(stats, f)
        os.replace(tmp, os.path.join(self.outdir, 'stats.json'))
        return stats


def main(cfg, dirs, outdir):
    os.makedirs(outdir, exist_ok=True)
    bindir = cfg.get('bindir') or '/tmp/elfbin'
    os.makedirs(bindir, exist_ok=True)
    pz = find_file(dirs, 'puzzles.csv')
    if not pz: raise SystemExit('puzzles.csv not found in %s' % dirs)
    with open(pz, newline='', encoding='utf-8-sig') as f:
        P = {r['id']: Puz(r) for r in csv.DictReader(f)}
    start = {}
    pp = find_file(dirs, cfg.get('programs_file', 'programs.csv'))
    if pp: start = read_programs(pp)
    for k, prog in (cfg.get('programs') or {}).items():      # fresh programs embedded at push time
        if k not in start or len(prog) < len(start[k]): start[k] = prog
    ids = [k for k in cfg['ids'] if k in P and k in start]
    bad = [k for k in ids if P[k].check(start[k])]
    if bad: log('start programs invalid, dropped:', bad)
    ids = [k for k in ids if k not in bad]
    log('shard %d puzzles, %d instructions, budget %.0f s, cpus %s' % (
        len(ids), sum(len(start[k]) for k in ids), cfg['budget_s'], os.cpu_count()))
    exes = build_engines(cfg, dirs, bindir)
    S = State(P, ids, start, outdir, cfg)
    vp = find_file(dirs, cfg.get('variants_file', 'variants.csv'))
    if vp:   # near-length public variants: crossover seeds for the seeded strategies
        nv = 0
        with open(vp, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                k = row['id']
                if k not in S.variants: continue
                try: prog = json.loads(row['program'])
                except ValueError: continue
                if prog != S.best[k] and len(prog) <= len(S.best[k]) + 3 and not P[k].check(prog):
                    S.variants[k].append(prog); nv += 1
        log('loaded %d variants' % nv)
    S.write()
    deadline = T_START + float(cfg['budget_s'])
    engines = cfg['engines']
    weights = [float(e.get('weight', 1)) for e in engines]
    rng = random.Random(cfg.get('seed', 1))
    seed_lock = threading.Lock()

    def worker(wi):
        while time.time() < deadline - float(cfg.get('min_task_s', 20)):
            got = S.pick()
            if got is None:
                time.sleep(1); continue
            k, npass, variants, prog = got
            with seed_lock:
                e = engines[rng.choices(range(len(engines)), weights)[0]] if cfg.get('random_engine', True) \
                    else engines[(npass - 1) % len(engines)]
                seed = rng.randrange(1, 2 ** 31)
            try:
                new, cpu = run_engine(e, exes[e['name']], P[k], prog, variants, seed, deadline)
            except Exception as ex:  # keep the worker alive
                log('engine error', e['name'], k, repr(ex)[:200]); new, cpu = None, 1.0
            S.report(k, e['name'], new, cpu)

    nw = int(cfg.get('workers') or os.cpu_count() or 4)
    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(nw)]
    for t in th: t.start()
    last = time.time()
    while any(t.is_alive() for t in th):
        time.sleep(5)
        if time.time() - last > float(cfg.get('write_every_s', 180)):
            st = S.write(); last = time.time()
            log('progress gain %d, runs %s' % (st['gain'], {n: v['runs'] for n, v in st['engines'].items()}))
        if time.time() > deadline + 600:
            log('workers overran the deadline, stopping'); break
    st = S.write(final=True)
    log('done: gain %d on %d puzzles; engines %s' % (st['gain'], sum(1 for k in ids if len(S.best[k]) < len(S.base[k])),
                                                    json.dumps(st['engines'])))
    shutil.rmtree(bindir, ignore_errors=True)


if __name__ == '__main__':
    if 'CONFIG' in globals():
        main(CONFIG, ['/kaggle/input'], '/kaggle/working')
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('--config', required=True); ap.add_argument('--input', nargs='+', required=True)
        ap.add_argument('--out', required=True)
        a = ap.parse_args()
        main(json.load(open(a.config)), a.input, a.out)

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
    """Copy every engine's sources into one directory first, then compile in parallel (retry failures serially).
    An engine that still fails to build is dropped from the run instead of aborting it."""
    src = os.path.join(bindir, 'src'); os.makedirs(src, exist_ok=True)
    todo = {}
    for e in cfg['engines']:
        name = e['name']
        if name in todo: continue
        ok = True
        for fn in (e.get('sources') or [e['main']]) + e.get('includes', []):
            p = find_file(dirs, fn)
            if p is None:
                log('engine %s: source %s not found, dropped' % (name, fn)); ok = False; break
            dst = os.path.join(src, fn)
            if not os.path.exists(dst): shutil.copy(p, dst)
        if ok: todo[name] = e

    def cmd_for(name, e):
        exe = os.path.join(bindir, name)
        cmd = e.get('compile') or ['g++', '-std=c++17', '-O3', '-march=native', '-o', exe, os.path.join(src, e['main'])]
        return exe, [c.replace('{exe}', exe).replace('{src}', src) for c in cmd]

    built, failed = {}, {}
    names = list(todo)
    nj = max(1, int(cfg.get('build_jobs') or os.cpu_count() or 4))
    for b in range(0, len(names), nj):
        jobs = []
        for name in names[b:b + nj]:
            exe, cmd = cmd_for(name, todo[name])
            jobs.append((name, exe, subprocess.Popen(cmd, cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        for name, exe, pr in jobs:
            out, _ = pr.communicate()
            if pr.returncode == 0 and os.path.exists(exe): built[name] = exe
            else: failed[name] = out
    for name in list(failed):
        exe, cmd = cmd_for(name, todo[name])
        r = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(exe):
            built[name] = exe; del failed[name]
        else:
            log('compile %s failed, engine dropped: %s' % (name, (r.stdout + r.stderr)[-1500:]))
    if not built: raise SystemExit('no engine could be built')
    log('built engines', sorted(built))
    return built


def enc_elf(p, prog):
    s = ['P', p.id, str(len(p.sv))] + [str(x) for x in p.sv] + [str(len(p.targets))] + [str(x) for x in p.targets]
    s += [str(p.limit), str(p.nreg), str(len(prog))]
    return '\n'.join([' '.join(s)] + ['%s %d %d %d' % tuple(x) for x in prog]) + '\n'


def parse_elf(text, variants=None):
    """R blocks (strictly shorter programs) -> {id: prog}; V blocks (equal-length plateau programs) appended to
    variants as (id, prog) when a list is given."""
    res, lines, k = {}, text.split('\n'), 0
    while k < len(lines):
        ln = lines[k].split()
        if ln and ln[0] in ('R', 'V') and len(ln) >= 3:
            pid, n = ln[1], int(ln[3] if ln[0] == 'R' else ln[2])
            try:
                prog = [[o, int(i), int(j), int(d)] for o, i, j, d in (lines[k + 1 + t].split() for t in range(n))]
                if ln[0] == 'R': res[pid] = prog
                elif variants is not None: variants.append((pid, prog))
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


def run_engine(e, exe, p, prog, variants, seed, deadline, newvars):
    """Returns (program or None, cpu seconds used); equal-length donor programs are appended to newvars."""
    t0 = time.time()
    kind = e.get('kind', 'elf')
    if kind == 'recomb':
        return run_recomb(e, exe, p, prog, variants, seed, deadline, t0, newvars)
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
    vv = []
    new = parse_elf(out, vv).get(p.id) if kind == 'elf' else parse_wj(out.split('\n')[0] if out else '')
    newvars.extend(q for k, q in vv if k == p.id)
    return new, time.time() - t0


# ------------------------------------------------------------------ CP-SAT recombination
# Adapted from goodpjw2008's recombine.py (dataset goodpjw2008/santa-2026-dag-lns-programs, Apache-2.0):
# pool the recipes v = a op b seen by an LNS run (--hall 1 --dump) and in known programs, add every cross recipe
# between pooled values, and let CP-SAT pick the fewest values (one recipe each, acyclic via integer levels)
# that still build every target; then list-schedule into the registers.
OPS = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b, '/': lambda a, b: a // b}


def program_recipes(sv, nreg, prog):
    regs = list(sv) + [None] * (nreg - len(sv)); out = set()
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]; v = OPS[op](a, b); regs[d] = v
        out.add((op, a, b, v))
    return out


def read_dump(path):
    out = set()
    try:
        lines = open(path).read().split('\n')
    except OSError:
        return out
    k = 0
    while k < len(lines):
        ln = lines[k].split()
        if ln and ln[0] == 'D':
            n = int(ln[2])
            for t in range(n):
                try:
                    op, a, b, v = lines[k + 1 + t].split(); out.add((op, int(a), int(b), int(v)))
                except (ValueError, IndexError):
                    pass
            k += n + 1
        else:
            k += 1
    return out


def cross_recipes(limit, values):
    vs = sorted(values); S = set(vs); out = set()
    for i, a in enumerate(vs):
        for b in vs[i:]:
            s = a + b
            if s in S: out.add(('+', a, b, s))
            m = a * b
            if m <= limit and m in S: out.add(('*', a, b, m))
            if b > a and b - a in S: out.add(('-', b, a, b - a))
            if b != a and b % a == 0 and b // a in S: out.add(('/', b, a, b // a))
    return out


def schedule(sv, nreg, chosen, level):
    import collections
    starts = list(sv); sset = set(starts)
    users = collections.defaultdict(list)
    for v, (op, a, b) in chosen.items():
        users[a].append(v)
        if b != a: users[b].append(v)
    remaining = {v: len(users[v]) for v in list(chosen) + starts}
    indeg = {v: sum(1 for x in {chosen[v][1], chosen[v][2]} if x not in sset) for v in chosen}
    ready = [v for v in chosen if indeg[v] == 0]
    done = set(sset); order = []
    while ready:
        def key(v):
            op, a, b = chosen[v]
            fr = (remaining[a] == 1) + (b != a and remaining[b] == 1)
            return ((1 if users[v] else 0) - fr, level.get(v, 0))
        ready.sort(key=key); v = ready.pop(0); order.append(v); done.add(v)
        op, a, b = chosen[v]
        remaining[a] -= 1
        if b != a: remaining[b] -= 1
        for u in users[v]:
            indeg[u] -= 1
        for u in users[v]:
            if indeg[u] == 0 and u not in done and u not in ready:
                if chosen[u][1] in done and chosen[u][2] in done: ready.append(u)
    if len(order) != len(chosen): return None
    last = {}
    for k, v in enumerate(order):
        op, a, b = chosen[v]; last[a] = k; last[b] = k
    reg_of = {}; occ = [None] * nreg
    for k, s0 in enumerate(starts):
        if s0 not in reg_of: reg_of[s0] = k; occ[k] = s0
    for s0 in list(reg_of):
        if s0 not in last: occ[reg_of[s0]] = None
    prog = []
    for k, v in enumerate(order):
        op, a, b = chosen[v]; ra, rb = reg_of[a], reg_of[b]
        if occ[ra] != a or occ[rb] != b: return None
        if last.get(a) == k: occ[ra] = None
        if last.get(b) == k: occ[rb] = None
        try: d = occ.index(None)
        except ValueError: return None
        prog.append([op, ra, rb, d]); reg_of[v] = d; occ[d] = v
        if last.get(v, -1) <= k: occ[d] = None
    return prog


def recombine(sv, targets, limit, nreg, recipes, hint_prog, time_limit):
    import collections
    from ortools.sat.python import cp_model
    sset = set(sv); T = set(targets) - sset
    values = {v for (_, a, b, v) in recipes} | {a for (_, a, b, v) in recipes} | {b for (_, a, b, v) in recipes} | sset
    recipes = set(recipes) | cross_recipes(limit, values)
    rec_by_v = collections.defaultdict(list)
    for r in recipes:
        op, a, b, v = r
        if v in sset or v == a or v == b: continue
        rec_by_v[v].append(r)
    V = list(rec_by_v)
    m = cp_model.CpModel()
    x = {v: m.NewBoolVar('') for v in V}
    lvl = {v: m.NewIntVar(1, len(V) + 1, '') for v in V}
    for t in T:
        if t not in x: return None
        m.Add(x[t] == 1)
    for v in V:
        ys = []
        for (op, a, b, _) in rec_by_v[v]:
            if (a not in sset and a not in x) or (b not in sset and b not in x): continue
            y = m.NewBoolVar(''); ys.append(((op, a, b), y))
            for o in {a, b}:
                if o in sset: continue
                m.AddImplication(y, x[o]); m.Add(lvl[v] >= lvl[o] + 1).OnlyEnforceIf(y)
        if not ys: m.Add(x[v] == 0); rec_by_v[v] = []; continue
        m.Add(sum(y for _, y in ys) == x[v])
        rec_by_v[v] = ys
    m.Minimize(sum(x.values()))
    hr = {}
    for (op, a, b, v) in program_recipes(sv, nreg, hint_prog): hr.setdefault(v, (op, a, b))
    need, st = set(), list(T)
    while st:
        v = st.pop()
        if v in need or v in sset: continue
        need.add(v); op, a, b = hr[v]; st += [a, b]
    depth = {}
    def dep(v):
        if v in sset: return 0
        if v not in depth: op, a, b = hr[v]; depth[v] = 1 + max(dep(a), dep(b))
        return depth[v]
    for v in V:
        m.AddHint(x[v], 1 if v in need else 0)
        if v in need: m.AddHint(lvl[v], dep(v))
        for rec, y in rec_by_v[v]:
            m.AddHint(y, 1 if (v in need and rec == hr.get(v)) else 0)
    sol = cp_model.CpSolver(); sol.parameters.max_time_in_seconds = time_limit; sol.parameters.num_workers = 1
    status = sol.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None
    chosen, level = {}, {}
    for v in V:
        if sol.Value(x[v]):
            for rec, y in rec_by_v[v]:
                if sol.Value(y): chosen[v] = rec; level[v] = sol.Value(lvl[v]); break
    return schedule(sv, nreg, chosen, level)


def recomb_subprocess():
    d = json.load(sys.stdin)
    rec = [tuple(r) for r in d['recipes']]
    try:
        prog = recombine(d['sv'], d['targets'], d['limit'], d['nreg'], rec, d['hint'], float(d['time']))
    except ImportError as ex:
        print(json.dumps({'error': 'import: %s' % ex})); return
    print(json.dumps({'program': prog}))


HAVE_CPSAT = [None]
SELF_PATH = [None]   # script to run with --recomb (this file, else the dataset copy)


def run_recomb(e, exe, p, prog, variants, seed, deadline, t0, newvars):
    tl = min(float(e.get('time', 15)), max(5.0, deadline - time.time() - 60))
    dump = '/tmp/elfdump_%d_%d.txt' % (os.getpid(), seed)
    cmd = [exe, '--time', '%.1f' % tl, '--seed', str(seed), '--kicks', '1000', '--kslack', '2', '--hall', '1',
           '--dump', dump] + list(e.get('args', []))
    best = None
    try:
        r = subprocess.run(cmd, input=enc_elf(p, prog), capture_output=True, text=True, timeout=tl * 2 + 120)
        vv = []
        best = parse_elf(r.stdout, vv).get(p.id)
        newvars.extend(q for k, q in vv if k == p.id)
    except subprocess.TimeoutExpired:
        pass
    if best is not None and (len(best) >= len(prog) or p.check(best)): best = None
    hint = best or prog
    recs = read_dump(dump)
    try: os.remove(dump)
    except OSError: pass
    cp = min(float(e.get('cp', 25)), deadline - time.time() - 30)
    if HAVE_CPSAT[0] is False or cp < 5 or not SELF_PATH[0]:
        return best, time.time() - t0
    for q in [prog] + list(variants[:int(e.get('max_seeds', 4))]) + ([best] if best else []):
        recs |= program_recipes(p.sv, p.nreg, q)
    inp = json.dumps({'sv': p.sv, 'targets': p.targets, 'limit': p.limit, 'nreg': p.nreg, 'hint': hint,
                      'recipes': [list(r) for r in recs], 'time': cp})
    try:
        r = subprocess.run([sys.executable, SELF_PATH[0], '--recomb'], input=inp, capture_output=True,
                           text=True, timeout=cp * 3 + 120)
        out = json.loads(r.stdout.strip().split('\n')[-1]) if r.stdout.strip() else {'error': r.stderr[-300:]}
    except (subprocess.TimeoutExpired, ValueError) as ex:
        out = {'error': repr(ex)[:200]}
    if 'error' in out:
        if HAVE_CPSAT[0] is None and 'import' in out['error']:
            HAVE_CPSAT[0] = False; log('CP-SAT unavailable (%s); recomb runs LNS only' % out['error'])
        return best, time.time() - t0
    HAVE_CPSAT[0] = True
    rp = out.get('program')
    if rp and len(rp) < len(hint) and not p.check(rp):
        best = rp
    return best, time.time() - t0


# ------------------------------------------------------------------ pydriver mode (our research drivers, e.g. lever B)
def run_pydriver(cfg, dirs, outdir, P, start, ids):
    """Rebuild a minimal repo tree in /tmp, run a research driver (grun.py / jrun.py) on the shard with a kernel-local
    store, and export every program that got shorter (re-validated) as improved.csv."""
    import signal
    root = os.path.abspath(cfg.get('repo_root') or '/tmp/elfrepo')
    for base, rel in cfg['tree'].items():
        src = find_file(dirs, base)
        if src is None: raise SystemExit('pydriver file %s not found' % base)
        dst = os.path.join(root, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
    os.makedirs(os.path.join(root, 'data'), exist_ok=True); os.makedirs(os.path.join(root, 'store'), exist_ok=True)
    shutil.copy(find_file(dirs, 'puzzles.csv'), os.path.join(root, 'data', 'puzzles.csv'))
    store = {k: {'len': len(v), 'prog': v, 'tag': 'kg-start'} for k, v in start.items()}
    spath = os.path.join(root, 'store', 'best.json')
    with open(spath, 'w') as f: json.dump(store, f, separators=(',', ':'))
    drv = cfg['driver']; ddir = os.path.join(root, drv['cwd'])
    with open(os.path.join(ddir, 'ids_kshard.txt'), 'w') as f: f.write(','.join(ids))
    for cmd in cfg.get('builds', []):
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if r.returncode != 0: raise SystemExit('build %s failed: %s' % (cmd, (r.stdout + r.stderr)[-1500:]))
    log('built', [c[-1] for c in cfg.get('builds', [])])
    tag = cfg.get('driver_tag', 'k')
    nw = int(cfg.get('workers') or os.cpu_count() or 4)
    args = [a.replace('{tag}', tag).replace('{workers}', str(nw)).replace('{idsfile}', 'ids_kshard.txt') for a in drv['args']]
    name = drv.get('name', drv['script'])
    dlog = open(os.path.join(outdir, 'driver.log'), 'w')
    t0 = time.time(); deadline = T_START + float(cfg['budget_s'])
    pr = subprocess.Popen([sys.executable, '-u', drv['script']] + args, cwd=ddir, stdout=dlog, stderr=subprocess.STDOUT,
                          start_new_session=True, env=dict(os.environ, **drv.get('env', {})))
    log('driver started:', drv['script'], ' '.join(args))
    base = {k: start[k] for k in ids}

    def export(final=False):
        try: cur = json.load(open(spath))
        except (OSError, ValueError): cur = None   # mid-write: try again next time
        rows = []
        if cur is not None:
            for k in ids:
                prog = cur.get(k, {}).get('prog')
                if prog and len(prog) < len(base[k]) and not P[k].check(prog):
                    rows.append((k, len(prog), len(base[k]), name, prog))
            tmp = os.path.join(outdir, '.improved.csv.tmp')
            with open(tmp, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(['id', 'len', 'base_len', 'engine', 'program'])
                for k, n, b, e, prog in rows: w.writerow([k, n, b, e, json.dumps(prog, separators=(',', ':'))])
            os.replace(tmp, os.path.join(outdir, 'improved.csv'))
        runs = []
        rp = os.path.join(ddir, 'runs_%s.jsonl' % tag)
        if os.path.exists(rp):
            for l in open(rp):
                try: runs.append(json.loads(l))
                except ValueError: pass
            shutil.copy(rp, os.path.join(outdir, 'runs_%s.jsonl' % tag))
        ip = os.path.join(ddir, 'improved_%s.jsonl' % tag)
        if os.path.exists(ip): shutil.copy(ip, os.path.join(outdir, 'improved_%s.jsonl' % tag))
        el = time.time() - t0
        gain = sum(r[2] - r[1] for r in rows)
        stats = {'engines': {name: {'runs': len(runs), 'cpu_s': el * nw, 'gain': gain, 'hits': len(rows), 'invalid': 0}},
                 'elapsed_s': time.time() - T_START, 'final': final, 'shard': ids, 'mode': 'pydriver',
                 'spent_s': {r[0]: float(r[4]) for r in runs}, 'finished': [r[0] for r in runs],
                 'gain': gain, 'tag': cfg.get('tag', ''), 'env': cfg.get('_env', {}), 'timeline': []}
        tmp = os.path.join(outdir, '.stats.json.tmp')
        with open(tmp, 'w') as f: json.dump(stats, f)
        os.replace(tmp, os.path.join(outdir, 'stats.json'))
        return stats

    last = 0
    while pr.poll() is None and time.time() < deadline:
        time.sleep(10)
        if time.time() - last > float(cfg.get('write_every_s', 300)):
            st = export(); last = time.time()
            log('progress: %d finished, gain %d' % (len(st['finished']), st['gain']))
    if pr.poll() is None:
        log('budget reached, stopping driver')
        try: os.killpg(pr.pid, signal.SIGTERM)
        except OSError: pass
        try: pr.wait(60)
        except subprocess.TimeoutExpired:
            try: os.killpg(pr.pid, signal.SIGKILL)
            except OSError: pass
            pr.wait()
    dlog.close()
    st = export(final=True)
    log('done: driver exit %s, %d of %d puzzles finished, gain %d on %d puzzles' % (
        pr.returncode, len(st['finished']), len(ids), st['gain'], st['engines'][name]['hits']))


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

    def sig(self, k, prog):
        p = self.P[k]
        return frozenset(r[3] for r in program_recipes(p.sv, p.nreg, prog))

    def add_variant(self, k, prog, front=True):
        """Keep distinct (by value set) valid programs within 3 of the best; at most 8 per puzzle."""
        if len(prog) > len(self.best[k]) + 3 or self.P[k].check(prog): return False
        sg = self.sig(k, prog)
        if sg == self.sig(k, self.best[k]) or any(sg == self.sig(k, v) for v in self.variants[k]): return False
        if front: self.variants[k].insert(0, prog)
        else: self.variants[k].append(prog)
        self.variants[k] = [v for v in self.variants[k] if len(v) <= len(self.best[k]) + 3][:8]
        return True

    def report(self, k, ename, new, cpu, newvars=()):
        with self.lock:
            self.busy.discard(k)
            st = self.stats[ename]; st['runs'] += 1; st['cpu_s'] += cpu; self.spent[k] += cpu
            for v in newvars:
                if len(v) == len(self.best[k]) and self.add_variant(k, v):
                    st['donors'] = st.get('donors', 0) + 1
            if new is None: return
            cur = self.best[k]
            if len(new) >= len(cur):
                return
            err = self.P[k].check(new)
            if err:
                st['invalid'] += 1; log('INVALID from', ename, k, err); return
            st['gain'] += len(cur) - len(new); st['hits'] += 1
            self.best[k] = new; self.who[k] = ename
            self.variants[k] = [v for v in ([cur] + self.variants[k]) if len(v) <= len(new) + 3][:8]
            self.timeline.append([round(time.time() - T_START), k, ename, len(cur), len(new)])
            log('GAIN %s %s %d -> %d (base %d)' % (k, ename, len(cur), len(new), len(self.base[k])))

    def write(self, final=False):
        with self.lock:
            rows = [(k, len(self.best[k]), len(self.base[k]), self.who.get(k, ''), self.best[k])
                    for k in self.ids if len(self.best[k]) < len(self.base[k])]
            stats = {'engines': self.stats, 'elapsed_s': time.time() - T_START, 'final': final,
                     'shard': self.ids, 'spent_s': self.spent, 'timeline': self.timeline,
                     'gain': sum(r[2] - r[1] for r in rows), 'tag': self.cfg.get('tag', ''),
                     'env': self.cfg.get('_env', {})}
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
    bindir = os.path.abspath(cfg.get('bindir') or '/tmp/elfbin')
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
    env = {'python': sys.version.split()[0], 'cpus': os.cpu_count()}
    try:
        env['cpu'] = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
    except Exception:
        pass
    try:
        import ortools; env['ortools'] = ortools.__version__
    except Exception as ex:
        env['ortools'] = 'missing: %r' % (ex,)
    f = globals().get('__file__')
    SELF_PATH[0] = os.path.abspath(f) if f and os.path.exists(f) else find_file(dirs, 'runner.py')
    env['self'] = SELF_PATH[0]
    log('env', env)
    cfg['_env'] = env
    if cfg.get('mode') == 'pydriver':
        cfg['_env'] = env
        return run_pydriver(cfg, dirs, outdir, P, start, ids)
    exes = build_engines(cfg, dirs, bindir)
    cfg['engines'] = [e for e in cfg['engines'] if e['name'] in exes]
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
                if S.add_variant(k, prog, front=False): nv += 1
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
                nv = []
                new, cpu = run_engine(e, exes[e['name']], P[k], prog, variants, seed, deadline, nv)
            except Exception as ex:  # keep the worker alive
                log('engine error', e['name'], k, repr(ex)[:200]); new, cpu, nv = None, 1.0, []
            S.report(k, e['name'], new, cpu, nv)

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
    if '--recomb' in sys.argv:
        recomb_subprocess()
    elif 'CONFIG' in globals():
        main(CONFIG, ['/kaggle/input'], '/kaggle/working')
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('--config', required=True); ap.add_argument('--input', nargs='+', required=True)
        ap.add_argument('--out', required=True)
        a = ap.parse_args()
        main(json.load(open(a.config)), a.input, a.out)

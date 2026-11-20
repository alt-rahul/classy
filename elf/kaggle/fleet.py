"""Kaggle compute fleet: private kernels improve shards of our store; results merge through offer + save_store.

  python3 kaggle/fleet.py dataset [--force]   build kaggle/build/dataset from the store, create/version the private dataset
  python3 kaggle/fleet.py poll                one cycle: harvest finished kernels, re-version dataset if due, launch free slots
  python3 kaggle/fleet.py loop [--every 300]  poll forever (run in the background; resumable, state on disk)
  python3 kaggle/fleet.py status              summary of slots, gains and gain per CPU-hour
  python3 kaggle/fleet.py localtest IDS SECS  run the runner locally (1 worker) on a few puzzles
Config: kaggle/fleet_config.json (read at every cycle; edit it to change engines, budget, shard size, slots).
State:  store/kaggle_state.json.  Log: store/logs/kaggle_compute.md.  Raw kernel outputs: store/kaggle_out/.
Stop:   touch kaggle/STOP (the loop exits after the current cycle; running kernels finish and are harvested
        on the next start).
"""
import csv, json, os, random, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)
import kapi  # noqa: E402
from common import load_puzzles, load_store, save_store, offer, check  # noqa: E402

csv.field_size_limit(10 ** 9)
CFG = os.path.join(HERE, 'fleet_config.json')
STATE = os.path.join(ROOT, 'store', 'kaggle_state.json')
LOG = os.path.join(ROOT, 'store', 'logs', 'kaggle_compute.md')
OUT = os.path.join(ROOT, 'store', 'kaggle_out')
DSDIR = os.path.join(HERE, 'build', 'dataset')
ENGDIR = os.path.join(HERE, 'engines')
FINISHED = ('complete', 'error', 'cancelAcknowledged', 'cancelRequested', 'cancelled')


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())


def mlog(msg):
    line = '- %s UTC %s' % (now(), msg)
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def load_cfg():
    return json.load(open(CFG))


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {'dataset': {}, 'slots': {}, 'coverage': {}, 'launches': {}, 'gains': {}, 'history': [],
            'engine_totals': {}, 'total_gain': 0, 'total_cpu_s': 0.0}


def save_state(s):
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, STATE)


def store_total(st):
    return sum(v['len'] for v in st.values())


# ------------------------------------------------------------------ dataset
def engine_files():
    out = [os.path.join(ENGDIR, 'elfopt.cpp')]
    out += sorted(glob_files(os.path.join(ENGDIR, 'wj')))
    extra = os.path.join(ENGDIR, 'extra')
    if os.path.isdir(extra):
        out += sorted(glob_files(extra))
    return out


def glob_files(d):
    return [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and not f.startswith('.')]


def build_dataset_dir(st):
    if os.path.isdir(DSDIR):
        shutil.rmtree(DSDIR)
    os.makedirs(DSDIR)
    P = load_puzzles()
    with open(os.path.join(DSDIR, 'programs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for p in P:
            w.writerow([p.id, json.dumps(st[p.id]['prog'], separators=(',', ':'))])
    names = set()
    for p in engine_files():
        b = os.path.basename(p)
        if b in names:
            raise SystemExit('duplicate engine file name ' + b)
        names.add(b)
        dst = 'NOTICE_wj.md' if b == 'NOTICE.md' else b
        shutil.copy(p, os.path.join(DSDIR, dst))
    shutil.copy(os.path.join(HERE, 'runner.py'), os.path.join(DSDIR, 'runner.py'))
    vp = os.path.join(HERE, 'build', 'variants.csv')
    if os.path.exists(vp):
        shutil.copy(vp, os.path.join(DSDIR, 'variants.csv'))
    with open(os.path.join(DSDIR, 'NOTICE.md'), 'w') as f:
        f.write(open(os.path.join(ENGDIR, 'NOTICE_engines.md')).read())
    return sorted(os.path.join(DSDIR, x) for x in os.listdir(DSDIR))


def cmd_dataset(force=False, state=None, quiet_if_same=True):
    cfg = load_cfg()
    s = state if state is not None else load_state()
    st = load_store()
    tot = store_total(st)
    ds = s.setdefault('dataset', {})
    efp = engine_fingerprint()
    if not force and ds.get('total') == tot and ds.get('engines') == efp:
        if not quiet_if_same:
            print('dataset up to date (total %d)' % tot)
        return False
    paths = build_dataset_dir(st)
    slug = cfg['dataset']
    status = kapi.dataset_status(slug)
    if status is None:
        r = kapi.dataset_create(slug, cfg.get('dataset_title', 'elf compute'), paths)
        what = 'created'
    else:
        r = kapi.dataset_version(slug, paths, 'store %d' % tot)
        what = 'versioned'
    err = r.get('error') if isinstance(r, dict) else None
    if err:
        mlog('dataset %s FAILED: %s' % (what, err))
        return False
    ds.update({'slug': slug, 'total': tot, 'engines': efp, 'at': time.time(), 'ref': r.get('ref') or r.get('url')})
    mlog('dataset %s %s/%s with store total %d (%d files)' % (what, kapi.OWNER, slug, tot, len(paths)))
    if state is None:
        save_state(s)
    return True


def engine_fingerprint():
    import hashlib
    h = hashlib.sha1()
    for p in engine_files() + [os.path.join(HERE, 'runner.py')]:
        h.update(os.path.basename(p).encode()); h.update(open(p, 'rb').read())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ kernels
def kernel_text(cfg, ids, st, tag, seed):
    kc = {'ids': ids, 'budget_s': cfg['budget_s'], 'workers': cfg.get('workers', 4), 'engines': cfg['engines'],
          'seed': seed, 'tag': tag, 'write_every_s': cfg.get('write_every_s', 300),
          'random_engine': cfg.get('random_engine', True),
          'programs': {k: st[k]['prog'] for k in ids}}
    src = open(os.path.join(HERE, 'runner.py')).read()
    return 'CONFIG = __import__("json").loads(%r)\n' % json.dumps(kc, separators=(',', ':')) + src


def choose_shard(cfg, s, st, n, exclude):
    P = load_puzzles()
    minlen, maxlen = cfg.get('min_len', 0), cfg.get('max_len', 10 ** 9)
    only = set(cfg.get('only_ids') or [])
    cand = [p.id for p in P if minlen <= st[p.id]['len'] <= maxlen and p.id not in exclude and (not only or p.id in only)]
    cov = s.get('coverage', {})
    gains = s.get('gains', {})
    # adaptive band weights: merged gains per core-hour by program-length band, with a prior
    edges = cfg.get('band_edges', [60, 100, 140, 180, 220])
    band = lambda L: sum(1 for e in edges if L >= e)
    G, H = {}, {}
    for p in P:
        b = band(st[p.id]['len'])
        G[b] = G.get(b, 0) + gains.get(p.id, 0)
        H[b] = H.get(b, 0.0) + cov.get(p.id, 0.0) / 3600
    a0, h0 = float(cfg.get('band_prior_gain', 1)), float(cfg.get('band_prior_hours', 20))
    w = {b: (G.get(b, 0) + a0) / (H.get(b, 0.0) + h0) for b in range(len(edges) + 1)}
    c0 = float(cfg.get('pseudo_coverage_s', 0))

    # least covered first, relative to priority (length^p, band gain rate, past gains of the puzzle)
    def key(k):
        L = st[k]['len']
        pr = L ** float(cfg.get('len_power', 1.0)) * (1.0 + float(cfg.get('gain_bonus', 0.5)) * gains.get(k, 0))
        if cfg.get('adaptive', False):
            pr *= w[band(L)]
        return ((cov.get(k, 0.0) + c0) / pr, -L, k)
    cand.sort(key=key)
    return cand[:n]


def launch(cfg, s, slot, st):
    busy = set()
    for sl in s['slots'].values():
        if sl.get('state') in ('running', 'pushed', 'queued'):
            busy.update(sl.get('ids', []))
    ids = choose_shard(cfg, s, st, int(cfg['puzzles_per_kernel']), busy)
    if not ids:
        return False
    seed = random.randrange(1, 2 ** 31)
    tag = '%s_%d' % (slot, int(time.time()))
    text = kernel_text(cfg, ids, st, tag, seed)
    title = slot.replace('-', ' ')
    r = kapi.kernel_push(slot, title, text, datasets=['%s/%s' % (kapi.OWNER, cfg['dataset'])])
    err = r.get('error') if isinstance(r, dict) else None
    if err:
        mlog('push %s FAILED: %s' % (slot, err))
        s['slots'][slot] = {'state': 'push_failed', 'error': err, 'at': time.time()}
        return False
    ver = r.get('versionNumber')
    s['slots'][slot] = {'state': 'pushed', 'ids': ids, 'tag': tag, 'version': ver, 'pushed_at': time.time(),
                        'budget_s': cfg['budget_s'], 'base_total': sum(st[k]['len'] for k in ids),
                        'engines': [e['name'] for e in cfg['engines']], 'url': r.get('url')}
    for k in ids:
        s['launches'][k] = s['launches'].get(k, 0) + 1
        # provisional coverage so the next shard picks other puzzles; corrected at harvest
        s['coverage'][k] = s['coverage'].get(k, 0.0) + cfg['budget_s'] * cfg.get('workers', 4) / len(ids)
    mlog('launched %s v%s: %d puzzles (len %d..%d, %d instr), budget %.1f h, engines %s' % (
        slot, ver, len(ids), min(st[k]['len'] for k in ids), max(st[k]['len'] for k in ids),
        sum(st[k]['len'] for k in ids), cfg['budget_s'] / 3600, ','.join(e['name'] for e in cfg['engines'])))
    return True


def harvest(cfg, s, slot, sl, status):
    P = {p.id: p for p in load_puzzles()}
    dest = os.path.join(OUT, '%s_v%s' % (slot, sl.get('version')))
    try:
        names, _ = kapi.kernel_output(slot, dest)
    except kapi.KaggleError as e:
        mlog('output download %s failed: %s' % (slot, e))
        return False
    imp = os.path.join(dest, 'improved.csv')
    stats = {}
    if os.path.exists(os.path.join(dest, 'stats.json')):
        try:
            stats = json.load(open(os.path.join(dest, 'stats.json')))
        except ValueError:
            stats = {}
    gained, n_imp, n_bad, details = 0, 0, 0, []
    if os.path.exists(imp):
        st = load_store()
        for row in csv.DictReader(open(imp, newline='', encoding='utf-8')):
            k = row['id']
            if k not in P:
                continue
            try:
                prog = json.loads(row['program'])
            except ValueError:
                n_bad += 1; continue
            if check(P[k], prog) is not None:
                n_bad += 1; continue
            old = st[k]['len']
            if offer(st, P[k], prog, 'kg:%s' % (row.get('engine') or 'x')):
                gained += old - len(prog); n_imp += 1
                s['gains'][k] = s['gains'].get(k, 0) + old - len(prog)
                details.append('%s %d->%d %s' % (k, old, len(prog), row.get('engine')))
        if n_imp:
            save_store(st)
    # bookkeeping: real core-seconds per puzzle and per engine
    spent = stats.get('spent_s', {})
    ids = sl.get('ids', [])
    prov = sl.get('budget_s', 0) * cfg.get('workers', 4) / max(1, len(ids))
    for k in ids:
        s['coverage'][k] = max(0.0, s['coverage'].get(k, 0.0) - prov) + float(spent.get(k, 0.0))
    cpu = sum(float(v) for v in spent.values())
    for name, e in (stats.get('engines') or {}).items():
        t = s['engine_totals'].setdefault(name, {'runs': 0, 'cpu_s': 0.0, 'gain_kernel': 0, 'hits': 0})
        t['runs'] += e.get('runs', 0); t['cpu_s'] += e.get('cpu_s', 0.0)
        t['gain_kernel'] += e.get('gain', 0); t['hits'] += e.get('hits', 0)
    s['total_gain'] += gained
    s['total_cpu_s'] += cpu
    tot = store_total(load_store())
    s['history'].append({'slot': slot, 'version': sl.get('version'), 'status': status, 'gain_store': gained,
                         'gain_kernel': stats.get('gain', 0), 'cpu_s': cpu, 'n': len(ids), 'finished_at': time.time(),
                         'store_total': tot, 'files': names})
    mlog('harvested %s v%s (%s): kernel gain %s, merged gain %d on %d puzzles (%d rejected), %.1f CPU-h, store now %d%s' % (
        slot, sl.get('version'), status, stats.get('gain', '?'), gained, n_imp, n_bad, cpu / 3600, tot,
        ('; ' + ', '.join(details[:12])) if details else ''))
    if not stats:
        mlog('  %s produced no stats.json (files: %s); see %s' % (slot, names, dest))
    return True


def cmd_poll():
    cfg = load_cfg()
    s = load_state()
    st = load_store()
    # 1. status of every active slot
    for slot, sl in sorted(s['slots'].items()):
        if sl.get('state') not in ('pushed', 'queued', 'running'):
            continue
        try:
            r = kapi.kernel_status(slot)
        except kapi.KaggleError as e:
            mlog('status %s failed: %s' % (slot, e)); continue
        status = r.get('status', '?')
        if status in FINISHED:
            if status != 'complete':
                mlog('%s v%s ended with status %s: %s' % (slot, sl.get('version'), status, r.get('failureMessage', '')))
            if harvest(cfg, s, slot, sl, status):
                sl['state'] = 'done'; sl['status'] = status; sl['done_at'] = time.time()
        else:
            if sl.get('state') != status:
                sl['state'] = status if status in ('queued', 'running') else sl['state']
                if status == 'running':
                    sl['running_at'] = time.time()
            # safety: a kernel far past its budget is reported (Kaggle stops it at 12 h)
            if time.time() - sl.get('pushed_at', time.time()) > sl.get('budget_s', 0) + 4 * 3600:
                sl.setdefault('overdue_logged', False)
                if not sl['overdue_logged']:
                    mlog('%s is %.1f h past push and still %s' % (slot, (time.time() - sl['pushed_at']) / 3600, status))
                    sl['overdue_logged'] = True
    save_state(s)
    st = load_store()
    # 2. dataset re-version when the store improved (kernels embed fresh programs, so this is hourly at most)
    ds = s.get('dataset', {})
    if ds.get('total') is None or (store_total(st) < ds['total'] and time.time() - ds.get('at', 0) > cfg.get('dataset_every_s', 3600)) \
            or ds.get('engines') != engine_fingerprint():
        try:
            cmd_dataset(state=s)
            save_state(s)
        except kapi.KaggleError as e:
            mlog('dataset update failed: %s' % e)
    # 3. launch free slots
    if os.path.exists(os.path.join(HERE, 'STOP_LAUNCH')):
        return s
    nslots = int(cfg['max_kernels'])
    for i in range(1, nslots + 1):
        slot = '%s%02d' % (cfg['slot_prefix'], i)
        sl = s['slots'].get(slot, {})
        if sl.get('state') in ('pushed', 'queued', 'running'):
            continue
        if sl.get('state') == 'push_failed':
            # session-count refusals clear as soon as Kaggle releases a finished session: retry next poll
            wait = 200 if 'Maximum batch' in str(sl.get('error', '')) else 1800
            if time.time() - sl.get('at', 0) < wait:
                continue
        try:
            launch(cfg, s, slot, st)
        except kapi.KaggleError as e:
            mlog('push %s error: %s' % (slot, e))
            s['slots'][slot] = {'state': 'push_failed', 'error': str(e)[:300], 'at': time.time()}
        save_state(s)
    return s


def cmd_status():
    s = load_state()
    st = load_store()
    print('store total', store_total(st))
    for slot, sl in sorted(s['slots'].items()):
        age = (time.time() - sl.get('pushed_at', time.time())) / 3600
        print('%-10s %-12s v%-4s %3d puzzles  pushed %.1f h ago  %s' % (slot, sl.get('state'), sl.get('version'),
                                                                     len(sl.get('ids', [])), age, sl.get('error', '')))
    cpu = s.get('total_cpu_s', 0) / 3600
    print('merged gain %d over %.1f CPU-h (%.2f per CPU-h)' % (s.get('total_gain', 0), cpu, s.get('total_gain', 0) / cpu if cpu else 0))
    for name, t in sorted(s.get('engine_totals', {}).items()):
        h = t['cpu_s'] / 3600
        print('  %-22s runs %6d  %7.1f CPU-h  kernel gain %4d  (%.2f per CPU-h)' % (name, t['runs'], h, t['gain_kernel'],
                                                                                    t['gain_kernel'] / h if h else 0))


def code_mtime():
    return max(os.path.getmtime(os.path.join(HERE, f)) for f in ('fleet.py', 'kapi.py'))


def cmd_loop(every):
    mlog('fleet loop started (pid %d)' % os.getpid())
    t_code = code_mtime()
    while True:
        if code_mtime() != t_code:   # fleet.py / kapi.py edited: restart with the new code (state is on disk)
            mlog('fleet code changed, re-exec')
            os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), 'loop', '--every', str(every)])
        try:
            cmd_poll()
        except Exception as e:  # keep the loop alive; everything is on disk
            mlog('poll error: %r' % (e,))
        if os.path.exists(os.path.join(HERE, 'STOP')):
            mlog('STOP file found, fleet loop exits'); return
        time.sleep(every)


def cmd_localtest(ids, secs):
    cfg = load_cfg()
    st = load_store()
    build_dataset_dir(st)
    ids = ids.split(',')
    kc = {'ids': ids, 'budget_s': float(secs), 'workers': 1, 'engines': cfg['engines'], 'seed': 1, 'tag': 'local',
          'write_every_s': 60, 'bindir': os.path.join(HERE, 'build', 'localbin'), 'build_jobs': 1, 'random_engine': cfg.get('random_engine', True)}
    tdir = os.path.join(HERE, 'build', 'localtest')
    os.makedirs(tdir, exist_ok=True)
    json.dump(kc, open(os.path.join(tdir, 'cfg.json'), 'w'))
    subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HERE, 'runner.py'), '--config', os.path.join(tdir, 'cfg.json'),
                    '--input', DSDIR, os.path.join(ROOT, 'data'), '--out', tdir], check=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(0)
    if a[0] == 'dataset':
        cmd_dataset(force='--force' in a, quiet_if_same=False)
    elif a[0] == 'poll':
        cmd_poll()
    elif a[0] == 'loop':
        cmd_loop(int(a[a.index('--every') + 1]) if '--every' in a else 300)
    elif a[0] == 'status':
        cmd_status()
    elif a[0] == 'localtest':
        cmd_localtest(a[1], a[2])
    else:
        print(__doc__)

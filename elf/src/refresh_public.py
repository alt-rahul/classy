"""Re-download openly licensed public Santa 2026 program datasets via the Kaggle API and merge them into the store.

Only datasets whose Kaggle license is in ALLOWED are used. Files are parsed as CSV only; nothing downloaded is executed.
Every program is re-validated by common.offer before it enters the store (tag pub:<owner>/<file>).
Usage: python3 src/refresh_public.py [--dry] [--max N]
"""
import csv, io, json, os, subprocess, sys, tempfile, zipfile
sys.path.insert(0, os.path.dirname(__file__))
from common import load_puzzles, load_store, save_store, offer

csv.field_size_limit(10**9)
API = 'https://www.kaggle.com/api/v1'
ALLOWED = {'Apache 2.0', 'CC0: Public Domain', 'CC BY-SA 4.0', 'CC BY 4.0', 'MIT', 'Attribution 4.0 International (CC BY 4.0)'}


def get(url, out=None):
    cmd = ['curl', '-s', '-L', '--max-time', '300', url]
    if out:
        cmd += ['-o', out]
        subprocess.run(cmd, check=False)
        return None
    return subprocess.run(cmd, capture_output=True, check=False).stdout


def list_datasets(pages=3):
    seen = {}
    for page in range(1, pages + 1):
        raw = get(f'{API}/datasets/list?search=santa-2026&sortBy=updated&page={page}')
        try:
            d = json.loads(raw)
        except Exception:
            break
        if not d:
            break
        for x in d:
            seen[x['ref']] = x
    return list(seen.values())


def programs_from_zip(path):
    out = {}
    try:
        z = zipfile.ZipFile(path)
    except Exception:
        return out
    for name in z.namelist():
        if not name.endswith('.csv'):
            continue
        with z.open(name) as f:
            head = f.read(200).decode('utf-8', 'replace').splitlines()[0] if True else ''
        cols = [c.strip() for c in head.split(',')]
        if 'id' not in cols or 'program' not in cols:
            continue
        with z.open(name) as f:
            rows = csv.DictReader(io.TextIOWrapper(f, 'utf-8', errors='replace'))
            progs = {}
            for r in rows:
                try:
                    progs[r['id']] = json.loads(r['program'])
                except Exception:
                    pass
        if len(progs) >= 50:
            out[name] = progs
    return out


def main(argv):
    dry = '--dry' in argv
    mx = int(argv[argv.index('--max') + 1]) if '--max' in argv else 40
    puzzles = load_puzzles('scored')
    st = load_store()
    before = sum(v['len'] for v in st.values())
    ds = [x for x in list_datasets() if x.get('licenseName') in ALLOWED and x.get('totalBytes', 0) < 60e6
          and any(k in (x.get('title', '') + x['ref']).lower() for k in ('santa', 'elf'))]
    print(f'store before {before}; {len(ds)} openly licensed candidate datasets')
    tmp = tempfile.mkdtemp(prefix='elfpub_')
    for x in ds[:mx]:
        ref = x['ref']
        zp = os.path.join(tmp, ref.replace('/', '_') + '.zip')
        get(f'{API}/datasets/download/{ref}', zp)
        files = programs_from_zip(zp)
        os.remove(zp) if os.path.exists(zp) else None
        for name, progs in files.items():
            wins = gain = 0
            for p in puzzles:
                prog = progs.get(p.id)
                if not isinstance(prog, list):
                    continue
                cur = st[p.id]['len']
                if len(prog) < cur and offer(st, p, prog, f'pub:{ref.split("/")[0]}/{os.path.basename(name)}'):
                    wins += 1
                    gain += cur - len(prog)
            if wins:
                print(f'  {ref} [{x["licenseName"]}] {name}: improves {wins} puzzles by {gain}')
    after = sum(v['len'] for v in st.values())
    print(f'store after {after} (gain {before - after})')
    if not dry and after < before:
        save_store(st)
        print('saved')


if __name__ == '__main__':
    main(sys.argv[1:])

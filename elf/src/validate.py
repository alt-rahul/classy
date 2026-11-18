import csv, json, sys
csv.field_size_limit(10**9)
def run(sv, targets, limit, prog, nreg=48):
    if not isinstance(prog, list): return 'not list'
    if len(prog) > 4096: return 'too long'
    reg = [None]*nreg
    for i, v in enumerate(sv): reg[i] = v
    need = set(targets) - set(sv)
    for k, ins in enumerate(prog):
        if not (isinstance(ins, list) and len(ins) == 4): return f'bad ins {k}'
        op, i, j, d = ins
        if op not in '+-*/' or len(op) != 1: return f'bad op {k}'
        for x in (i, j, d):
            if type(x) is not int or not 0 <= x < nreg: return f'bad reg {k}'
        a, b = reg[i], reg[j]
        if a is None or b is None: return f'undef {k}'
        if op == '+': r = a + b
        elif op == '-':
            if a <= b: return f'neg {k}'
            r = a - b
        elif op == '*': r = a * b
        else:
            if a % b: return f'inexact {k}'
            r = a // b
        if r > limit or r < 1: return f'limit {k}'
        reg[d] = r
        need.discard(r)
    return 'missing %d' % len(need) if need else None
def main(puz, sub):
    P = {r['id']: r for r in csv.DictReader(open(puz))}
    rows = list(csv.reader(open(sub, encoding='utf-8')))
    assert rows[0] == ['id', 'program'], rows[0]
    ids = [r[0] for r in rows[1:]]
    assert len(ids) == len(set(ids)) == len(P) and set(ids) == set(P), 'id mismatch'
    tot = 0; bad = 0
    for pid, cell in rows[1:]:
        if len(cell.encode()) > 131072: print(pid, 'cell too big'); bad += 1; continue
        prog = json.loads(cell); p = P[pid]
        err = run(json.loads(p['starting_values']), json.loads(p['targets']), int(p['value_limit']), prog, int(p['registers']))
        if err: print(pid, err); bad += 1
        tot += len(prog)
    print('invalid', bad, 'score', tot)
main(sys.argv[1], sys.argv[2])

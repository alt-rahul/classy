"""Where do the ops go? Marginal cost per target (ops between consecutive first coverages) from the store."""
import sys, collections
sys.path.insert(0, '/home/user/classy/elf/src')
from common import *
st = load_store(); P = load_puzzles('scored')
def marg(p, prog):
    reg = {i: v for i, v in enumerate(p.sv)}; ts = set(p.targets) - set(p.sv); seen = set(); last = 0; out = []
    for k, (op, i, j, d) in enumerate(prog):
        a, b = reg[i], reg[j]
        v = a + b if op == '+' else a - b if op == '-' else a * b if op == '*' else a // b
        reg[d] = v
        if v in ts and v not in seen:
            seen.add(v); out.append((v, k + 1 - last, len(seen))); last = k + 1
    return out
for cls in ('other>=20', 'indep>=20'):
    H = collections.Counter(); B = collections.defaultdict(lambda: [0, 0]); R = collections.defaultdict(lambda: [0, 0]); tot = 0
    for p in P:
        if pclass(p) != cls: continue
        for v, c, r in marg(p, st[p.id]['prog']):
            H[min(c, 15)] += 1; tot += c
            b = B[v.bit_length() // 5 * 5]; b[0] += 1; b[1] += c
            q = R[min((r - 1) // 3, 10)]; q[0] += 1; q[1] += c
    print(cls, 'total', tot)
    print(' marginal-cost hist (cost: n targets, share of ops):', ' '.join('%d:%d,%.0f%%' % (c, n, 100 * c * n / tot) for c, n in sorted(H.items())))
    print(' by bits (bits: n, avg cost, share):', ' '.join('%d:%d,%.1f,%.0f%%' % (b, n, s / n, 100 * s / tot) for b, (n, s) in sorted(B.items())))
    print(' by build rank/3 (n, avg):', ' '.join('%d:%.1f' % (r, s / n) for r, (n, s) in sorted(R.items())))

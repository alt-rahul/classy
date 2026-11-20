"""engine wrapper with round-3 additions; sets ELF_CORE to core3.so before importing engine."""
import os, sys, ctypes as C
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('ELF_CORE', os.path.join(HERE, 'core3.so'))
sys.path.insert(0, '/home/user/classy/elf/src')
import engine
from engine import _lib, u64p
_lib.level_ptr.restype = u64p
_lib.level_ptr.argtypes = [C.c_int]
_lib.all_costs.argtypes = [u64p, C.c_int, C.c_int, C.POINTER(C.c_int)]
_lib.hub_scan.restype = C.c_int64
_lib.hub_scan.argtypes = [u64p, C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, u64p, C.c_int64]
_lib.resid_lev.restype = C.c_int
_lib.resid_lev.argtypes = [C.c_uint64, C.c_uint64, C.c_int]
_HUB = (C.c_uint64 * 400000)()

def all_costs(ts, maxc, default=None):
    n = len(ts); arr = (C.c_uint64 * n)(*ts); out = (C.c_int * n)()
    _lib.all_costs(arr, n, maxc, out)
    return [out[i] if out[i] >= 0 else default for i in range(n)]

def hub_scan(ts, base, lu, lw, ming=1, minserve=2):
    n = len(ts); arr = (C.c_uint64 * n)(*ts); b = (C.c_int * n)(*base)
    m = _lib.hub_scan(arr, b, n, lu, lw, ming, minserve, _HUB, 400000)
    res = []
    for i in range(m):
        g = _HUB[4 * i + 1]
        if g >= 1 << 63: g -= 1 << 64
        res.append((_HUB[4 * i], g, _HUB[4 * i + 2], _HUB[4 * i + 3]))
    return res

def resid_lev(t, u, lw):
    return _lib.resid_lev(t, u, lw)

_lib.enum_pairs.restype = C.c_int64
_lib.enum_pairs.argtypes = [C.c_uint64, C.c_int, u64p, C.c_int64]
_EN = (C.c_uint64 * (5 * 200000))()

def enum_pairs(t, c):
    """All pair plans of exact cost c: list of (op, x, y, lx, ly) with t = x op y."""
    m = _lib.enum_pairs(t, c, _EN, 200000)
    m = min(m, 200000)
    return [(engine.OPS[_EN[5 * i]], _EN[5 * i + 1], _EN[5 * i + 2], _EN[5 * i + 3], _EN[5 * i + 4]) for i in range(m)]

def plan_steps(t, rec):
    op, x, y, lx, ly = rec
    return engine.expand(x) + engine.expand(y) + [(op, x, y, t)]

if hasattr(_lib, 'peel2'):
    _lib.peel2.restype = C.c_int
    _lib.peel2.argtypes = [C.c_uint64, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, u64p]
_PB = (C.c_uint64 * 4)()

def _op_of(a, b, r):
    """step producing r from a and b (either order), or None."""
    if a + b == r: return ('+', a, b, r)
    if a - b == r: return ('-', a, b, r)
    if b - a == r: return ('-', b, a, r)
    if a * b == r: return ('*', a, b, r)
    if b and a % b == 0 and a // b == r: return ('/', a, b, r)
    if a and b % a == 0 and b // a == r: return ('/', b, a, r)
    return None

TIERS = ((0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0))

def peel(t, maxcost, tiers=TIERS, ly=3):
    """Cheapest chain plan t = (y op v1) op v2 with cost <= maxcost. Returns (cost, steps) or None."""
    best = None
    for (la, lb, full) in tiers:
        mc = maxcost if best is None else best[0] - 1
        if mc < la + lb + 2:
            continue
        c = _lib.peel2(t, la, lb, ly, full, mc, _PB)
        if c >= 0:
            best = (c, _PB[0], _PB[1], _PB[2], _PB[3])
    if best is None:
        return None
    c, y, v1, r1, v2 = best
    s1 = _op_of(y, v1, r1); s2 = _op_of(r1, v2, t)
    assert s1 and s2, (t, best)
    return c, engine.expand(y) + engine.expand(v1) + [s1] + engine.expand(v2) + [s2]

if hasattr(_lib, 'peel3_full'):
    _lib.peel3_full.restype = C.c_int
    _lib.peel3_full.argtypes = [C.c_uint64, C.c_int, C.c_int, u64p]
    _lib.peel3_amc.restype = C.c_int
    _lib.peel3_amc.argtypes = [C.c_uint64, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, u64p]
_PB6 = (C.c_uint64 * 6)()
AMC = ((0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 1), (1, 1, 0))

def peel3(t, maxcost, amc=AMC, ly=3, full=True):
    """Cheapest three-leaf chain t = ((y op v1) op v2) op v3 with cost <= maxcost: (cost, steps) or None."""
    best = None
    if full and maxcost >= 3:
        c = _lib.peel3_full(t, ly, maxcost, _PB6)
        if c >= 0: best = (c,) + tuple(_PB6)
    for (la, lm, lc) in amc:
        mc = maxcost if best is None else best[0] - 1
        if mc < 3 + la + lm + lc: continue
        c = _lib.peel3_amc(t, la, lm, lc, ly, mc, _PB6)
        if c >= 0: best = (c,) + tuple(_PB6)
    if best is None:
        return None
    c, y, v1, r1, v2, r2, v3 = best
    s1 = _op_of(y, v1, r1); s2 = _op_of(r1, v2, r2); s3 = _op_of(r2, v3, t)
    assert s1 and s2 and s3, (t, best)
    return c, engine.expand(y) + engine.expand(v1) + [s1] + engine.expand(v2) + [s2] + engine.expand(v3) + [s3]

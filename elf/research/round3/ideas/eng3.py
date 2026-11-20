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

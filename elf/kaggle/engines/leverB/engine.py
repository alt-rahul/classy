"""ctypes wrapper around core.so (level sets, pair search, window search, Horner candidates)."""
import ctypes, os
HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.environ.get('ELF_CORE') or os.path.join(HERE, 'core.so'))
u64p = ctypes.POINTER(ctypes.c_uint64)
i64p = ctypes.POINTER(ctypes.c_int64)
C = ctypes
_lib.build_levels.restype = C.c_int64
_lib.build_levels.argtypes = [u64p, C.c_int, C.c_uint64, C.c_int, C.c_int, C.c_int64, i64p]
MAXLEV = 8
_lib.find_pair.restype = C.c_int
_lib.find_pair.argtypes = [C.c_uint64, C.c_int, u64p, C.c_int]
_lib.find_window.restype = C.c_int
_lib.find_window.argtypes = [C.c_uint64, C.c_int, C.c_double, u64p, C.c_int]
_lib.level_of.restype = C.c_int
_lib.level_of.argtypes = [C.c_uint64, C.c_int]
_lib.expand_value.restype = C.c_int
_lib.expand_value.argtypes = [C.c_uint64, C.c_int, u64p, C.c_int]
_lib.level_size.restype = C.c_int64
_lib.level_size.argtypes = [C.c_int]
_lib.get_outn.restype = C.c_int
_lib.horner_candidates.restype = C.c_int
_lib.horner_candidates.argtypes = [C.c_uint64, C.c_int, C.c_int, u64p, C.c_int]
_lib.set_seed.argtypes = [C.c_uint64]
_lib.derive_failed.restype = C.c_int
_lib.set_wemin.argtypes = [C.c_double]
_lib.window_cands.restype = C.c_int64
_lib.level_hist.argtypes = [C.c_int, i64p]
if hasattr(_lib, 'set_pair_eps'):
    _lib.set_pair_eps.argtypes = [C.c_double]
if hasattr(_lib, 'regret_pick'):
    _lib.regret_pick.restype = C.c_int
    _lib.regret_pick.argtypes = [u64p, C.c_int, C.c_int, C.c_double, u64p, C.c_int, C.POINTER(C.c_int)]
if hasattr(_lib, 'set_peel'):
    _lib.set_peel.argtypes = [C.c_int]
    _lib.peel_stats.restype = C.c_int64
    _lib.peel_stats.argtypes = [C.c_int]
    _lib.last_pick_chain.restype = C.c_int
_RC = C.c_int(0)

OPS = '+-*/'
_BUF = (C.c_uint64 * 8192)()
_SZ = (C.c_int64 * 8)()
_HB = (C.c_uint64 * (4 * 50000))()


def build(pool, limit, maxlev=3, keep=0, cap=20_000_000):
    """Build levels 0..maxlev. keep=1: rebuild L0..L2 only and keep existing deep levels."""
    n = len(pool)
    arr = (C.c_uint64 * n)(*pool)
    _lib.build_levels(arr, n, limit, maxlev, keep, int(cap), _SZ)
    return [_SZ[k] for k in range(MAXLEV) if _SZ[k]]


def _steps(m):
    b = _BUF
    return [(OPS[b[i]], b[i + 1], b[i + 2], b[i + 3]) for i in range(0, m, 4)]


def find_pair(t, maxcost):
    """Cheapest t = u op w (or direct) with cost <= maxcost. Returns (cost, steps) or None."""
    c = _lib.find_pair(t, maxcost, _BUF, 8192)
    if c < 0:
        return None
    return c, _steps(_lib.get_outn())


def find_window(t, bestcost, budget):
    """t = X*Y +- Z with cost < bestcost. Returns (cost, steps) or None."""
    c = _lib.find_window(t, bestcost, budget, _BUF, 8192)
    if c < 0:
        return None
    return c, _steps(_lib.get_outn())


def window_cands():
    return _lib.window_cands()


def horner_candidates(t, ax=1, cz=2):
    """List of (X, Z, cx, cz, sign): t = X*Y + Z (sign +1) or t = X*Y - Z (sign -1, Y = ceil)."""
    q = _lib.horner_candidates(t, ax, cz, _HB, 50000)
    b = _HB
    res = []
    for i in range(q):
        cx = b[4 * i + 2]
        res.append((b[4 * i], b[4 * i + 1], cx & 255, b[4 * i + 3], -1 if cx & 256 else 1))
    return res


def expand(v, maxlev=7):
    m = _lib.expand_value(v, maxlev, _BUF, 8192)
    if m < 0:
        raise KeyError(v)
    return _steps(m)


def level_of(v, maxlev=7):
    return _lib.level_of(v, maxlev)


def set_seed(s):
    _lib.set_seed(s)


def hist(k):
    h = (C.c_int64 * 65)()
    _lib.level_hist(k, h)
    return list(h)


def set_wemin(e):
    _lib.set_wemin(e)


def derive_failed():
    return _lib.derive_failed()


def set_pair_eps(e):
    if hasattr(_lib, 'set_pair_eps'):
        _lib.set_pair_eps(C.c_double(e))


def regret_pick(ts, maxc, alpha):
    """Regret ordering pick over targets ts (fresh levels required). Returns (cost, t, steps) or None."""
    n = len(ts)
    arr = (C.c_uint64 * n)(*ts)
    i = _lib.regret_pick(arr, n, maxc, alpha, _BUF, 8192, C.byref(_RC))
    if i < 0:
        return None
    return _RC.value, ts[i], _steps(_lib.get_outn())


def set_peel(ntiers):
    """Chain ("peel") plans inside regret_pick: number of tiers used (0 = off, 4 = all)."""
    if hasattr(_lib, 'set_peel'):
        _lib.set_peel(int(ntiers))
    elif ntiers:
        raise RuntimeError('core library has no chain search')


def last_pick_chain():
    return _lib.last_pick_chain() if hasattr(_lib, 'last_pick_chain') else 0


def peel_stats():
    return (_lib.peel_stats(0), _lib.peel_stats(1)) if hasattr(_lib, 'peel_stats') else (0, 0)

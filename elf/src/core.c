// Core search engine for the elf calculator puzzles.
// Level sets L0..L3 of values reachable from a pool with exact (tree) cost k.
// L0..L2 keep witnesses; L3 keeps values only, witnesses are re-derived on demand.
// Searches: pair (meet in the middle), window (t = X*Y +- Z), Horner candidates.
// Build: gcc -O3 -march=native -shared -fPIC -o core.so core.c
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>

typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned __int128 u128;

#define MAXLEV 8
static u64 LIM = 0;
static u64 *LV[MAXLEV];   // sorted values
static u64 *LW[MAXLEV];   // witnesses (levels 1,2)
static i64 LN[MAXLEV];
static i64 LCAP[MAXLEV];
static int NLEV = 0;

// witness: op(3) la(2) ia(29) lb(2) ib(28)
#define MKW(op, la, ia, lb, ib) (((u64)(op) << 61) | ((u64)(la) << 59) | ((u64)(ia) << 30) | ((u64)(lb) << 28) | (u64)(ib))
#define W_OP(w) ((int)((w) >> 61))
#define W_LA(w) ((int)(((w) >> 59) & 3))
#define W_IA(w) ((i64)(((w) >> 30) & ((1ULL << 29) - 1)))
#define W_LB(w) ((int)(((w) >> 28) & 3))
#define W_IB(w) ((i64)((w) & ((1ULL << 28) - 1)))
enum { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2, OP_DIV = 3 };

// large buffers: 2 MB aligned with transparent huge pages (cuts TLB misses in scatters)
static void *big_alloc(size_t bytes) {
    size_t al = 1 << 21;
    size_t sz = (bytes + al - 1) / al * al;
    void *p = NULL;
    if (posix_memalign(&p, al, sz)) return malloc(bytes);
    madvise(p, sz, MADV_HUGEPAGE);
    return p;
}
typedef struct { u64 v, w; } VW;
static VW *TMP = NULL, *TMP2 = NULL; static i64 TMPCAP = 0;
static u64 *K1 = NULL, *K2 = NULL; static i64 KCAP = 0;

static void ensure_tmp(i64 n) {
    if (n > TMPCAP) {
        free(TMP); free(TMP2);
        TMPCAP = n + n / 4 + 1024;
        TMP = (VW *)big_alloc(sizeof(VW) * TMPCAP);
        TMP2 = (VW *)big_alloc(sizeof(VW) * TMPCAP);
    }
}
static void ensure_k(i64 n) {
    if (n > KCAP) {
        free(K1); free(K2);
        KCAP = n + n / 4 + 1024;
        K1 = (u64 *)big_alloc(8 * KCAP);
        K2 = (u64 *)big_alloc(8 * KCAP);
    }
}
static void ensure_level(int k, i64 n, int wit) {
    if (n > LCAP[k]) {
        free(LV[k]); free(LW[k]);
        LCAP[k] = n + n / 4 + 1024;
        LV[k] = (u64 *)big_alloc(8 * LCAP[k]);
        LW[k] = wit ? (u64 *)big_alloc(8 * LCAP[k]) : NULL;
    }
}

#define RB 12
#define RMASK ((1 << RB) - 1)
static i64 CNT[1 << 16];
static void radix_vw(VW *a, i64 n) {
    if (n < 2) return;
    VW *src = a, *dst = TMP2;
    u64 maxv = 0;
    for (i64 i = 0; i < n; i++) if (a[i].v > maxv) maxv = a[i].v;
    for (int sh = 0; sh == 0 || (sh < 64 && (maxv >> sh)); sh += RB) {
        memset(CNT, 0, sizeof(i64) << RB);
        for (i64 i = 0; i < n; i++) CNT[(src[i].v >> sh) & RMASK]++;
        i64 s = 0;
        for (int b = 0; b <= RMASK; b++) { i64 c = CNT[b]; CNT[b] = s; s += c; }
        for (i64 i = 0; i < n; i++) dst[CNT[(src[i].v >> sh) & RMASK]++] = src[i];
        VW *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, sizeof(VW) * n);
}
static void radix_u64(u64 *a, i64 n) {
    if (n < 2) return;
    u64 *src = a, *dst = K2;
    u64 maxv = 0;
    for (i64 i = 0; i < n; i++) if (a[i] > maxv) maxv = a[i];
    for (int sh = 0; sh == 0 || (sh < 64 && (maxv >> sh)); sh += RB) {
        memset(CNT, 0, sizeof(i64) << RB);
        for (i64 i = 0; i < n; i++) CNT[(src[i] >> sh) & RMASK]++;
        i64 s = 0;
        for (int b = 0; b <= RMASK; b++) { i64 c = CNT[b]; CNT[b] = s; s += c; }
        for (i64 i = 0; i < n; i++) dst[CNT[(src[i] >> sh) & RMASK]++] = src[i];
        u64 *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, 8 * n);
}

static inline i64 bsearch_u64(const u64 *a, i64 n, u64 x) {
    i64 lo = 0, hi = n;
    while (lo < hi) { i64 m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; }
    return (lo < n && a[lo] == x) ? lo : -1;
}
static inline i64 lower_bound(const u64 *a, i64 n, u64 x) {
    i64 lo = 0, hi = n;
    while (lo < hi) { i64 m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; }
    return lo;
}

// ---- fast membership filter for all levels: bitset keyed by hash ----
#define HBITS 27
static u64 *HSET = NULL;  // 2^HBITS bits
static inline u64 hmix(u64 x) { x ^= x >> 31; x *= 0x9E3779B97F4A7C15ULL; x ^= x >> 29; return x >> (64 - HBITS); }
static void hset_build(void) {
    if (!HSET) HSET = (u64 *)malloc((1ULL << HBITS) / 8);
    memset(HSET, 0, (1ULL << HBITS) / 8);
    for (int k = 0; k < NLEV; k++)
        for (i64 i = 0; i < LN[k]; i++) { u64 h = hmix(LV[k][i]); HSET[h >> 6] |= 1ULL << (h & 63); }
}
static inline int hset_maybe(u64 x) { u64 h = hmix(x); return (HSET[h >> 6] >> (h & 63)) & 1; }
// per-level filters: blocked Bloom filter, one 64-bit word per probe, 2 bits per element
static u64 *LF[8]; static int LFB[8]; static i64 LFCAP[8];
static inline u64 lfh(u64 x) { x ^= x >> 31; x *= 0x9E3779B97F4A7C15ULL; x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 32; return x; }
static void lf_build(int k) {
    int b = 10;  // log2 words
    while ((1LL << b) * 64 < LN[k] * 16 && b < 28) b++;
    i64 words = 1LL << b;
    if (words > LFCAP[k]) { free(LF[k]); LF[k] = (u64 *)big_alloc(8 * words); LFCAP[k] = words; }
    memset(LF[k], 0, 8 * words);
    LFB[k] = b;
    for (i64 i = 0; i < LN[k]; i++) { u64 h = lfh(LV[k][i]); LF[k][h >> (64 - b)] |= (1ULL << (h & 63)) | (1ULL << ((h >> 6) & 63)); }
}
static inline int lf_maybe(int k, u64 x) { u64 h = lfh(x); u64 m = (1ULL << (h & 63)) | (1ULL << ((h >> 6) & 63)); return (LF[k][h >> (64 - LFB[k])] & m) == m; }
static inline int lookup(u64 x, int maxlev, i64 *idx) {
    if (maxlev >= NLEV) maxlev = NLEV - 1;
    for (int k = 0; k <= maxlev; k++) {
        i64 i = bsearch_u64(LV[k], LN[k], x);
        if (i >= 0) { *idx = i; return k; }
    }
    return -1;
}
static inline int lookup_fast(u64 x, int maxlev, i64 *idx) {
    if (maxlev >= NLEV) maxlev = NLEV - 1;
    for (int k = 0; k <= maxlev; k++) {
        if (!lf_maybe(k, x)) continue;
        i64 i = bsearch_u64(LV[k], LN[k], x);
        if (i >= 0) { *idx = i; return k; }
    }
    return -1;
}

static void finalize_vw(int k, i64 n) {
    radix_vw(TMP, n);
    i64 m = 0;
    for (i64 i = 0; i < n; i++) {
        if (m > 0 && TMP[m - 1].v == TMP[i].v) continue;
        TMP[m++] = TMP[i];
    }
    i64 ptr[MAXLEV] = {0};
    i64 r = 0;
    for (i64 i = 0; i < m; i++) {
        u64 v = TMP[i].v; int dup = 0;
        for (int j = 0; j < k; j++) {
            while (ptr[j] < LN[j] && LV[j][ptr[j]] < v) ptr[j]++;
            if (ptr[j] < LN[j] && LV[j][ptr[j]] == v) { dup = 1; break; }
        }
        if (!dup) TMP[r++] = TMP[i];
    }
    ensure_level(k, r, 1);
    for (i64 i = 0; i < r; i++) { LV[k][i] = TMP[i].v; LW[k][i] = TMP[i].w; }
    LN[k] = r;
}
// LSD radix of a[0..n) using tmp (>= n), result in a.
static void radix_range(u64 *a, i64 n, u64 *tmp) {
    if (n < 2) return;
    u64 *src = a, *dst = tmp;
    u64 maxv = 0, minv = ~0ULL;
    for (i64 i = 0; i < n; i++) { if (a[i] > maxv) maxv = a[i]; if (a[i] < minv) minv = a[i]; }
    u64 diff = maxv ^ minv;
    static i64 C8[256];
    for (int sh = 0; sh < 64 && (diff >> sh); sh += 8) {
        memset(C8, 0, sizeof(C8));
        for (i64 i = 0; i < n; i++) C8[(src[i] >> sh) & 255]++;
        i64 s = 0;
        for (int b = 0; b < 256; b++) { i64 c = C8[b]; C8[b] = s; s += c; }
        for (i64 i = 0; i < n; i++) dst[C8[(src[i] >> sh) & 255]++] = src[i];
        u64 *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, 8 * n);
}
static inline void isort_u64(u64 *a, i64 n) {
    if (n <= 1) return;
    if (n <= 32) {
        // branchless rank sort (stable): rank_i = #{j: a_j < a_i} + #{j < i: a_j == a_i}
        u64 b[32];
        for (i64 i = 0; i < n; i++) {
            u64 v = a[i]; i64 r = 0;
            for (i64 j = 0; j < n; j++) r += (a[j] < v) | ((a[j] == v) & (j < i));
            b[r] = v;
        }
        memcpy(a, b, 8 * n);
        return;
    }
    for (i64 i = 1; i < n; i++) {
        u64 v = a[i]; i64 j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}
// two-level MSD radix on a log-linear key (bit length, then mantissa bits), then small sorts.
#define D1B 10
#define D1N (65 << D1B)
static i64 MC1[D1N + 1];
static inline int msd_d1(u64 v) { int b = 64 - __builtin_clzll(v); u64 x = v << (64 - b); return (b << D1B) | (int)((x >> (63 - D1B)) & ((1 << D1B) - 1)); }
static i64 RCNT[24][4097];
// a[0..n): all values have bit length b and equal bits of x = v << (64 - b) above position top.
static void rec_sort(u64 *a, i64 n, u64 *tmp, int top, int b, int depth) {
    if (n <= 32) { isort_u64(a, n); return; }
    if (top <= 0) return;
    if (depth >= 23) { radix_range(a, n, tmp); return; }
    int lg = 63 - __builtin_clzll((u64)n);
    int w = lg - 1; if (w < 4) w = 4; if (w > 12) w = 12; if (w > top) w = top;
    int nb = 1 << w; u64 msk = (u64)nb - 1;
    i64 *cnt = RCNT[depth];
    memset(cnt, 0, 8 * (nb + 1));
    int sh = 64 - b, s2 = top - w;
    for (i64 i = 0; i < n; i++) cnt[(((a[i] << sh) >> s2) & msk) + 1]++;
    for (int d = 0; d < nb; d++) cnt[d + 1] += cnt[d];
    // scatter a -> tmp using running positions (cnt[d] = start of d); restore starts afterwards
    for (i64 i = 0; i < n; i++) { u64 v = a[i]; tmp[cnt[((v << sh) >> s2) & msk]++] = v; }
    memcpy(a, tmp, 8 * n);
    i64 prev = 0;
    for (int d = 0; d < nb; d++) {
        i64 end = cnt[d];  // after scatter cnt[d] = end of bucket d
        i64 m = end - prev;
        if (m > 1) rec_sort(a + prev, m, tmp + prev, s2, b, depth + 1);
        prev = end;
    }
}
// src -> dst (sorted), same preconditions as rec_sort.
static void rec_sort_from(const u64 *src, u64 *dst, u64 *scratch, i64 n, int top, int b) {
    if (n <= 32) { memcpy(dst, src, 8 * n); isort_u64(dst, n); return; }
    int lg = 63 - __builtin_clzll((u64)n);
    int w = lg - 1; if (w < 4) w = 4; if (w > 12) w = 12; if (w > top) w = top;
    if (w <= 0) { memcpy(dst, src, 8 * n); return; }
    int nb = 1 << w; u64 msk = (u64)nb - 1;
    i64 *cnt = RCNT[23];
    memset(cnt, 0, 8 * (nb + 1));
    int sh = 64 - b, s2 = top - w;
    for (i64 i = 0; i < n; i++) cnt[(((src[i] << sh) >> s2) & msk) + 1]++;
    for (int d = 0; d < nb; d++) cnt[d + 1] += cnt[d];
    for (i64 i = 0; i < n; i++) { u64 v = src[i]; dst[cnt[((v << sh) >> s2) & msk]++] = v; }
    i64 prev = 0;
    for (int d = 0; d < nb; d++) {
        i64 end = cnt[d];
        i64 m = end - prev;
        if (m > 1) rec_sort(dst + prev, m, scratch + prev, s2, b, 1);
        prev = end;
    }
}
// Sort a[0..n) (tmp scratch, a is clobbered), drop duplicates and values present in levels < k,
// write the result to out (capacity >= n). Returns the number written.
static i64 msd_sort_dedupe(u64 *a, i64 n, u64 *tmp, int k, u64 *out) {
    static i64 C4[4][D1N];
    memset(C4, 0, sizeof(C4));
    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        C4[0][msd_d1(a[i])]++; C4[1][msd_d1(a[i + 1])]++;
        C4[2][msd_d1(a[i + 2])]++; C4[3][msd_d1(a[i + 3])]++;
    }
    for (; i < n; i++) C4[0][msd_d1(a[i])]++;
    i64 s = 0;
    for (int bk = 0; bk < D1N; bk++) { i64 c = C4[0][bk] + C4[1][bk] + C4[2][bk] + C4[3][bk]; MC1[bk] = s; s += c; }
    MC1[D1N] = n;
    {
        static i64 pos[D1N];
        memcpy(pos, MC1, sizeof(pos));
        for (i = 0; i + 2 <= n; i += 2) {
            u64 v0 = a[i], v1 = a[i + 1];
            int d0 = msd_d1(v0), d1 = msd_d1(v1);
            if (d0 == d1) { i64 p = pos[d0]; tmp[p] = v0; tmp[p + 1] = v1; pos[d0] = p + 2; }
            else { tmp[pos[d0]++] = v0; tmp[pos[d1]++] = v1; }
        }
        for (; i < n; i++) { u64 v = a[i]; tmp[pos[msd_d1(v)]++] = v; }
    }
    i64 ptr[MAXLEV] = {0};
    i64 r = 0;
    u64 last = 0;
    for (int bk = 0; bk < D1N; bk++) {
        i64 st = MC1[bk], m = MC1[bk + 1] - st;
        if (m <= 0) continue;
        u64 *d = a + st;
        if (m == 1) d[0] = tmp[st];
        else rec_sort_from(tmp + st, d, tmp + st, m, 63 - D1B, bk >> D1B);
        for (i64 q = 0; q < m; q++) {
            u64 v = d[q];
            if (v == last) continue;
            last = v;
            int dup = 0;
            for (int j = 0; j < k; j++) {
                while (ptr[j] < LN[j] && LV[j][ptr[j]] < v) ptr[j]++;
                if (ptr[j] < LN[j] && LV[j][ptr[j]] == v) { dup = 1; break; }
            }
            if (!dup) out[r++] = v;
        }
    }
    return r;
}
static void finalize_k(int k, i64 n) {
    ensure_level(k, n, 0);
    LN[k] = msd_sort_dedupe(K1, n, K2, k, LV[k]);
}

static inline i64 emit_vw(i64 n, u64 x, int la, i64 ia, u64 y, int lb, i64 ib) {
    u64 s = x + y;
    if (s <= LIM) { TMP[n].v = s; TMP[n].w = MKW(OP_ADD, la, ia, lb, ib); n++; }
    u128 p = (u128)x * y;
    if (p <= LIM) { TMP[n].v = (u64)p; TMP[n].w = MKW(OP_MUL, la, ia, lb, ib); n++; }
    if (x > y) {
        TMP[n].v = x - y; TMP[n].w = MKW(OP_SUB, la, ia, lb, ib); n++;
        if (y > 1 && x % y == 0) { TMP[n].v = x / y; TMP[n].w = MKW(OP_DIV, la, ia, lb, ib); n++; }
    } else if (y > x) {
        TMP[n].v = y - x; TMP[n].w = MKW(OP_SUB, lb, ib, la, ia); n++;
        if (x > 1 && y % x == 0) { TMP[n].v = y / x; TMP[n].w = MKW(OP_DIV, lb, ib, la, ia); n++; }
    } else {
        TMP[n].v = 1; TMP[n].w = MKW(OP_DIV, la, ia, lb, ib); n++;
    }
    return n;
}
static inline i64 emit_k(i64 n, u64 x, u64 y) {
    u64 s = x + y;
    if (s <= LIM) K1[n++] = s;
    u128 p = (u128)x * y;
    if (p <= LIM) K1[n++] = (u64)p;
    if (x > y) {
        K1[n++] = x - y;
        if (y > 1 && x % y == 0) K1[n++] = x / y;
    } else if (y > x) {
        K1[n++] = y - x;
        if (x > 1 && y % x == 0) K1[n++] = y / x;
    }
    return n;
}

static int POOLN = 0;
static u64 POOL[256];

// divisibility test by precomputed constants: d = 2^k * o (o odd)
typedef struct { u64 inv, lim; int k; } DivC;
static inline DivC divc_make(u64 d) {
    DivC c; c.k = __builtin_ctzll(d); u64 o = d >> c.k;
    u64 inv = o; for (int i = 0; i < 6; i++) inv *= 2 - o * inv;
    c.inv = inv; c.lim = ~0ULL / o; return c;
}
static inline int divc_test(u64 x, DivC c) {
    if (x & ((1ULL << c.k) - 1)) return 0;
    return ((x >> c.k) * c.inv) <= c.lim;
}
// emit x op y where y has precomputed divisor constants dc (y >= 1)
static inline i64 emit_kd(i64 n, u64 x, u64 y, DivC dc) {
    u64 s = x + y;
    if (s <= LIM) K1[n++] = s;
    u128 p = (u128)x * y;
    if (p <= LIM) K1[n++] = (u64)p;
    if (x > y) {
        K1[n++] = x - y;
        if (y > 1 && divc_test(x, dc)) K1[n++] = x / y;
    } else if (y > x) {
        K1[n++] = y - x;
        if (x > 1 && y % x == 0) K1[n++] = y / x;
    }
    return n;
}

// Build levels 0..maxlev (maxlev < MAXLEV). L0..L2 keep witnesses. L3 is always complete
// (all (2,0) and (1,1) combinations). Level k >= 4 is built from pairs (k-1-j, j), j = 0, 1, ...
// while the cumulative candidate estimate stays <= cap (so deep levels are partial).
// keep: if nonzero and deep levels exist, only L0..L2 are rebuilt and deep levels are kept (stale).
static int HAVEDEEP = 0, DEEPN = 0;
static unsigned PAIRMASK[MAXLEV];
i64 build_levels(const u64 *pool, int n, u64 lim, int maxlev, int keep, i64 cap, i64 *out) {
    LIM = lim;
    if (maxlev >= MAXLEV) maxlev = MAXLEV - 1;
    int keepdeep = keep && HAVEDEEP;
    if (!keep) HAVEDEEP = 0;
    POOLN = n;
    for (int i = 0; i < n; i++) POOL[i] = pool[i];
    ensure_tmp((i64)n * n * 4 + 64);
    for (int i = 0; i < n; i++) { TMP[i].v = pool[i]; TMP[i].w = 0; }
    finalize_vw(0, n);
    NLEV = 1;
    if (maxlev >= 1) {
        i64 m = 0;
        for (i64 i = 0; i < LN[0]; i++)
            for (i64 j = i; j < LN[0]; j++)
                m = emit_vw(m, LV[0][i], 0, i, LV[0][j], 0, j);
        finalize_vw(1, m);
        NLEV = 2;
    }
    if (maxlev >= 2) {
        ensure_tmp(LN[1] * LN[0] * 4 + 64);
        i64 m = 0;
        for (i64 a = 0; a < LN[1]; a++) {
            u64 x = LV[1][a];
            for (i64 j = 0; j < LN[0]; j++) m = emit_vw(m, x, 1, a, LV[0][j], 0, j);
        }
        finalize_vw(2, m);
        NLEV = 3;
    }
    if (keepdeep) {
        NLEV = DEEPN;
    } else if (maxlev >= 3) {
        for (int k = 3; k <= maxlev; k++) {
            unsigned mask = 0; double need = 0;
            for (int j = 0; j <= (k - 1) / 2; j++) {
                int i = k - 1 - j;
                double c = (i == j) ? 2.0 * LN[i] * (LN[i] + 1) : 4.0 * LN[i] * LN[j];
                if (k == 3 || need + c <= (double)cap) { mask |= 1u << j; need += c; }
                else break;
            }
            if (!mask) break;
            ensure_k((i64)need + 64);
            i64 m = 0;
            for (int j = 0; j <= (k - 1) / 2; j++) {
                if (!(mask >> j & 1)) continue;
                int i = k - 1 - j;
                for (i64 b = 0; b < LN[j]; b++) {
                    u64 y = LV[j][b];
                    DivC dc = divc_make(y);
                    i64 a0 = (i == j) ? b : 0;
                    const u64 *X = LV[i];
                    for (i64 a = a0; a < LN[i]; a++) m = emit_kd(m, X[a], y, dc);
                }
            }
            finalize_k(k, m);
            PAIRMASK[k] = mask;
            NLEV = k + 1;
            if (LN[k] == 0) { NLEV = k; break; }
        }
        HAVEDEEP = (NLEV > 3);
        DEEPN = NLEV;
    }
    for (int k = 0; k < NLEV; k++) if (!keepdeep || k < 3) lf_build(k);
    i64 tot = 0;
    for (int k = 0; k < MAXLEV; k++) { out[k] = (k < NLEV) ? LN[k] : 0; tot += out[k]; }
    return tot;
}

// ---------------- output of plans ----------------
static u64 *OUT; static int OUTN; static int OUTCAP;
static void emit_step(int op, u64 a, u64 b, u64 r) {
    if (OUTN + 4 > OUTCAP) return;
    OUT[OUTN++] = (u64)op; OUT[OUTN++] = a; OUT[OUTN++] = b; OUT[OUTN++] = r;
}
static int expand(int lev, i64 idx);
// derive witness for v in L3: v = x op a (x in L2, a in L0) or x op y (x, y in L1)
static int DERIVE_FAIL = 0;
static inline int lk(u64 x, int maxlev, int *lev, i64 *idx) {
    int k = lookup_fast(x, maxlev, idx);
    if (k < 0) return 0;
    *lev = k; return 1;
}
static int derive_deep(u64 v, int k) {
    i64 jx; int lj;
    for (int j = 0; j <= (k - 1) / 2; j++) {
        if (!(PAIRMASK[k] >> j & 1)) continue;
        int lxm = k - 1 - j;
        int ly = j;
        for (i64 i = 0; i < LN[ly]; i++) {
            u64 a = LV[ly][i];
            if (a < v && lk(v - a, lxm, &lj, &jx)) { expand(lj, jx); expand(ly, i); emit_step(OP_ADD, v - a, a, v); return 1; }
            if (v + a <= LIM && lk(v + a, lxm, &lj, &jx)) { expand(lj, jx); expand(ly, i); emit_step(OP_SUB, v + a, a, v); return 1; }
            if (a > v && lk(a - v, lxm, &lj, &jx)) { expand(lj, jx); expand(ly, i); emit_step(OP_SUB, a, a - v, v); return 1; }
            if (a >= 2 && v % a == 0 && lk(v / a, lxm, &lj, &jx)) { expand(lj, jx); expand(ly, i); emit_step(OP_MUL, v / a, a, v); return 1; }
            if (a >= 2 && (u128)v * a <= LIM && lk(v * a, lxm, &lj, &jx)) { expand(lj, jx); expand(ly, i); emit_step(OP_DIV, v * a, a, v); return 1; }
            if (a > v && a % v == 0 && lk(a / v, lxm, &lj, &jx)) { expand(lj, jx); expand(ly, i); emit_step(OP_DIV, a, a / v, v); return 1; }
        }
    }
    DERIVE_FAIL = 1;
    return 0;
}
int derive_failed(void) { int f = DERIVE_FAIL; DERIVE_FAIL = 0; return f; }
static int expand(int lev, i64 idx) {
    if (lev == 0) return 1;
    if (lev >= 3) return derive_deep(LV[lev][idx], lev);
    u64 w = LW[lev][idx];
    int la = W_LA(w), lb = W_LB(w);
    i64 ia = W_IA(w), ib = W_IB(w);
    expand(la, ia);
    expand(lb, ib);
    emit_step(W_OP(w), LV[la][ia], LV[lb][ib], LV[lev][idx]);
    return 1;
}
int get_outn(void) { return OUTN; }

int expand_value(u64 v, int maxlev, u64 *out, int outcap) {
    OUT = out; OUTN = 0; OUTCAP = outcap;
    i64 idx; int k = lookup(v, maxlev, &idx);
    if (k < 0) return -1;
    expand(k, idx);
    return OUTN;
}
int level_of(u64 v, int maxlev) { i64 idx; return lookup(v, maxlev, &idx); }
i64 level_size(int k) { return (k < NLEV) ? LN[k] : 0; }

// ---------------- pair search ----------------
#define SMALL 50000
typedef struct { int op; int la; i64 ia; int lb; i64 ib; } PairRes;

// t = u op w with u in level a, w in level b (either order for - and /)
static int pair_search(u64 t, int a, int b, PairRes *res) {
    const u64 *A = LV[a], *B = LV[b];
    i64 na = LN[a], nb = LN[b];
    if (nb <= SMALL || na <= SMALL) {
        int sw = (nb > na);
        const u64 *S = sw ? A : B; i64 ns = sw ? na : nb; int ls = sw ? a : b;
        const u64 *G = sw ? B : A; i64 ng = sw ? nb : na; int lg = sw ? b : a;
        i64 j;
#define hset_maybe(x) lf_maybe(lg, (x))
        for (i64 i = 0; i < ns; i++) {
            u64 w = S[i];
            if (w < t) {
                u64 x = t - w;
                if (hset_maybe(x) && (j = bsearch_u64(G, ng, x)) >= 0) { res->op = OP_ADD; res->la = lg; res->ia = j; res->lb = ls; res->ib = i; return 1; }
            }
            if (w + t <= LIM) {
                u64 x = w + t;
                if (hset_maybe(x) && (j = bsearch_u64(G, ng, x)) >= 0) { res->op = OP_SUB; res->la = lg; res->ia = j; res->lb = ls; res->ib = i; return 1; }
            }
            if (w > t) {
                u64 x = w - t;
                if (hset_maybe(x) && (j = bsearch_u64(G, ng, x)) >= 0) { res->op = OP_SUB; res->la = ls; res->ia = i; res->lb = lg; res->ib = j; return 1; }
            }
            if (w >= 2 && w < t && t % w == 0) {
                u64 x = t / w;
                if (hset_maybe(x) && (j = bsearch_u64(G, ng, x)) >= 0) { res->op = OP_MUL; res->la = lg; res->ia = j; res->lb = ls; res->ib = i; return 1; }
            }
            if (w >= 2) {
                u128 p = (u128)t * w;
                if (p <= LIM && hset_maybe((u64)p) && (j = bsearch_u64(G, ng, (u64)p)) >= 0) { res->op = OP_DIV; res->la = lg; res->ia = j; res->lb = ls; res->ib = i; return 1; }
                if (w > t && w % t == 0) {
                    u64 x = w / t;
                    if (x >= 2 && hset_maybe(x) && (j = bsearch_u64(G, ng, x)) >= 0) { res->op = OP_DIV; res->la = ls; res->ia = i; res->lb = lg; res->ib = j; return 1; }
                }
            }
        }
        return 0;
#undef hset_maybe
    }
    // both large: two-pointer for + and -
    {
        i64 i = 0, j = lower_bound(B, nb, t) - 1;
        while (i < na && j >= 0) {
            u64 u = A[i];
            if (u >= t) break;
            u64 s = u + B[j];
            if (s == t) { res->op = OP_ADD; res->la = a; res->ia = i; res->lb = b; res->ib = j; return 1; }
            if (s < t) i++; else j--;
        }
    }
    for (int dir = 0; dir < 2; dir++) {
        const u64 *X = dir ? B : A; i64 nx = dir ? nb : na; int lx = dir ? b : a;
        const u64 *Y = dir ? A : B; i64 ny = dir ? na : nb; int ly = dir ? a : b;
        i64 i = lower_bound(X, nx, t + 1), j = 0;
        while (i < nx && j < ny) {
            if (X[i] <= Y[j]) { i++; continue; }
            u64 d = X[i] - Y[j];
            if (d == t) { res->op = OP_SUB; res->la = lx; res->ia = i; res->lb = ly; res->ib = j; return 1; }
            if (d < t) i++; else j++;
        }
        if (a == b) break;
    }
    for (int dir = 0; dir < 2; dir++) {
        const u64 *X = dir ? B : A; i64 nx = dir ? nb : na; int lx = dir ? b : a;
        const u64 *Y = dir ? A : B; i64 ny = dir ? na : nb; int ly = dir ? a : b;
        for (i64 i = 0; i < nx; i++) {
            u64 u = X[i];
            if ((u128)u * u > t) break;
            if (u >= 2 && t % u == 0) {
                i64 j = bsearch_u64(Y, ny, t / u);
                if (j >= 0) { res->op = OP_MUL; res->la = lx; res->ia = i; res->lb = ly; res->ib = j; return 1; }
            }
        }
        // t = x / w, w in Y (w >= 2, ascending), x = t * w in X
        {
            i64 j0 = lower_bound(Y, ny, 2);
            i64 j1 = lower_bound(Y, ny, LIM / t + 1);
            if (j1 - j0 <= 64) {
                for (i64 j = j0; j < j1; j++) {
                    u128 p = (u128)t * Y[j];
                    if (p > LIM) break;
                    i64 i = bsearch_u64(X, nx, (u64)p);
                    if (i >= 0) { res->op = OP_DIV; res->la = lx; res->ia = i; res->lb = ly; res->ib = j; return 1; }
                }
            } else {
                i64 i = lower_bound(X, nx, (u64)((u128)t * 2 <= LIM ? t * 2 : LIM));
                for (i64 j = j0; j < j1 && i < nx; j++) {
                    u128 p128 = (u128)t * Y[j];
                    if (p128 > LIM) break;
                    u64 p = (u64)p128;
                    while (i < nx && X[i] < p) i++;
                    if (i < nx && X[i] == p) { res->op = OP_DIV; res->la = lx; res->ia = i; res->lb = ly; res->ib = j; return 1; }
                }
            }
        }
        if (a == b) break;  // same level: the second direction repeats the first
    }
    return 0;
}

// cheapest (direct or pair) cost for t with cost <= maxcost. Returns cost or -1.
int find_pair(u64 t, int maxcost, u64 *out, int outcap) {
    OUT = out; OUTN = 0; OUTCAP = outcap;
    i64 idx;
    int k = lookup(t, NLEV - 1, &idx);
    if (k >= 0) { if (k <= maxcost) { expand(k, idx); return k; } return -1; }
    for (int c = (NLEV < 4 ? NLEV : 4); c <= maxcost; c++) {
        for (int b = 0; b <= (c - 1) / 2; b++) {
            int a = c - 1 - b;
            if (a >= NLEV) continue;
            PairRes r;
            if (pair_search(t, a, b, &r)) {
                expand(r.la, r.ia); expand(r.lb, r.ib);
                emit_step(r.op, LV[r.la][r.ia], LV[r.lb][r.ib], t);
                return c;
            }
        }
    }
    return -1;
}

// ---------------- window search: t = X*Y +- Z ----------------
static u64 rng_state = 88172645463325252ULL;
static inline u64 xrand(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }
void set_seed(u64 s) { rng_state = s * 2654435761ULL + 88172645463325252ULL; if (!rng_state) rng_state = 1; }

static double est_cands(u64 t, int lx, int ly, u64 zmax) {
    const u64 *X = LV[lx]; i64 nx = LN[lx];
    const u64 *Y = LV[ly]; i64 ny = LN[ly];
    i64 hi = lower_bound(X, nx, t + zmax + 1);
    i64 lo = lower_bound(X, nx, 2);
    if (hi <= lo) return 0;
    i64 cnt = hi - lo;
    int ns = cnt < 256 ? (int)cnt : 256;
    double tot = 0;
    for (int s = 0; s < ns; s++) {
        i64 i = (cnt <= 256) ? lo + s : lo + (i64)(xrand() % (u64)cnt);
        u64 x = X[i];
        u64 l = (t > zmax) ? t - zmax : 1;
        u64 ylo = (l + x - 1) / x, yhi = (t + zmax) / x;
        if (yhi < ylo) continue;
        tot += (double)(lower_bound(Y, ny, yhi + 1) - lower_bound(Y, ny, ylo));
    }
    return tot * (double)cnt / ns;
}

static i64 WCAND = 0;
i64 window_cands(void) { return WCAND; }

// Tries costs c = 4 .. bestcost-1; for each (lx <= ly, lz = c-2-lx-ly) combo the window
// half-width is the largest power of two whose estimated candidate count <= budget.
static double WEMIN = 0.02;
void set_wemin(double e) { WEMIN = e; }
int find_window(u64 t, int bestcost, double budget, u64 *out, int outcap) {
    OUT = out; OUTN = 0; OUTCAP = outcap;
    int best = bestcost;
    int bxl = -1, byl = -1, bzl = -1; i64 bxi = 0, byi = 0, bzi = 0; int bsign = 0;
    int L = NLEV - 1;
    WCAND = 0;
    for (int c = 4; c < best; c++) {
        for (int lx = 0; lx <= L; lx++) {
            for (int ly = lx; ly <= L; ly++) {
                int lz = c - 2 - lx - ly;
                if (lz < 0 || lz > L) continue;
                u64 zmax = t;
                int bits = 64 - __builtin_clzll(t);
                double e = est_cands(t, lx, ly, zmax);
                while (e > budget && bits > 1) { bits--; zmax = 1ULL << bits; e = est_cands(t, lx, ly, zmax); }
                {
                    // expected hits: candidates * density of level lz below zmax
                    double nzb = (double)lower_bound(LV[lz], LN[lz], zmax + 1);
                    double eh = e * nzb / (double)zmax;
                    if (eh < WEMIN) continue;
                }
                const u64 *X = LV[lx]; i64 nx = LN[lx];
                const u64 *Y = LV[ly]; i64 ny = LN[ly];
                const u64 *Z = LV[lz]; i64 nz = LN[lz];
                u64 l = (t > zmax) ? t - zmax : 1;
                u64 hiv = t + zmax;
                double dl = (double)l, dh = (double)hiv;
                i64 jp = ny;
                // ring buffer of pending candidates for prefetching filter words
                enum { WRB = 16 };
                u64 rz[WRB]; i64 ri[WRB], rj[WRB]; int rs[WRB]; int rn = 0, rh = 0;
                int fb = LFB[lz]; const u64 *F = LF[lz];
#define FH(z) lfh(z)
#define PROCESS_ONE() do { \
                    u64 _z = rz[rh]; u64 _h = FH(_z); u64 _m = (1ULL << (_h & 63)) | (1ULL << ((_h >> 6) & 63)); \
                    if ((F[_h >> (64 - fb)] & _m) == _m) { \
                        i64 zi = bsearch_u64(Z, nz, _z); \
                        if (zi >= 0) { best = c; bxl = lx; bxi = ri[rh]; byl = ly; byi = rj[rh]; bzl = lz; bzi = zi; bsign = rs[rh]; goto found; } \
                    } \
                    rh = (rh + 1) % WRB; rn--; } while (0)
                for (i64 i = lower_bound(X, nx, 2); i < nx; i++) {
                    u64 x = X[i];
                    if (x > hiv) break;
                    double dx = (double)x;
                    u64 ylo = (u64)(dl / dx), yhi = (u64)(dh / dx);
                    // exact correction: ylo = ceil(l/x), yhi = floor(hiv/x)
                    while ((u128)ylo * x < l) ylo++;
                    while (ylo > 1 && (u128)(ylo - 1) * x >= l) ylo--;
                    while ((u128)yhi * x > hiv) yhi--;
                    while ((u128)(yhi + 1) * x <= hiv) yhi++;
                    if (ylo < 1) ylo = 1;
                    while (jp > 0 && Y[jp - 1] >= ylo) jp--;
                    if (yhi < ylo) continue;
                    for (i64 j = jp; j < ny && Y[j] <= yhi; j++) {
                        u128 p128 = (u128)x * Y[j];
                        if (p128 > LIM) break;
                        WCAND++;
                        u64 p = (u64)p128;
                        u64 z = (p < t) ? (t - p) : (p - t);
                        if (z == 0) continue;
                        if (rn == WRB) PROCESS_ONE();
                        int slot = (rh + rn) % WRB;
                        rz[slot] = z; ri[slot] = i; rj[slot] = j; rs[slot] = (p < t) ? 1 : -1; rn++;
                        u64 h = FH(z);
                        __builtin_prefetch(&F[h >> (64 - fb)]);
                    }
                }
                while (rn > 0) PROCESS_ONE();
#undef PROCESS_ONE
#undef FH
            }
        }
    }
found:
    if (bxl < 0) return -1;
    {
        expand(bxl, bxi); expand(byl, byi);
        u64 x = LV[bxl][bxi], y = LV[byl][byi];
        u64 p = x * y;
        emit_step(OP_MUL, x, y, p);
        expand(bzl, bzi);
        u64 z = LV[bzl][bzi];
        if (bsign > 0) emit_step(OP_ADD, p, z, t);
        else emit_step(OP_SUB, p, z, t);
    }
    return best;
}

// Horner candidates: X in levels <= ax (X >= 2), Z = t mod X (variant +) or X - t mod X (variant -),
// Z == 0 or Z in levels <= cz. Writes quads (X, Z, cx | (minus<<8), cz).
int horner_candidates(u64 t, int ax, int cz, u64 *out, int maxq) {
    if (ax >= NLEV) ax = NLEV - 1;
    if (cz >= NLEV) cz = NLEV - 1;
    int q = 0;
    for (int lx = 0; lx <= ax; lx++) {
        for (i64 i = 0; i < LN[lx]; i++) {
            u64 x = LV[lx][i];
            if (x < 2 || x > t / 2) continue;
            u64 z = t % x;
            i64 idx; int lz;
            if (z == 0) {
                if (q < maxq) { out[4*q] = x; out[4*q+1] = 0; out[4*q+2] = lx; out[4*q+3] = 0; q++; }
                continue;
            }
            if ((lz = lookup_fast(z, cz, &idx)) >= 0 && q < maxq) { out[4*q] = x; out[4*q+1] = z; out[4*q+2] = lx; out[4*q+3] = lz; q++; }
            u64 z2 = x - z;
            u128 up = (u128)x * (t / x + 1);
            if (up <= LIM && (lz = lookup_fast(z2, cz, &idx)) >= 0 && q < maxq) { out[4*q] = x; out[4*q+1] = z2; out[4*q+2] = lx | 256; out[4*q+3] = lz; q++; }
        }
    }
    return q;
}

void level_hist(int k, i64 *h) {
    for (int b = 0; b < 65; b++) h[b] = 0;
    for (i64 i = 0; i < LN[k]; i++) h[64 - __builtin_clzll(LV[k][i])]++;
}

// ---------------- regret pick (batch over all uncovered targets) ----------------
// For each target the cheapest plan (direct lookup, or pair search at cost 4..maxc, the same search
// order as find_pair). Commits argmin of (cost - alpha * bitlen, cost, t) lexicographically.
// Searches run breadth-first by cost and skip any (target, cost) that cannot beat the current best.
// Returns the chosen index (or -1); writes the plan to out and the cost to *cost_out.
static double RP_EPS = 0.0;
void set_pair_eps(double e) { RP_EPS = e; }
int regret_pick(const u64 *ts, int n, int maxc, double alpha, u64 *out, int outcap, int *cost_out) {
    OUT = out; OUTN = 0; OUTCAP = outcap;
    static int st_lev[4096]; static i64 st_idx[4096]; static int st_c[4096]; static PairRes st_r[4096];
    static char alive[4096];
    if (n > 4096) n = 4096;
    int bi = -1; double bsc = 0; int bc = 0;
#define BETTER(sc, c, t) (bi < 0 || (sc) < bsc || ((sc) == bsc && ((c) < bc || ((c) == bc && (t) < ts[bi]))))
    for (int i = 0; i < n; i++) {
        u64 t = ts[i];
        i64 idx; int k = lookup(t, NLEV - 1, &idx);
        st_c[i] = -1; alive[i] = 1;
        if (k >= 0) {
            alive[i] = 0;
            if (k <= maxc) {
                st_lev[i] = k; st_idx[i] = idx; st_c[i] = k;
                double sc = (double)k - alpha * (double)(64 - __builtin_clzll(t));
                if (BETTER(sc, k, t)) { bi = i; bsc = sc; bc = k; }
            }
        }
    }
    for (int c = (NLEV < 4 ? NLEV : 4); c <= maxc; c++) {
        for (int i = 0; i < n; i++) {
            if (!alive[i]) continue;
            u64 t = ts[i];
            double sc = (double)c - alpha * (double)(64 - __builtin_clzll(t));
            if (!BETTER(sc, c, t)) { alive[i] = 0; continue; }
            for (int b = 0; b <= (c - 1) / 2; b++) {
                int a = c - 1 - b;
                if (a >= NLEV) continue;
                // skip level pairs whose expected number of solutions for t is negligible
                if (RP_EPS > 0 && a >= 2 && 4.0 * (double)LN[a] * (double)LN[b] < RP_EPS * (double)t) continue;
                PairRes r;
                if (pair_search(t, a, b, &r)) {
                    st_r[i] = r; st_c[i] = c; st_lev[i] = -1; alive[i] = 0;
                    bi = i; bsc = sc; bc = c;
                    break;
                }
            }
        }
    }
#undef BETTER
    if (bi < 0) return -1;
    if (st_lev[bi] >= 0) expand(st_lev[bi], st_idx[bi]);
    else {
        PairRes r = st_r[bi];
        expand(r.la, r.ia); expand(r.lb, r.ib);
        emit_step(r.op, LV[r.la][r.ia], LV[r.lb][r.ib], ts[bi]);
    }
    *cost_out = bc;
    return bi;
}

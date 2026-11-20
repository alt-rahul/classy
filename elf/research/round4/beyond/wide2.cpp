// wide.cpp - exact-ish wide-pool rebuild of one value.
// Input (stdin):  LIMIT T CMIN CMAX HEAVY \n N \n v1 .. vN
// Finds a tree for T with at most c new values (c from CMIN to CMAX, first success) whose leaves are pool values.
// Shapes: T = X*Y +- R with X in S_a, Y in S_b (a<=b<=2) and R of cost <= 3; exact-division shapes with Y of cost 3;
// chains T = u +- n, T = n*s with u, s in the pool (or u a one-step value) and n solved by the product shapes.
// Output: "SOL k" + k lines "a op b v" (new values in dependency order), or "NONE".
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64; typedef unsigned __int128 u128;
static u64 LIM;
static vector<u64> V;
struct E1 { u64 v, a, b; char op; };
static vector<E1> S1; static vector<u64> S1v, S2;
static vector<u64> bl1, bl2;
static const int B1 = 27, B2 = 32;
static inline u64 h1(u64 x) { return (x * 0x9E3779B97F4A7C15ULL) >> (64 - B1); }
static inline u64 h2(u64 x) { return (x * 0xC2B2AE3D27D4EB4FULL) >> (64 - B2); }
static inline bool inV(u64 x) { return binary_search(V.begin(), V.end(), x); }
static inline bool in1(u64 x) { u64 h = h1(x); if (!((bl1[h >> 6] >> (h & 63)) & 1)) return false; return binary_search(S1v.begin(), S1v.end(), x); }
static inline bool in2(u64 x) { u64 h = h2(x); if (!((bl2[h >> 6] >> (h & 63)) & 1)) return false; return binary_search(S2.begin(), S2.end(), x); }
static inline int cost(u64 n) { if (inV(n)) return 0; if (in1(n)) return 1; if (in2(n)) return 2; return 9; }
struct Op { u64 a; char op; u64 b; u64 v; };
typedef vector<Op> Ops;
static double now() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }
static double T0, TLIM = 1e9;

static void build1(u64 n, Ops &o) {
    auto it = lower_bound(S1v.begin(), S1v.end(), n); const E1 &e = S1[it - S1v.begin()];
    o.push_back({e.a, e.op, e.b, n});
}
// generic inversion: n = m op u or u op m, with membership test F on m. Returns op layout.
template <class F> static inline bool inv(u64 n, u64 u, F in, Op &r) {
    if (n > u && in(n - u)) { r = {n - u, '+', u, n}; return true; }
    if (n + u <= LIM && in(n + u)) { r = {n + u, '-', u, n}; return true; }
    if (u > n && in(u - n)) { r = {u, '-', u - n, n}; return true; }
    if (u > 1 && n % u == 0 && in(n / u)) { r = {n / u, '*', u, n}; return true; }
    if (u > 1 && (u128)n * u <= LIM && in(n * u)) { r = {n * u, '/', u, n}; return true; }
    if (n > 1 && u % n == 0 && in(u / n)) { r = {u, '/', u / n, n}; return true; }
    return false;
}
static bool build2(u64 n, Ops &o) {
    Op r;
    for (u64 u : V) if (inv(n, u, in1, r)) { u64 m = (r.a == u) ? r.b : r.a; if (r.a == u && r.b == u) m = u; build1(m, o); o.push_back(r); return true; }
    if (n % 2 == 0 && in1(n / 2)) { build1(n / 2, o); o.push_back({n / 2, '+', n / 2, n}); return true; }
    u64 s = (u64)sqrtl((long double)n); while ((u128)s * s > n) s--; while ((u128)(s + 1) * (s + 1) <= n) s++;
    if (s * s == n && in1(s)) { build1(s, o); o.push_back({s, '*', s, n}); return true; }
    return false;
}
static bool find3(u64 n, Ops *o) {
    Op r;
    for (u64 u : V) if (inv(n, u, in2, r)) {
        if (o) { u64 m = (r.a == u) ? r.b : r.a; if (!build2(m, *o)) return false; o->push_back(r); }
        return true;
    }
    for (const E1 &e : S1) if (inv(n, e.v, in1, r)) {
        if (o) { build1(r.a, *o); if (r.b != r.a) build1(r.b, *o); o->push_back(r); }
        return true;
    }
    return false;
}
static bool buildk(u64 n, Ops &o) { // n of cost <= 3
    int c = cost(n);
    if (c == 0) return true;
    if (c == 1) { build1(n, o); return true; }
    if (c == 2) return build2(n, o);
    return find3(n, &o);
}
static inline bool cheap(u64 n, int c, u64 cap3) { // cost(n) <= c ? (c capped at 3)
    int k = cost(n); if (k <= c) return true;
    if (c >= 3 && n <= cap3) return find3(n, nullptr);
    return false;
}
static const vector<u64> &setOf(int a) { return a == 0 ? V : a == 1 ? S1v : S2; }
static int HEAVY = 1;
static int ENUM = 0; static vector<Ops> SOLS;
// at depth 0 in enumeration mode: record the solution and keep searching (returns true when the cap is reached)
static inline bool done0(Ops &o, size_t s0, int depth) { if (!ENUM || depth > 0) return true; SOLS.push_back(o); o.resize(s0); return (int)SOLS.size() >= ENUM; }
static unordered_map<u64, int> failmemo; // n -> largest budget known to fail
static bool solve(u64 n, int c, Ops &o, int depth);
static inline int blen(u64 x) { return 64 - __builtin_clzll(x); }
static bool winr(u64 T, int a, int b, int cr, Ops &o, int depth) {
    const vector<u64> &A = setOf(a), &Bs = setOf(b);
    int bits = blen(T); int sh;
    if (cr <= 2) sh = bits / 2 + 7; else if (cr == 3) sh = bits / 2 + 4 + (a + b <= 1 ? 4 : 0);
    else sh = (a + b <= 2) ? (2 * bits / 3 + 2) : (bits / 2 + 4);
    u64 cap = 1ULL << min(62, sh); if (cap > T / 2) cap = T / 2;
    size_t cnt = 0;
    for (u64 X : A) {
        if (X < 2) continue;
        if (a == b && (u128)X * X > (u128)T + cap) break;
        u64 q = T / X; if (q < 2) break;
        u64 W = cap / X + 1; u64 lo = q > W ? q - W : 2; u64 hi = q + W; u64 mx = LIM / X; if (hi > mx) hi = mx;
        if (a == b && lo < X) lo = X;
        if (lo < 2) lo = 2;
        auto it = lower_bound(Bs.begin(), Bs.end(), lo);
        for (; it != Bs.end() && *it <= hi; ++it) {
            u64 Y = *it; u64 P = X * Y;
            if ((++cnt & 0xFFFF) == 0 && now() - T0 > TLIM) return false;
            if (P == T) { size_t s0 = o.size(); if (!buildk(X, o) || !buildk(Y, o)) { o.resize(s0); continue; } o.push_back({X, '*', Y, T}); if (done0(o, s0, depth)) return true; continue; }
            if (cr < 0) continue;
            u64 R = P > T ? P - T : T - P;
            if (R > cap) continue;
            size_t s0 = o.size();
            if (cr <= 2) { if (cost(R) > cr) continue; if (!buildk(R, o)) { o.resize(s0); continue; } }
            else if (!solve(R, cr, o, depth + 1)) { o.resize(s0); continue; }
            if (!buildk(X, o) || !buildk(Y, o)) { o.resize(s0); continue; }
            o.push_back({X, '*', Y, P});
            if (P > T) o.push_back({P, '-', R, T}); else o.push_back({P, '+', R, T});
            if (done0(o, s0, depth)) return true;
        }
    }
    return false;
}
static bool solve(u64 n, int c, Ops &o, int depth) {
    if (c < 0 || n == 0) return false;
    int k = cost(n);
    if (k <= c) return buildk(n, o);
    if (c < 3) return false;
    auto fm = failmemo.find(n);
    if (fm != failmemo.end() && fm->second >= c) return false;
    size_t s0 = o.size();
    if (find3(n, &o)) { if (done0(o, s0, depth)) return true; }
    o.resize(s0);
    if (c >= 4) {
        static const int AB[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
        int kmax = depth == 0 ? (HEAVY > 0 ? 6 : 5) : depth == 1 ? (HEAVY >= 3 ? 6 : 5) : (HEAVY >= 2 ? 5 : 4);
        for (int i = 0; i < kmax; i++) {
            int a = AB[i][0], b = AB[i][1];
            if (a + b + 1 > c) continue;
            if (winr(n, a, b, c - 2 - a - b, o, depth)) return true;
            o.resize(s0);
            if (now() - T0 > TLIM) return false;
        }
        if (depth == 0 || (depth == 1 && (c <= 5 || HEAVY >= 2))) {
            for (u64 u : V) {
                if (u == n || u < (n >> 12)) continue;
                if (now() - T0 > TLIM) return false;
                u64 m = u > n ? u - n : n - u;
                if (solve(m, c - 1, o, depth + 1)) { if (u > n) o.push_back({u, '-', m, n}); else o.push_back({u, '+', m, n}); if (done0(o, s0, depth)) return true; }
                o.resize(s0);
            }
            if (depth == 0 && HEAVY >= 1 && c >= 5) for (const E1 &e : S1) {
                u64 u = e.v; if (u == n || u < (n >> 8)) continue;
                if (now() - T0 > TLIM) return false;
                u64 m = u > n ? u - n : n - u;
                if (solve(m, c - 2, o, depth + 1)) { build1(u, o); if (u > n) o.push_back({u, '-', m, n}); else o.push_back({u, '+', m, n}); if (done0(o, s0, depth)) return true; }
                o.resize(s0);
            }
            for (u64 s : V) {
                if (s < 2 || s > (1ULL << 24) || n % s) continue;
                if (solve(n / s, c - 1, o, depth + 1)) { o.push_back({n / s, '*', s, n}); if (done0(o, s0, depth)) return true; }
                o.resize(s0);
            }
        }
    }
    if (now() - T0 <= TLIM) { int &f = failmemo[n]; if (f < c) f = c; }
    return false;
}

int main(int argc, char **argv) {
    u64 T; int cmin, cmax, heavy; size_t n;
    if (scanf("%llu %llu %d %d %d %zu", &LIM, &T, &cmin, &cmax, &heavy, &n) != 6) return 1;
    if (argc > 1) TLIM = atof(argv[1]);
    if (argc > 2) ENUM = atoi(argv[2]);
    V.resize(n); for (auto &x : V) if (scanf("%llu", &x) != 1) return 1;
    sort(V.begin(), V.end()); V.erase(unique(V.begin(), V.end()), V.end());
    T0 = now();
    // S1
    {
        vector<E1> t; n = V.size();
        for (size_t i = 0; i < n; i++) for (size_t j = i; j < n; j++) {
            u64 a = V[i], b = V[j]; // a <= b
            if (a + b <= LIM) t.push_back({a + b, a, b, '+'});
            if (b > a) t.push_back({b - a, b, a, '-'});
            if ((u128)a * b <= LIM) t.push_back({a * b, a, b, '*'});
            if (a > 1 && b % a == 0) t.push_back({b / a, b, a, '/'});
            if (a == b) t.push_back({1, a, a, '/'});
        }
        sort(t.begin(), t.end(), [](const E1 &x, const E1 &y) { return x.v < y.v; });
        for (auto &e : t) if (!inV(e.v) && (S1.empty() || S1.back().v != e.v)) S1.push_back(e);
        for (auto &e : S1) S1v.push_back(e.v);
        bl1.assign((1ULL << B1) / 64, 0); for (u64 x : S1v) { u64 h = h1(x); bl1[h >> 6] |= 1ULL << (h & 63); }
    }
    // S2
    {
        vector<u64> t; t.reserve(S1v.size() * V.size() * 3);
        for (u64 s : S1v) {
            for (u64 v : V) {
                if (s + v <= LIM) t.push_back(s + v);
                if (s > v) t.push_back(s - v); else if (v > s) t.push_back(v - s);
                if (v > 1 && (u128)s * v <= LIM) t.push_back(s * v);
                if (v > 1 && s % v == 0) t.push_back(s / v);
                if (s > 1 && v % s == 0) t.push_back(v / s);
            }
            if (2 * s <= LIM) t.push_back(2 * s);
            if ((u128)s * s <= LIM) t.push_back(s * s);
        }
        sort(t.begin(), t.end()); t.erase(unique(t.begin(), t.end()), t.end());
        S2.reserve(t.size());
        for (u64 x : t) if (!inV(x) && !binary_search(S1v.begin(), S1v.end(), x)) S2.push_back(x);
        bl2.assign((1ULL << B2) / 64, 0); for (u64 x : S2) { u64 h = h2(x); bl2[h >> 6] |= 1ULL << (h & 63); }
    }
    fprintf(stderr, "V %zu S1 %zu S2 %zu build %.1fs\n", V.size(), S1.size(), S2.size(), now() - T0);
    for (int c = cmin; c <= cmax; c++) {
        Ops o; double tb = now(); T0 = tb; failmemo.clear();
        HEAVY = heavy; bool ok = solve(T, c, o, 0);
        fprintf(stderr, "c=%d %s t=%.1fs\n", c, ok ? "FOUND" : "none", now() - tb);
        if (ENUM && !SOLS.empty()) {
            printf("ENUM %zu\n", SOLS.size());
            for (auto &sol : SOLS) { Ops r; set<u64> seen; for (auto &x : sol) if (!inV(x.v) && !seen.count(x.v)) { seen.insert(x.v); r.push_back(x); }
                printf("SOL %zu\n", r.size()); for (auto &x : r) printf("%llu %c %llu %llu\n", x.a, x.op, x.b, x.v); }
            return 0;
        }
        if (ok) {
            // dedupe by value, drop pool values
            Ops r; set<u64> seen;
            for (auto &x : o) if (!inV(x.v) && !seen.count(x.v)) { seen.insert(x.v); r.push_back(x); }
            printf("SOL %zu\n", r.size());
            for (auto &x : r) printf("%llu %c %llu %llu\n", x.a, x.op, x.b, x.v);
            return 0;
        }
    }
    printf("NONE\n");
    return 0;
}

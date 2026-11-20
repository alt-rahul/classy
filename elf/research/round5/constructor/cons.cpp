// cons.cpp - rolling-pool recipe constructor with cross-target lookahead.
// Recipes: T = x*y +- r (or exact x*y) with x, y, r chain values of the live pool (D_k = D_{k-1} op pool).
// stdin: LIMIT NSTART s.. NT t.. NINIT then NINIT lines "op a b v" (initial steps, values)
// args: cap kalt delta seed policy lookahead(0/1) nb rbits
// stdout: "S op a b v" per step (all steps incl. init); stderr: log
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64; typedef unsigned __int128 u128;
static u64 LIM;
struct E { u64 v, a, b; char op; unsigned char c; };
static double now() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }
struct Dict {
    vector<E> e; vector<u64> bl; int BB = 24; int levels = 0;
    inline u64 hh(u64 x) const { return (x * 0x9E3779B97F4A7C15ULL) >> (64 - BB); }
    const E* find(u64 v) const { auto it = lower_bound(e.begin(), e.end(), v, [](const E& x, u64 y) { return x.v < y; }); return (it != e.end() && it->v == v) ? &*it : nullptr; }
    inline int cost(u64 v) const { u64 h = hh(v); if (!((bl[h >> 6] >> (h & 63)) & 1)) return 99; const E* x = find(v); return x ? x->c : 99; }
};
static inline void comb(vector<E>& out, u64 a, u64 b, int c, u64 vmax) {
    if (a + b <= vmax) out.push_back({a + b, a, b, '+', (unsigned char)c});
    if (a > b) { if (a - b <= vmax) out.push_back({a - b, a, b, '-', (unsigned char)c}); } else if (b > a) { if (b - a <= vmax) out.push_back({b - a, b, a, '-', (unsigned char)c}); }
    if ((u128)a * b <= vmax) out.push_back({a * b, a, b, '*', (unsigned char)c});
    if (b > 1 && a % b == 0) out.push_back({a / b, a, b, '/', (unsigned char)c});
    if (a > 1 && b % a == 0) out.push_back({b / a, b, a, '/', (unsigned char)c});
}
static void buildDict(const vector<u64>& pool, Dict& d, int minLevel, int maxLevel, size_t maxRaw, u64 vmaxLast) {
    d.e.clear(); for (u64 v : pool) d.e.push_back({v, 0, 0, 0, 0});
    sort(d.e.begin(), d.e.end(), [](const E& a, const E& b) { return a.v < b.v; });
    d.e.erase(unique(d.e.begin(), d.e.end(), [](const E& a, const E& b) { return a.v == b.v; }), d.e.end());
    vector<u64> cur; for (auto& x : d.e) cur.push_back(x.v);
    vector<u64> P = cur; d.levels = 0;
    for (int k = 1; k <= maxLevel; k++) {
        if (k > minLevel && cur.size() * P.size() * 3 > maxRaw) break;
        u64 vmax = LIM; if (k == maxLevel && k > 2) vmax = min(LIM, vmaxLast);
        vector<E> raw; raw.reserve(cur.size() * P.size() * 3);
        if (k == 1) { for (size_t i = 0; i < P.size(); i++) for (size_t j = 0; j <= i; j++) comb(raw, P[i], P[j], 1, vmax); }
        else for (u64 v : cur) for (u64 p : P) comb(raw, v, p, k, vmax);
        sort(raw.begin(), raw.end(), [](const E& a, const E& b) { return a.v < b.v; });
        raw.erase(unique(raw.begin(), raw.end(), [](const E& a, const E& b) { return a.v == b.v; }), raw.end());
        vector<E> merged; merged.reserve(d.e.size() + raw.size()); cur.clear();
        size_t i = 0, j = 0;
        while (i < d.e.size() || j < raw.size()) {
            if (j == raw.size() || (i < d.e.size() && d.e[i].v <= raw[j].v)) { if (j < raw.size() && raw[j].v == d.e[i].v) j++; merged.push_back(d.e[i++]); }
            else { cur.push_back(raw[j].v); merged.push_back(raw[j++]); }
        }
        d.e.swap(merged); d.levels = k;
    }
    d.BB = d.e.size() > 4000000 ? 29 : d.e.size() > 400000 ? 27 : 24;
    d.bl.assign((size_t)1 << (d.BB - 6), 0); for (auto& x : d.e) { u64 h = d.hh(x.v); d.bl[h >> 6] |= 1ULL << (h & 63); }
}
struct Rec { int cost; u64 T, x, y, r; char sg; };
static int NB = 6; static u64 RMAX = 1ULL << 34;
// enumerate recipes of T with cost <= cmax. If out==nullptr only the min cost is tracked (cmax shrinks). returns min cost (99 none)
static int enumT(u64 T, const Dict& d, int cmax, vector<Rec>* out, int* cntAtMin = nullptr) {
    const vector<E>& e = d.e; size_t n = e.size(), j = n; int best = 99, cnt = 0;
    for (size_t i = 0; i < n; i++) {
        u64 x = e[i].v; if (x < 2) continue; if ((u128)x * x > (u128)T * 4) break;
        u64 q = T / x; while (j > 0 && e[j - 1].v >= q) --j;
        int cx = e[i].c; if (cx + 1 > cmax) continue;
        auto tryy = [&](const E& ey) {
            u64 y = ey.v; if (y < x) return; int cy = ey.c; if (cx + cy + 1 > cmax) return; u128 p = (u128)x * y; if (p > LIM) return;
            u64 pp = (u64)p; int c; u64 r = 0; char sg = '0';
            if (pp == T) c = cx + cy + 1;
            else { if (cx + cy + 2 > cmax) return; r = pp > T ? pp - T : T - pp; int cr = d.cost(r); c = cx + cy + cr + 2; if (c > cmax) return; sg = pp > T ? '-' : '+'; }
            if (out) out->push_back({c, T, x, y, r, sg});
            if (c < best) { best = c; cnt = 1; if (!out) cmax = c; } else if (c == best) cnt++;
        };
        int k = 0; for (size_t t = j; t > 0 && k < NB; k++) { --t; u64 y = e[t].v; if ((u128)(q - y) * x > RMAX) break; tryy(e[t]); }
        k = 0; for (size_t t = j; t < n && k < NB; t++, k++) { u64 y = e[t].v; if ((u128)(y - q) * x > RMAX) break; tryy(e[t]); }
    }
    if (cntAtMin) *cntAtMin = cnt;
    return best;
}
struct Step { char op; u64 a, b, v; };
static int CAP = 42, POLICY = 1; static u64 BIG, SMALL = 1 << 12; static set<u64> STARTS;
static void expandVal(const Dict& d, u64 v, vector<Step>& out, set<u64>& have) {
    if (have.count(v)) return; const E* x = d.find(v); if (!x) { fprintf(stderr, "expand miss %llu\n", v); exit(3); }
    if (x->c == 0) return;
    expandVal(d, x->a, out, have); expandVal(d, x->b, out, have);
    if (have.insert(v).second) out.push_back({x->op, x->a, x->b, v});
}
static vector<Step> expandRec(const Dict& d, const Rec& r) {
    vector<Step> out; set<u64> have;
    expandVal(d, r.x, out, have); expandVal(d, r.y, out, have); if (r.r) expandVal(d, r.r, out, have);
    u64 pp = r.x * r.y;
    if (r.sg == '0') { if (!have.count(pp)) out.push_back({'*', r.x, r.y, pp}); }
    else {
        if (!have.count(pp) && d.cost(pp) != 0) { out.push_back({'*', r.x, r.y, pp}); have.insert(pp); }
        if (r.sg == '+') out.push_back({'+', pp, r.r, r.T}); else out.push_back({'-', pp, r.r, r.T});
    }
    return out;
}
struct State { vector<u64> pool; map<u64, int> last; vector<u64> rem; vector<Step> steps; int clock = 0;
    // exact liveness: lu[v] = index of the last step using v (-1 none), born[v] = defining step (-1 start); occ[b] = values live at boundary b (before step b)
    unordered_map<u64, int> lu, born; vector<int> occ; };
static int REGCAP = 47, PCAP = 110, EXACT = 0;
// use value v at step index n (n == steps.size() at that moment): extends its live range. returns false if a boundary would exceed REGCAP
static bool useVal(State& s, u64 v, int n) {
    auto it = s.lu.find(v); if (it == s.lu.end()) return true;   // unknown (should not happen)
    int b0 = s.born[v]; int l = it->second; int from = max(l, b0) + 1;  // boundaries from..n get +1
    if ((int)s.occ.size() < n + 1) s.occ.resize(n + 1, 0);
    bool ok = true;
    for (int b = from; b <= n; b++) { if (++s.occ[b] > REGCAP) ok = false; }
    if (n > l) it->second = n;
    return ok;
}
static bool feasibleVal(const State& s, u64 v) {
    auto it = s.lu.find(v); if (it == s.lu.end()) return false;
    int n = s.steps.size(); int from = max(it->second, s.born.at(v)) + 1;
    for (int b = from; b <= n && b < (int)s.occ.size(); b++) if (s.occ[b] + 1 > REGCAP - 1) return false;
    return true;
}
static bool applyExact(State& s, const vector<Step>& ops, u64 T) {
    bool ok = true; set<u64> made;
    for (auto& o : ops) {
        int n = s.steps.size();
        if (!useVal(s, o.a, n)) ok = false; if (o.b != o.a && !useVal(s, o.b, n)) ok = false;
        s.steps.push_back(o); if (!s.lu.count(o.v)) { s.lu[o.v] = n; s.born[o.v] = n; }
        made.insert(o.v);
    }
    vector<u64> nr; for (u64 t : s.rem) if (t != T && !made.count(t)) nr.push_back(t); s.rem.swap(nr);
    return ok;
}
static void makePool(State& s) {
    vector<pair<int, u64>> c;
    for (auto& kv : s.lu) { u64 v = kv.first; if (v >= BIG && !STARTS.count(v)) continue; if (!feasibleVal(s, v)) continue; c.push_back({-max(kv.second, s.born[v]), v}); }
    sort(c.begin(), c.end()); if ((int)c.size() > PCAP) c.resize(PCAP);
    s.pool.clear(); for (auto& x : c) s.pool.push_back(x.second);
    for (u64 v : STARTS) if (find(s.pool.begin(), s.pool.end(), v) == s.pool.end() && feasibleVal(s, v)) s.pool.push_back(v);
}
static void applyRec(State& s, const vector<Step>& ops, u64 T) {
    s.clock++; set<u64> ps(s.pool.begin(), s.pool.end());
    for (auto& o : ops) {
        s.steps.push_back(o);
        if (s.last.count(o.a)) s.last[o.a] = s.clock; if (s.last.count(o.b)) s.last[o.b] = s.clock;
        if (o.v < BIG && !ps.count(o.v)) { s.pool.push_back(o.v); ps.insert(o.v); s.last[o.v] = s.clock; }
    }
    set<u64> made; for (auto& o : ops) made.insert(o.v);
    vector<u64> nr; for (u64 t : s.rem) if (t != T && !made.count(t)) nr.push_back(t); s.rem.swap(nr);
    while ((int)s.pool.size() > CAP) {
        int bi = -1; long bk = LONG_MAX;
        for (int i = 0; i < (int)s.pool.size(); i++) { u64 v = s.pool[i]; if (STARTS.count(v)) continue;
            long key = POLICY == 0 ? i : (long)s.last[v] * 1000 + i; if (v < SMALL) key += 3000;   // small values get a grace period
            if (key < bk) { bk = key; bi = i; } }
        s.last.erase(s.pool[bi]); s.pool.erase(s.pool.begin() + bi);
    }
}
int main(int argc, char** argv) {
    int kalt = 0, delta = 1, look = 0; u64 seed = 1; int rbits = 34; int cmaxBase = 8;
    if (argc > 1) CAP = atoi(argv[1]); if (argc > 2) kalt = atoi(argv[2]); if (argc > 3) delta = atoi(argv[3]); if (argc > 4) seed = atoll(argv[4]);
    if (argc > 5) POLICY = atoi(argv[5]); if (argc > 6) look = atoi(argv[6]); if (argc > 7) NB = atoi(argv[7]); if (argc > 8) rbits = atoi(argv[8]);
    if (argc > 9) EXACT = atoi(argv[9]); if (argc > 10) REGCAP = atoi(argv[10]); if (argc > 11) PCAP = atoi(argv[11]);
    RMAX = 1ULL << rbits; mt19937_64 rng(seed);
    size_t ns, nt, ni; if (scanf("%llu %zu", &LIM, &ns) != 2) return 1;
    State S; vector<u64> sv(ns); for (auto& x : sv) { scanf("%llu", &x); STARTS.insert(x); }
    scanf("%zu", &nt); vector<u64> T(nt); for (auto& x : T) scanf("%llu", &x);
    BIG = *min_element(T.begin(), T.end()) >> 3;
    scanf("%zu", &ni); set<u64> known(sv.begin(), sv.end());
    for (u64 v : sv) if (!S.last.count(v)) { S.pool.push_back(v); S.last[v] = 0; S.lu[v] = -1; S.born[v] = -1; }
    for (size_t i = 0; i < ni; i++) { char op[4]; u64 a, b, v; scanf("%s %llu %llu %llu", op, &a, &b, &v); S.steps.push_back({op[0], a, b, v}); known.insert(v);
        if (v < BIG && !S.last.count(v)) { S.pool.push_back(v); S.last[v] = 0; } }
    for (u64 t : T) if (!known.count(t)) S.rem.push_back(t);
    if (EXACT) { vector<Step> init = S.steps; S.steps.clear(); applyExact(S, init, 0); }
    else { vector<Step> none; State tmp = S; applyRec(S, none, 0); S.steps = tmp.steps; S.clock = 0; }
    double t0 = now(); int total = 0;
    auto evalState = [&](const State& s, Dict& d, vector<int>& mins) {
        int ml = s.pool.size() >= 30 ? 2 : s.pool.size() >= 14 ? 3 : s.pool.size() >= 8 ? 4 : 14;
        buildDict(s.pool, d, 2, ml, 6000000, BIG);
        mins.clear(); for (u64 t : s.rem) mins.push_back(enumT(t, d, cmaxBase, nullptr));
    };
    while (!S.rem.empty()) {
        if (EXACT) makePool(S);
        Dict d; int ml = S.pool.size() >= 30 ? 2 : S.pool.size() >= 14 ? 3 : S.pool.size() >= 8 ? 4 : 14;
        buildDict(S.pool, d, 2, ml, 6000000, BIG);
        vector<Rec> recs; int gmin = 99; int cm = 2 + 3 * d.levels; if (d.levels == 2) cm = cmaxBase;
        for (u64 t : S.rem) { int m = enumT(t, d, cm, nullptr); gmin = min(gmin, m); }
        if (gmin == 99 && d.levels == 2) { // fallback: level 3 dictionary
            buildDict(S.pool, d, 3, 3, 400000000, BIG << 3); cm = 11;
            for (u64 t : S.rem) { int m = enumT(t, d, cm, nullptr); gmin = min(gmin, m); cm = min(cm, gmin + delta); }
            fprintf(stderr, "  [fallback L3 size %zu gmin %d]\n", d.e.size(), gmin);
        }
        if (gmin == 99) { fprintf(stderr, "STUCK rem %zu pool %zu\n", S.rem.size(), S.pool.size()); return 2; }
        for (u64 t : S.rem) enumT(t, d, gmin + (look ? delta : 0), &recs);
        // candidates: all at gmin first (shuffled), then gmin+1.. shuffled
        shuffle(recs.begin(), recs.end(), rng);
        stable_sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) { return a.cost < b.cost; });
        int ncand = look ? min((int)recs.size(), max(1, kalt)) : 1;
        int bi = 0; double bs = 1e18; string binfo;
        if (look && S.rem.size() > 1) {
            for (int c = 0; c < ncand; c++) {
                vector<Step> ops = expandRec(d, recs[c]); State ch = S; applyRec(ch, ops, recs[c].T);
                Dict dc; vector<int> mins; evalState(ch, dc, mins);
                int m1 = 99, cnt = 0; double pot = 0; for (int m : mins) { m1 = min(m1, m); pot += max(0, 9 - min(m, 9)); }
                for (int m : mins) if (m == m1) cnt++;
                double sc = (double)ops.size() + min(m1, 9) - 0.02 * pot - 0.001 * cnt;
                if (sc < bs) { bs = sc; bi = c; char buf[100]; snprintf(buf, 100, "next %d x%d pot %.0f", m1, cnt, pot); binfo = buf; }
            }
        }
        vector<Step> ops;
        if (EXACT) {
            // among the cheapest feasible recipes take the one with the smallest register-time extension
            long bestKey = LONG_MAX; State bestS; bool found = false;
            for (int extra = 0; extra <= 4 && !found; extra++) {
                if (extra > 0) { recs.clear(); for (u64 t : S.rem) enumT(t, d, gmin + extra, &recs);
                    vector<Rec> f; for (auto& r : recs) if (r.cost == gmin + extra) f.push_back(r); recs.swap(f); shuffle(recs.begin(), recs.end(), rng); }
                int tried = 0;
                for (int c = 0; c < (int)recs.size() && tried < 600; c++) {
                    if (found && recs[c].cost > recs[bi].cost) break;
                    tried++; vector<Step> o = expandRec(d, recs[c]); State ch = S; long before = 0; for (int x : S.occ) before += x;
                    if (!applyExact(ch, o, recs[c].T)) continue;
                    long after = 0; for (int x : ch.occ) after += x; long key = (long)o.size() * 100000 + (after - before);
                    if (key < bestKey) { bestKey = key; bestS = ch; ops = o; bi = c; found = true; }
                }
            }
            if (!found) { fprintf(stderr, "NOFEASIBLE rem %zu\n", S.rem.size()); return 4; }
            S = bestS;
        } else { ops = expandRec(d, recs[bi]); applyRec(S, ops, recs[bi].T); }
        total += ops.size();
        fprintf(stderr, "step cost %zu (gmin %d, cands %zu/%d) pool %zu rem %zu total %d %s t %.0f\n", ops.size(), gmin, recs.size(), ncand, S.pool.size(), S.rem.size(), total, binfo.c_str(), now() - t0);
    }
    for (auto& o : S.steps) printf("S %c %llu %llu %llu\n", o.op, o.a, o.b, o.v);
    return 0;
}

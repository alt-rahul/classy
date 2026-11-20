// sched.cpp - register-pressure scheduler: simulated annealing with segment moves on a topological order.
// stdin: NS N NREG TL SEED RESTARTS ; then N lines "a b" (operand ids: 0..NS-1 = start values, NS+k = k-th node of the
// initial order). stdout: "OK" + the permutation (node indices in new order) or "FAIL best_excess".
#include <bits/stdc++.h>
using namespace std;
int NS, N, NREG; vector<int> A, B; vector<vector<int>> users;
static double now() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }
static vector<int> lastv, die;
static long energy(const vector<int> &od) {
    // od: order of node ids (0..N-1); value id of node k = NS+k
    fill(lastv.begin(), lastv.end(), -1);
    for (int i = 0; i < N; i++) { int v = od[i]; lastv[A[v]] = i; lastv[B[v]] = i; }
    fill(die.begin(), die.end(), 0);
    int live = 0;
    for (int s = 0; s < NS; s++) if (lastv[s] >= 0) { live++; die[lastv[s]]++; }
    for (int k = 0; k < N; k++) if (lastv[NS + k] >= 0) die[lastv[NS + k]]++;
    long e = 0;
    for (int i = 0; i < N; i++) {
        live -= die[i]; int x = live + 1;
        if (x > NREG - 2) e += (x - NREG + 2) + (x > NREG ? 100L * (x - NREG) : 0);
        if (lastv[NS + od[i]] >= 0) live++;
    }
    return e;
}
int main() {
    double TL; unsigned seed; int restarts;
    if (scanf("%d %d %d %lf %u %d", &NS, &N, &NREG, &TL, &seed, &restarts) != 6) return 1;
    A.resize(N); B.resize(N); users.assign(N, {});
    for (int k = 0; k < N; k++) { if (scanf("%d %d", &A[k], &B[k]) != 2) return 1; }
    for (int k = 0; k < N; k++) { if (A[k] >= NS) users[A[k] - NS].push_back(k); if (B[k] >= NS && B[k] != A[k]) users[B[k] - NS].push_back(k); }
    lastv.assign(NS + N, -1); die.assign(N + 1, 0);
    vector<int> init(N); iota(init.begin(), init.end(), 0);
    mt19937_64 rng(seed); double t00 = now(); long bestAll = LONG_MAX; vector<int> bestOd;
    static const double T0s[] = {1000, 400, 2000, 700, 1500, 250};
    for (int rs = 0; rs < restarts && bestAll >= 100; rs++) {
        vector<int> od = init, pos(N), rest, od2(N), prs(N);
        long E = energy(od); long best = E; vector<int> bo = od;
        double T0 = T0s[rs % 6], tl = TL / restarts, t0 = now(); double Tm = T0; long it = 0;
        while (best >= 100) {
            if ((it++ & 127) == 0) { double el = now() - t0; if (el > tl) break; Tm = T0 * (1 - el / tl) + 1.0; }
            int L = (rng() % 10 < 3) ? 1 : 2 + rng() % 13; if (L > N) L = N;
            int i = rng() % (N - L + 1);
            // rest = od without [i, i+L)
            rest.clear(); for (int k = 0; k < i; k++) rest.push_back(od[k]); for (int k = i + L; k < N; k++) rest.push_back(od[k]);
            fill(prs.begin(), prs.end(), -1); for (int k = 0; k < (int)rest.size(); k++) prs[rest[k]] = k;
            int lo = 0, hi = rest.size();
            for (int k = i; k < i + L; k++) { int v = od[k];
                if (A[v] >= NS && prs[A[v] - NS] >= 0) lo = max(lo, prs[A[v] - NS] + 1);
                if (B[v] >= NS && prs[B[v] - NS] >= 0) lo = max(lo, prs[B[v] - NS] + 1);
                for (int w : users[v]) if (prs[w] >= 0) hi = min(hi, prs[w]); }
            if (lo > hi || (lo == hi && lo == i)) continue;
            int j = lo + rng() % (hi - lo + 1); if (j == i) continue;
            int q = 0; for (int k = 0; k < j; k++) od2[q++] = rest[k]; for (int k = i; k < i + L; k++) od2[q++] = od[k]; for (int k = j; k < (int)rest.size(); k++) od2[q++] = rest[k];
            long E2 = energy(od2);
            if (E2 <= E || (double)(rng() % 1000000) / 1e6 < exp((double)(E - E2) / Tm)) { od.swap(od2); E = E2; if (E < best) { best = E; bo = od; } }
        }
        if (best < bestAll) { bestAll = best; bestOd = bo; }
    }
    if (bestAll < 100) { printf("OK\n"); for (int v : bestOd) printf("%d ", v); printf("\n"); }
    else printf("FAIL %ld %.1f\n", bestAll, now() - t00);
    return 0;
}

// xyr.cpp - recipe enumeration over a pool. Input: LIMIT CMAX NB RBITS \n N pool.. \n M targets..
// D1 = pool op pool, D2 = D1 op pool. Recipes: T = x*y +- r (x,y,r in D<=2, r may be 0), T = u +- r.
// Output: "R T cost x y r sgn"  (sgn: + means T=x*y+r, - means T=x*y-r, 0 exact; y==0 means additive T=x+r / x-r )
//         "D v cost a op b" for every non-pool value used.
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64; typedef unsigned __int128 u128;
static u64 LIM; struct E { u64 v, a, b; char op; unsigned char c; };
static vector<E> D; // sorted by v, unique
static vector<u64> V;
static void comb(vector<E>&out, u64 a, u64 b, int c){
    if (a + b <= LIM) out.push_back({a+b,a,b,'+',(unsigned char)c});
    if (a > b) out.push_back({a-b,a,b,'-',(unsigned char)c}); else if (b > a) out.push_back({b-a,b,a,'-',(unsigned char)c});
    if ((u128)a*b <= LIM) out.push_back({a*b,a,b,'*',(unsigned char)c});
    if (b>1 && a % b == 0) out.push_back({a/b,a,b,'/',(unsigned char)c});
    if (a>1 && b % a == 0) out.push_back({b/a,b,a,'/',(unsigned char)c});
}
static inline const E* find(u64 v){ auto it = lower_bound(D.begin(), D.end(), v, [](const E&e, u64 x){return e.v<x;}); if (it!=D.end() && it->v==v) return &*it; return nullptr; }
int main(){
    int CMAX, NB, RBITS; size_t n, m;
    if (scanf("%llu %d %d %d %zu", &LIM, &CMAX, &NB, &RBITS, &n)!=5) return 1;
    V.resize(n); for (auto&x: V) scanf("%llu", &x);
    scanf("%zu", &m); vector<u64> Ts(m); for (auto&x: Ts) scanf("%llu", &x);
    sort(V.begin(), V.end()); V.erase(unique(V.begin(), V.end()), V.end());
    vector<E> all; for (u64 v: V) all.push_back({v,0,0,0,0});
    vector<E> d1; for (size_t i=0;i<V.size();i++) for (size_t j=0;j<=i;j++) comb(d1, V[i], V[j], 1);
    auto norm = [](vector<E>&x){ stable_sort(x.begin(), x.end(), [](const E&a,const E&b){ return a.v<b.v || (a.v==b.v && a.c<b.c);}); x.erase(unique(x.begin(), x.end(), [](const E&a,const E&b){return a.v==b.v;}), x.end()); };
    all.insert(all.end(), d1.begin(), d1.end()); norm(all);
    vector<E> d2; for (auto&e: all) if (e.c==1) for (u64 v: V) comb(d2, e.v, v, 2);
    all.insert(all.end(), d2.begin(), d2.end()); norm(all); d2.clear(); d2.shrink_to_fit();
    D.swap(all);
    size_t c0=0,c1=0,c2=0; for (auto&e: D) (e.c==0?c0:e.c==1?c1:c2)++;
    fprintf(stderr, "D0 %zu D1 %zu D2 %zu\n", c0, c1, c2);
    // bloom
    const int BB=28; vector<u64> bl(1u<<(BB-6)); auto hh=[&](u64 x){ return (x*0x9E3779B97F4A7C15ULL)>>(64-BB); };
    for (auto&e: D){ u64 h=hh(e.v); bl[h>>6] |= 1ULL<<(h&63); }
    auto cost=[&](u64 v)->int{ u64 h=hh(v); if(!((bl[h>>6]>>(h&63))&1)) return 9; const E*e=find(v); return e? e->c : 9; };
    set<u64> used; u64 RMAX = 1ULL<<RBITS;
    auto use=[&](u64 v){ // mark v and parents
        while (true){ const E*e=find(v); if(!e||e->c==0||used.count(v)) return; used.insert(v); // parent of D2 is D1 value (a or b)
            const E*ea=find(e->a); u64 nxt = (ea && ea->c>0)? e->a : e->b; const E*en=find(nxt); if(!en||en->c==0) return; v=nxt; } };
    for (u64 T: Ts){
        long cnt=0;
        for (size_t i=0;i<D.size();i++){
            const E&ex=D[i]; u64 x=ex.v; if (x<2) continue; if ((u128)x*x > (u128)T*4 ) break; // x <= ~y (allow slack)
            int cx=ex.c; if (cx+2 > CMAX) continue;
            u64 q=T/x;
            auto it=lower_bound(D.begin(), D.end(), q, [](const E&e, u64 v){return e.v<v;});
            // go down
            auto tryy=[&](const E&ey){ u64 y=ey.v; if (y<x) return; int cy=ey.c; if (cx+cy+1>CMAX) return; u128 p=(u128)x*y; if (p>LIM) return;
                u64 pp=(u64)p; if (pp==T){ printf("R %llu %d %llu %llu 0 0\n", T, cx+cy+1, x, y); use(x); use(y); cnt++; return; }
                if (cx+cy+2>CMAX) return;
                u64 r = pp>T? pp-T : T-pp; int cr=cost(r); if (cx+cy+cr+2<=CMAX){ printf("R %llu %d %llu %llu %llu %c\n", T, cx+cy+cr+2, x, y, r, pp>T?'-':'+'); use(x); use(y); use(r); cnt++; } };
            int k=0; for (auto j=it; j!=D.begin() && k<NB; ){ --j; u64 y=j->v; if ((u128)(q-y)*x > RMAX) break; tryy(*j); k++; }
            k=0; for (auto j=it; j!=D.end() && k<NB; ++j){ u64 y=j->v; if ((u128)(y-q)*x > RMAX) break; tryy(*j); k++; }
        }
        // additive: T = u + r, T = u - r
        for (auto&eu: D){ u64 u=eu.v; int cu=eu.c; if (cu+1>CMAX) continue;
            if (u<T){ u64 r=T-u; if (r<u) continue; int cr=cost(r); if (cu+cr+1<=CMAX){ printf("R %llu %d %llu 0 %llu +\n", T, cu+cr+1, u, r); use(u); use(r); cnt++; } }
            else if (u>T){ u64 r=u-T; int cr=cost(r); if (cu+cr+1<=CMAX){ printf("R %llu %d %llu 0 %llu -\n", T, cu+cr+1, u, r); use(u); use(r); cnt++; } }
        }
        fprintf(stderr, "T %llu recipes %ld\n", T, cnt);
    }
    for (u64 v: used){ const E*e=find(v); printf("D %llu %d %llu %c %llu\n", v, e->c, e->a, e->op, e->b); }
}

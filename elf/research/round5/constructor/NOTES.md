# Round 5 constructor research - running log

## Framing (before measuring)
- Tree-over-pool counting: a k-op tree has k+1 leaves; with N=48 live leaves that is about 5.6 bits per leaf + 1-2 bits per op,
  i.e. 6.5-7.5 bits/op. Realised public: 7.0 bits/op at 40/50/60 bits. So the store sits at the TREE bound for a 48-pool;
  only two things can beat it: (a) less bootstrap (intercept 28-33 ops/puzzle), (b) nodes shared by several targets
  chosen among alternative recipes (floor+1 / floor+2 recipes are 100x / 10^4x more numerous than floor recipes).
- H4 (targets uniform random): already tested on scored puzzles in round 3 (generator_meta/NOTES.md H2 refuted, H3 confirmed;
  peel hit fraction 0.0% for independent). Not repeated.

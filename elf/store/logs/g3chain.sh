#!/bin/bash
# after g2 finishes: third config on the others (alpha 0.07, evict_size, pair_eps 0.005), lowest ops/target first,
# plus the beam pass on the small other puzzles (sm2, resumable)
cd /home/user/classy/elf
while pgrep -f "run.py --tag g2 " > /dev/null; do sleep 20; done
GRID='[{"beam":{"width":16,"branch":6,"gamma":0.4}},{"beam":{"width":16,"branch":6,"gamma":0.4,"alpha":0.05}},{"beam":{"width":12,"branch":5,"gamma":0.6,"alpha":0.15}}]'
python3 src/restarts.py --tag sm2 --classes 'other<20' --base '{"pool_cap":46,"evict_big":4}' --grid "$GRID" --budget 60 --repair --workers 1 >> store/logs/sm2.log 2>&1 &
python3 src/run.py --tag g3 --cfg '{"pool_cap":46,"evict_size":1,"regret":0.07,"pair_eps":0.005}' --classes 'other>=20' --order optasc --repair --workers 3 >> store/logs/g3.log 2>&1
wait

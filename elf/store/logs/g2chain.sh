#!/bin/bash
# g2 phase: regret 0.15 + evict_size on other puzzles; first the ones done by g1 with evict_big 4, then the rest
cd /home/user/classy/elf
python3 src/run.py --tag g2 --cfg '{"pool_cap":46,"evict_size":1,"regret":0.15}' --ids "$(cat store/logs/g2_ids.txt)" --order optasc --repair --workers 4 >> store/logs/g2.log 2>&1
python3 src/run.py --tag g2 --cfg '{"pool_cap":46,"evict_size":1,"regret":0.15}' --ids "$(cat store/logs/g2b_ids.txt)" --order optasc --repair --workers 4 >> store/logs/g2.log 2>&1

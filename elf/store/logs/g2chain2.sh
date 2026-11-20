#!/bin/bash
# wait for the running g2 (g2_ids) to finish, then run g2 on the rest with pair_eps 0.005 (1.3-1.8x faster on 60-bit mixed)
cd /home/user/classy/elf
while pgrep -f "run.py --tag g2 " > /dev/null; do sleep 20; done
python3 src/run.py --tag g2 --cfg '{"pool_cap":46,"evict_size":1,"regret":0.15,"pair_eps":0.005}' --ids "$(cat store/logs/g2b_ids.txt)" --order optasc --repair --workers 4 >> store/logs/g2.log 2>&1

#!/bin/bash
# After the main run (s7): checkpoint, independent restarts (r1), other restarts (r2), checkpoints.
cd /home/user/classy/elf
while pgrep -f "run.py --tag s7 " > /dev/null; do sleep 15; done
src/checkpoint.sh >> store/logs/chain2.log 2>&1
IDS=$(python3 src/select_ids.py --classes 'indep>=20,indep<20' --not-tags s7,xesi)
echo "r1 $(echo $IDS | tr ',' '\n' | wc -l) puzzles $(date)" >> store/logs/chain2.log
python3 src/run.py --tag r1 --workers 4 --repair --order len --ids "$IDS" --cfg '{"pool_cap":46,"evict_size":1}' > store/logs/r1.log 2>&1
src/checkpoint.sh >> store/logs/chain2.log 2>&1
IDS=$(python3 src/select_ids.py --classes 'other>=20,other<20' --tags s2a,s3,s4,s5)
echo "r2 $(echo $IDS | tr ',' '\n' | wc -l) puzzles $(date)" >> store/logs/chain2.log
python3 src/run.py --tag r2 --workers 4 --repair --order len --ids "$IDS" --cfg '{"pool_cap":46,"evict_big":4}' > store/logs/r2.log 2>&1
src/checkpoint.sh >> store/logs/chain2.log 2>&1
echo "chain2 done $(date)" >> store/logs/chain2.log

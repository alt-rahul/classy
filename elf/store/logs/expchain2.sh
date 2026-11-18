#!/bin/bash
cd /home/user/classy/elf
I=scored_0001,scored_0058,scored_0123,scored_0230,scored_0306
python3 src/exp.py $I '{"pool_cap":46,"evict_big":8}' xeb8 > store/logs/exp_eb8.log 2>&1
python3 src/exp.py $I '{"pool_cap":46,"evict_big":4,"win_k":3}' xeb4w3 > store/logs/exp_eb4w3.log 2>&1
python3 src/exp.py $I '{"pool_cap":46,"evict_big":1}' xeb1 > store/logs/exp_eb1.log 2>&1

#!/bin/bash
cd /home/user/classy/elf
I=scored_0001,scored_0058,scored_0123,scored_0230,scored_0306
python3 src/exp.py $I '{"pool_cap":46,"evict_big":14}' xeb14 > store/logs/exp_eb14.log 2>&1
python3 src/exp.py $I '{"pool_cap":46,"evict_big":20}' xeb20 > store/logs/exp_eb20.log 2>&1
python3 src/exp.py $I '{"pool_cap":46,"evict_unused":1}' xeui > store/logs/exp_eu_i.log 2>&1

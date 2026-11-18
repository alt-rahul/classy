#!/bin/bash
cd /home/user/classy/elf
while pgrep -f "expchain1.sh" > /dev/null; do sleep 5; done
O=scored_0003,scored_0079,scored_0148,scored_0200,scored_0266,scored_0335
python3 src/exp.py $O '{"pool_cap":46,"evict_unused":1}' xeu > store/logs/exp_eu_o.log 2>&1
python3 src/exp.py $O '{"pool_cap":46,"evict_big":4}' xeb4o > store/logs/exp_eb4_o.log 2>&1

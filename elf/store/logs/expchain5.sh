#!/bin/bash
cd /home/user/classy/elf
I=scored_0001,scored_0058,scored_0123,scored_0230,scored_0306
O=scored_0003,scored_0079,scored_0148,scored_0200,scored_0266,scored_0335
python3 src/exp.py $O '{"pool_cap":46,"evict_big":10}' xeb10o > store/logs/exp_eb10_o.log 2>&1 &
python3 src/exp.py $I '{"pool_cap":46,"evict_big":28}' xeb28 > store/logs/exp_eb28.log 2>&1
wait
python3 src/exp.py $O '{"pool_cap":46,"evict_size":1}' xeso > store/logs/exp_es_o.log 2>&1 &
python3 src/exp.py $I '{"pool_cap":46,"evict_size":1}' xesi > store/logs/exp_es_i.log 2>&1
wait

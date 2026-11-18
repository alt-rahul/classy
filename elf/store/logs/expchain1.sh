#!/bin/bash
cd /home/user/classy/elf
while pgrep -f "exp.py" > /dev/null; do sleep 5; done
I=scored_0001,scored_0058,scored_0123,scored_0230,scored_0306
python3 src/exp.py $I '{"pool_cap":46,"win_k":3}' xw3 > store/logs/exp_w3.log 2>&1
python3 src/exp.py $I '{"pool_cap":46,"win_budget":4e6}' xb4 > store/logs/exp_b4.log 2>&1
O=scored_0003,scored_0079,scored_0148,scored_0200,scored_0266,scored_0335
python3 src/exp.py $O '{"pool_cap":46,"lookahead":4}' xla4 > store/logs/exp_la4.log 2>&1
python3 src/exp.py $O '{"pool_cap":46,"noise":2.0,"seed":3}' xn2 > store/logs/exp_n2.log 2>&1

#!/bin/bash
cd /home/user/classy/elf
while pgrep -f "run.py --tag s3 " > /dev/null; do sleep 20; done
python3 src/run.py --tag s3 --workers 4 --cfg '{"pool_cap":46}' > store/logs/s3b.log 2>&1
python3 src/repair_store.py --workers 4 > store/logs/rep1.log 2>&1

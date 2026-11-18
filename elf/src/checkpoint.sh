#!/bin/bash
# Write the submission from the store to a temp file, validate it independently, then rename.
cd /home/user/classy/elf
T=submissions/.sub_tmp.csv
python3 src/write_sub.py $T > store/logs/checkpoint_write.log 2>&1 || { echo "write failed"; cat store/logs/checkpoint_write.log; exit 1; }
OUT=$(python3 src/validate.py data/puzzles.csv $T)
echo "$OUT"
if echo "$OUT" | grep -q "invalid 0"; then mv $T submissions/submission.csv; echo "checkpoint ok $(date)"; else echo "VALIDATION FAILED, kept old submission"; rm -f $T; exit 1; fi

#!/bin/bash
# Background loop: every 5 minutes, if the store beats our best Kaggle score and >= 50 minutes have passed since
# our last submission (and none is pending), write + validate the submission and submit it.
cd /home/user/classy/elf
LOG=store/logs/submit_daemon.log
while true; do
  TOTAL=$(python3 -c "import json;print(sum(v['len'] for v in json.load(open('store/best.json')).values()))" 2>/dev/null)
  BEST=$(curl -s --max-time 60 "https://www.kaggle.com/api/v1/competitions/submissions/list/santa-2026" | python3 -I -c "
import json,sys
d=json.load(sys.stdin); sc=[int(s['publicScore']) for s in d if s.get('publicScore')]; print(min(sc) if sc else 10**9)" 2>/dev/null)
  if [ -n "$TOTAL" ] && [ -n "$BEST" ] && [ "$TOTAL" -lt "$BEST" ]; then
    echo "$(date -u +%FT%TZ) store $TOTAL < kaggle best $BEST" >> $LOG
    if bash src/checkpoint.sh >> $LOG 2>&1; then GAP=50 bash src/kaggle_submit.sh "auto" >> $LOG 2>&1; fi
  fi
  sleep 300
done

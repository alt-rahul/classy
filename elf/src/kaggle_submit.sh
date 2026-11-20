#!/bin/bash
# Submit submissions/submission.csv to Kaggle santa-2026 if it validates, beats our best Kaggle score,
# and at least $GAP minutes (default 50) have passed since our last submission. Auth is injected by the session proxy.
# Usage: bash src/kaggle_submit.sh "description"   (add FORCE=1 to skip the score check)
cd /home/user/classy/elf
F=submissions/submission.csv
LOG=store/logs/kaggle_submissions.log
API=https://www.kaggle.com/api/v1
TMP=$(mktemp -d)
OUT=$(python3 src/validate.py data/puzzles.csv $F)
echo "$OUT"
echo "$OUT" | grep -q "invalid 0" || { echo "NOT VALID, not submitting"; exit 1; }
LOCAL=$(echo "$OUT" | awk '{print $4}')
curl -s --max-time 60 "$API/competitions/submissions/list/santa-2026" -o $TMP/subs.json || { echo "list failed"; exit 1; }
read BEST AGE PENDING < <(python3 -I -c "
import json,datetime
d=json.load(open('$TMP/subs.json'))
sc=[int(s['publicScore']) for s in d if s.get('publicScore')]
now=datetime.datetime.now(datetime.timezone.utc)
ages=[(now-datetime.datetime.fromisoformat(s['date'].replace('Z','+00:00'))).total_seconds()/60 for s in d]
pend=sum(1 for s in d if s.get('status')=='pending')
print(min(sc) if sc else 10**9, int(min(ages)) if ages else 10**6, pend)")
echo "local $LOCAL, best kaggle $BEST, minutes since last submission $AGE, pending $PENDING"
GAP=${GAP:-50}
if [ "$AGE" -lt "$GAP" ]; then echo "last submission was $AGE min ago (<$GAP), not submitting"; exit 0; fi
if [ "$PENDING" -gt 0 ]; then echo "a submission is still pending, not submitting"; exit 0; fi
if [ -z "$FORCE" ] && [ "$LOCAL" -ge "$BEST" ]; then echo "not better than $BEST, not submitting"; exit 0; fi
SZ=$(stat -c %s $F); MT=$(stat -c %Y $F)
cp $F $TMP/submission.csv
curl -s --max-time 60 -X POST --form-string "fileName=submission.csv" "$API/competitions/santa-2026/submissions/url/$SZ/$MT" -o $TMP/up.json
URL=$(python3 -I -c "import json;print(json.load(open('$TMP/up.json'))['createUrl'])") || { echo "no upload url"; cat $TMP/up.json; exit 1; }
TOK=$(python3 -I -c "import json;print(json.load(open('$TMP/up.json'))['token'])")
curl -s --max-time 300 -X PUT -H "Content-Type: application/octet-stream" --data-binary @$TMP/submission.csv "$URL" -o /dev/null -w "upload %{http_code}\n"
RES=$(curl -s --max-time 60 -X POST --form-string "blobFileTokens=$TOK" --form-string "submissionDescription=${1:-auto} local $LOCAL" "$API/competitions/submissions/submit/santa-2026")
echo "$RES"
echo "$(date -u +%FT%TZ) local=$LOCAL prev_best=$BEST result=$RES" >> $LOG
rm -rf $TMP

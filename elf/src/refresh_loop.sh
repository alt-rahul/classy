#!/bin/bash
# Hourly: merge newer openly licensed public programs into the store (submit_daemon.sh submits when the store beats Kaggle).
cd /home/user/classy/elf
while true; do
  echo "$(date -u +%FT%TZ) refresh" >> store/logs/refresh_loop.log
  nice -n 10 timeout 1800 python3 src/refresh_public.py >> store/logs/refresh_loop.log 2>&1
  sleep 3600
done

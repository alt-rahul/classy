#!/bin/bash
cd /home/user/classy/elf/research/round5/structured
SX_TH=100000 SX_K2=0 SX_TAG=sx-th nice -n 10 python3 sx.py 40 $(cat ids_ml20.txt) > sx_ml20.log 2>&1
SX_K2=0 SX_CAP=400 SX_TAG=sx-full nice -n 10 python3 sx.py 600 scored_0855 scored_0749 scored_0617 > sx_full.log 2>&1

#!/bin/bash
# usage: bench.sh '<cfg json>' [ids]
IDS=${2:-practice_0005,practice_0008,practice_0013,practice_0002,practice_0003,practice_0012,practice_0004,practice_0006,practice_0015,practice_0001,practice_0007,practice_0009}
timeout 3000 python3 /home/user/classy/elf/src/test_practice.py $IDS "$1"

#!/bin/bash
# Download public program datasets via the Kaggle API into $1 (default: /tmp/elf_public). Data only; nothing is executed.
D=${1:-/tmp/elf_public}; mkdir -p $D
for ref in goodpjw2008/santa-2026-dag-lns-programs asteriospanoras/santa-2026-elf-programs-and-experiments alexchilton/santa-2026-rudolph-programs; do
  d=$D/$(echo $ref | tr '/' '_'); mkdir -p $d
  curl -s -L --max-time 300 -o $d/data.zip "https://www.kaggle.com/api/v1/datasets/download/$ref" && (cd $d && unzip -q -o data.zip && rm data.zip)
  echo "$ref -> $d"
done

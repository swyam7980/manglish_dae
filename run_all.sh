#!/usr/bin/env bash
# Full pipeline. Usage: bash run_all.sh [small|base]   (size for mt5/byt5; default base)
set -e
SIZE=${1:-base}
cd "$(dirname "$0")/src"
python data_prep.py
python check_lengths.py
python evaluate.py --baseline
for M in mbart mt5 byt5; do
  python train.py --model $M --size $SIZE
  python evaluate.py --model $M --size $SIZE
done
python compare.py

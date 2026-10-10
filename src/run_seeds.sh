#!/bin/sh
# Extra training seeds for the top backbones. Split and validation noise stay fixed; only the training seed changes.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for s in 43 44; do
  $PY train.py --model mbart --seed $s --tag-suffix _s$s && $PY evaluate.py --model mbart --tag-suffix _s$s
  $PY train.py --model byt5 --seed $s --tag-suffix _s$s && $PY evaluate.py --model byt5 --tag-suffix _s$s
  $PY train.py --model mt5 --lr 1e-3 --seed $s --tag-suffix _lr1e-3_s$s && $PY evaluate.py --model mt5 --tag-suffix _lr1e-3_s$s
done

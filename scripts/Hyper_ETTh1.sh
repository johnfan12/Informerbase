#!/bin/bash
set -euo pipefail

python -u main_informer.py \
  --model hyperinformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --attn prob \
  --factor 3 \
  --hyper_z_dim 128 \
  --hyper_hidden_dim 256 \
  --hyper_pool mean \
  --des 'Hyper' \
  --itr 1

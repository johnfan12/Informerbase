#!/bin/bash

# ETTh1 数据集后处理训练示例脚本
# 使用前请确保已经有训练好的模型checkpoint

# 设置参数
model_name=informer
data=ETTh1
seq_len=96
label_len=48
pred_len=24

# 预训练模型的checkpoint路径 (请根据实际情况修改)
# 格式示例: ./checkpoints/informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/checkpoint.pth
checkpoint_path="./checkpoints/your_pretrained_model_checkpoint_path/checkpoint.pth"

# 后处理训练参数
post_train_epochs=6
learning_rate=0.00001  # 通常使用更小的学习率进行后处理训练
batch_size=32
patience=3

echo "Starting Post-Training Process for $data"
echo "Using checkpoint: $checkpoint_path"

python -u main_post_training.py \
  --model $model_name \
  --data $data \
  --root_path ./data/ETT/ \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 2048 \
  --factor 5 \
  --attn prob \
  --embed timeF \
  --freq h \
  --distil \
  --mix \
  --train_epochs $post_train_epochs \
  --batch_size $batch_size \
  --patience $patience \
  --learning_rate $learning_rate \
  --des post_train \
  --checkpoint_path $checkpoint_path \
  --itr 1

echo "Post-Training Completed!"

# 后处理训练模块使用说明

## 模块概述

后处理训练模块 (`exp_post_training.py`) 是一个用于对已训练的Informer模型进行后处理优化的工具。

## 工作原理

该模块采用以下三步训练策略：

### 第一阶段：推理阶段
1. 加载预训练的模型参数
2. 对整个训练数据集进行推理，获取所有预测序列
3. 保存预测结果和真实值

### 第二阶段：计算平均值
1. 计算预测序列与真实序列的平均值：`mean = (pred + true) / 2`
2. 这个平均值作为目标参考

### 第三阶段：后处理训练
1. 使用预测序列与平均值的差值作为训练目标
2. 损失函数：`loss = MSE(pred - mean, 0)`
3. 通过最小化这个损失，使模型预测更接近预测值和真实值的平均水平
4. 遍历整个训练数据集进行优化

## 使用方法

### 1. 基本使用

```bash
python main_post_training.py \
  --model informer \
  --data ETTh1 \
  --checkpoint_path ./checkpoints/your_model/checkpoint.pth \
  --train_epochs 6 \
  --learning_rate 0.00001
```

### 2. 使用脚本

```bash
# 修改脚本中的checkpoint_path为你的模型路径
bash scripts/post_training_ETTh1.sh
```

### 3. 完整参数示例

```bash
python main_post_training.py \
  --model informer \
  --data ETTh1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
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
  --train_epochs 6 \
  --batch_size 32 \
  --patience 3 \
  --learning_rate 0.00001 \
  --checkpoint_path ./checkpoints/your_pretrained_model/checkpoint.pth \
  --des post_train
```

## 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint_path` | **必需**，预训练模型checkpoint路径 | - |
| `--model` | 模型类型 (informer/informerstack) | informer |
| `--data` | 数据集名称 (ETTh1/ETTh2/ETTm1/ETTm2) | ETTh1 |
| `--train_epochs` | 后处理训练轮数 | 6 |
| `--learning_rate` | 学习率（建议使用较小值） | 0.0001 |
| `--batch_size` | 批次大小 | 32 |
| `--patience` | Early stopping耐心值 | 3 |

## 输出结果

### 1. 模型checkpoint
保存在：`./checkpoints/{setting}_post/checkpoint.pth`

### 2. 测试结果
保存在：`./results/{setting}_post/`
- `metrics.npy`: 包含 [MAE, MSE, RMSE, MAPE, MSPE]
- `pred.npy`: 预测结果
- `true.npy`: 真实值

## 支持的数据集

- ETTh1 (电力变压器温度 - 小时级别)
- ETTh2 (电力变压器温度 - 小时级别)
- ETTm1 (电力变压器温度 - 15分钟级别)
- ETTm2 (电力变压器温度 - 15分钟级别)
- WTH (天气数据)
- ECL (电力负载)
- Solar (太阳能发电)
- Custom (自定义数据)

## 使用流程示例

### Step 1: 正常训练模型
```bash
python main_informer.py \
  --model informer \
  --data ETTh1 \
  --features M \
  --train_epochs 6
```

### Step 2: 后处理训练
```bash
# 假设训练好的模型在：
# ./checkpoints/informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/checkpoint.pth

python main_post_training.py \
  --model informer \
  --data ETTh1 \
  --features M \
  --checkpoint_path ./checkpoints/informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/checkpoint.pth \
  --train_epochs 6 \
  --learning_rate 0.00001
```

## 注意事项

1. **checkpoint_path是必需参数**，必须指向一个有效的预训练模型
2. 后处理训练建议使用**较小的学习率**（如0.00001），避免破坏原有训练效果
3. 模型参数（d_model, n_heads等）必须与预训练模型保持一致
4. 后处理训练会在训练数据上进行，因此需要足够的内存来存储所有预测结果
5. 建议先在验证集上评估后处理效果，再决定是否使用

## 代码结构

```
exp/
├── exp_basic.py              # 基础实验类
├── exp_informer.py           # 标准Informer训练
└── exp_post_training.py      # 后处理训练模块 (NEW)

main_post_training.py          # 后处理训练主程序 (NEW)

scripts/
└── post_training_ETTh1.sh    # 后处理训练脚本示例 (NEW)
```

## 理论依据

后处理训练的核心思想是：
- 通过让模型预测接近"预测值"和"真实值"的平均水平
- 可以减少过拟合，提高模型的泛化能力
- 这是一种模型正则化和微调的技术

损失函数设计：
```
mean_value = (prediction + ground_truth) / 2
loss = MSE(prediction - mean_value, 0)
     = MSE(prediction, mean_value)
```

这使得模型在保持预测能力的同时，向预测-真实的中间状态靠拢，从而获得更平衡的性能。

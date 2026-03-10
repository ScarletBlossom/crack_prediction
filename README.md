## 项目结构

```text
TwoStageGAN/
├── twostage_gan/
│   ├── config.py                    # 默认配置
│   ├── datasets/
│   │   └── triplet_dataset.py       # 数据切分、增强、DataLoader
│   ├── models/
│   │   ├── blocks.py                # 基础网络模块
│   │   ├── generator.py             # 生成器
│   │   └── discriminator.py         # 判别器
│   ├── losses/
│   │   └── gan_losses.py            # GAN + L1 损失
│   ├── engine/
│   │   └── trainer.py               # 单步训练与整阶段训练
│   ├── utils/
│   │   ├── checkpoint.py            # 权重保存/加载
│   │   └── visualization.py         # 推理可视化
│   ├── train.py                     # 训练入口
│   └── infer.py                     # 推理入口
├── requirements.txt
└── README.md
```

## 数据约定

### 训练集
训练图像默认命名为 `Train_*.jpg`，并且按横向拼接为：

```text
input | sed | crack
```

### 测试集
测试图像默认命名为 `Test_*.jpg`，并且按横向拼接为：

```text
input | sed | unused | crack | unused
```

这部分逻辑已在 `datasets/triplet_dataset.py` 中单独封装。

## 环境安装

```bash
pip install -r requirements.txt
```

## 训练

### Stage 1: geometry -> SED

```bash
python -m twostage_gan.train --stage stage1 --train-dir ./train_img --test-dir ./test_img
```

### Stage 2: SED -> crack

```bash
python -m twostage_gan.train --stage stage2 --train-dir ./train_img --test-dir ./test_img
```

默认会把权重保存到：

```text
./checkpoints/
```

## 推理

```bash
python -m twostage_gan.infer \
  --test-dir ./test_img \
  --generator1-path ./checkpoints/generator1_sed.pth \
  --generator2-path ./checkpoints/generator2_crack.pth
```

输出为 5 列对比图：

```text
Input | SED Ground Truth | SED Prediction | Crack Ground Truth | Crack Prediction
```


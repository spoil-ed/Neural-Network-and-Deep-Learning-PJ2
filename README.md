# Project 2 - Neural Network and Deep Learning

CIFAR-10 分类与 Batch Normalization 分析项目。

**作者**：Yu Xinglei，Student ID: 23300290012  
**报告**：[report/report.pdf](report/report.pdf)  
**数据集与权重**：https://www.modelscope.cn/models/imspoiled/Neural-Network-and-Deep-Learning-PJ2-Checkpoin

## 结果

| 任务 | 模型 / 设置 | 最佳结果 |
|---|---|---:|
| Part 1 | ResNet-18 baseline | 95.34% |
| Part 1 | WideResNet-28-10 + strong recipe | 98.02% |
| Part 2 | VGG-A | 78.6% |
| Part 2 | VGG-A + BatchNorm | 83.0% |

## 目录

```text
codes/
  part1_cifar/        # CIFAR-10 分类、消融、可视化
  VGG_BatchNorm/      # VGG-A 与 BatchNorm 分析
results/
  logs/               # 训练日志
  figures/            # 实验图表
  models/             # .pth 权重，需下载或训练生成
data/                 # CIFAR-10，首次运行自动下载
report/report.pdf     # 实验报告
```

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

推荐使用 CUDA GPU。指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```

## 数据集

CIFAR-10 会由脚本自动下载到 `data/`：

```bash
python - <<'PY'
from codes.part1_cifar.data import get_loaders
get_loaders(batch_size=128, num_workers=0)
print("CIFAR-10 ready")
PY
```

若手动下载，请保证目录为：

```text
data/cifar-10-batches-py/
```

## 权重

下载预训练权重后放入：

```text
results/models/
```

常用文件名：

```text
best.pth
manual.pth
wrn_mix300.pth
VGG_A.pth
VGG_A_BatchNorm.pth
```

## 训练

所有命令在仓库根目录运行。

```bash
# ResNet-18 baseline
python codes/part1_cifar/run_best.py --model resnet18 --optimizer sgd --epochs 150 --tag best

# 手写 SGD 对照
python codes/part1_cifar/run_best.py --model resnet18 --optimizer manual_sgd --epochs 150 --tag manual

# 消融实验
for group in width activation loss optimizer optional optional_arch optional_training; do
  python codes/part1_cifar/run_ablations.py --group "$group" --epochs 40
done

# WideResNet-28-10 strong recipe
python codes/part1_cifar/run_strong.py --model wrn28_10 --mix 1 --epochs 300 --tag wrn_mix300

# 手写优化器验证
python codes/part1_cifar/test_optimizers.py

# 纯张量 ConvNet
python codes/part1_cifar/manual_net.py --selftest
python codes/part1_cifar/manual_net.py --epochs 40

# VGG-A vs VGG-A + BN
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --epochs 20
python codes/VGG_BatchNorm/bn_gradient_analysis.py --epochs 3
```

## 评估与画图

训练脚本会自动在 test set 上评估并保存最佳权重。已有权重时可直接生成图表：

```bash
python codes/part1_cifar/visualize.py --model resnet18 --ckpt best
python codes/part1_cifar/visualize.py --model wrn28_10 --ckpt wrn_mix300

python codes/part1_cifar/plot_part1.py
python codes/part1_cifar/plot_strong.py

python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --replot
python codes/VGG_BatchNorm/bn_gradient_analysis.py --replot
```

## 输出

```text
results/models/*.pth       # 模型权重
results/logs/*.json        # 训练曲线与指标
results/figures/*.png/pdf  # 报告图表
```

## 报告

```bash
cd report
latexmk -pdf report.tex
```

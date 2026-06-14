# Model Checkpoints

This directory stores the trained checkpoints used by the report
**CIFAR--10 Classification and Batch Normalization**. The same files can be
uploaded to ModelScope as the model-weight artifact package.

## Expected Files

```text
best.pth              # ResNet-18 baseline, 95.34% test accuracy
manual.pth            # ResNet-18 trained with the manual SGD optimizer, 95.34%
wrn_mix300.pth        # WideResNet-28-10 strong recipe, 98.02% test accuracy
VGG_A.pth             # VGG-A baseline for the BatchNorm study
VGG_A_BatchNorm.pth   # VGG-A with BatchNorm
```

The exact uploaded set may vary, but these are the names expected by the report
and scripts.

## Dataset

The experiments use CIFAR-10. The dataset does not need to be included with the
checkpoints because the scripts download it automatically through
`torchvision.datasets.CIFAR10`.

Expected local dataset path after download:

```text
data/cifar-10-batches-py/
```

## Results

| Model / setting | Metric | Result |
|---|---:|---:|
| ResNet-18 baseline | CIFAR-10 test accuracy | 95.34% |
| ResNet-18 + manual SGD | CIFAR-10 test accuracy | 95.34% |
| WideResNet-28-10 + AutoAugment + Cutout + Mixup/CutMix + EMA | CIFAR-10 test accuracy | 98.02% |
| VGG-A | validation accuracy | 78.64% |
| VGG-A + BatchNorm | validation accuracy | 83.04% |

## Usage

Place downloaded checkpoint files in this directory:

```text
results/models/
```

Install dependencies from the project root:

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

Regenerate Part 1 figures from existing checkpoints:

```bash
python codes/part1_cifar/visualize.py --model resnet18 --ckpt best
python codes/part1_cifar/visualize.py --model wrn28_10 --ckpt wrn_mix300
python codes/part1_cifar/plot_part1.py
python codes/part1_cifar/plot_strong.py
```

Regenerate the BatchNorm analysis figures:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --replot
python codes/VGG_BatchNorm/bn_gradient_analysis.py --replot
```

## Reproducing Training

```bash
# ResNet-18 baseline
python codes/part1_cifar/run_best.py --model resnet18 --optimizer sgd --epochs 150 --tag best

# Manual SGD comparison
python codes/part1_cifar/run_best.py --model resnet18 --optimizer manual_sgd --epochs 150 --tag manual

# WideResNet-28-10 strong recipe
python codes/part1_cifar/run_strong.py --model wrn28_10 --mix 1 --epochs 300 --tag wrn_mix300

# VGG-A vs. VGG-A + BatchNorm
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --epochs 20
python codes/VGG_BatchNorm/bn_gradient_analysis.py --epochs 3
```

## Report

The compiled report is available at:

```text
report/report.pdf
```

The report includes the full experimental setup, ablation tables, figures, and
BatchNorm landscape analysis.

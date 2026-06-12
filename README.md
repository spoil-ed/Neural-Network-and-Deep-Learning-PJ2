# Project-2 — Neural Network and Deep Learning

CIFAR-10 image classification and a study of Batch Normalization.

**Author:** Yu Xinglei (Student ID: 23300290012)
**Report:** [`report/report.pdf`](report/report.pdf)
**Trained weights & dataset:** `[your Google-Drive / Netdisk link]`

---

## Results at a glance

### Part 1 — CIFAR-10 classification
| Model | Recipe | Params | Test acc | Error |
|---|---|---:|---:|---:|
| ResNet-18 | crop+flip, 150 ep (baseline) | 11.2M | 95.34% | 4.66% |
| **WideResNet-28-10** | **AutoAug+Cutout+Mixup/CutMix+EMA+TTA, 300 ep** | **36.5M** | **98.02%** | **1.98%** |

Controlled ablations on a configurable ConvNet:
- **Width** (#filters): 0.25×→87.5%, 0.5×→89.4%, 1.0×→90.9%, 2.0×→92.0% (diminishing returns).
- **Activation**: LeakyReLU 91.3% ≳ GELU 91.1% ≈ ReLU 90.9% ≫ Tanh 88.3% ≫ Sigmoid 84.4%.
- **Loss/regularization**: label-smoothing 91.5% > L2 90.9% > dropout 90.7% > MSE 90.0% > plain CE 89.8%.
- **Optimizer**: SGD 90.9% **=** hand-written `ManualSGD` 90.9% (exact); Adam 90.4%, AdamW 90.4%, `ManualAdam` 90.1%, RMSprop 79.6%.
- **Optional choices** on the ConvNet: architecture options show deeper stages
  perform best (**93.08%**), with BN+Residual+Dropout at 91.50%; training
  options show label smoothing best (**91.74%**), followed by combined
  regularization 91.60% and Cutout 91.44%, while no augmentation and constant
  LR underperform.

The two **from-scratch optimizers** (`ManualSGD`, `ManualAdam`) reproduce `torch.optim`
to `<1e-6` (SGD bit-exact) and train the full ResNet to the identical 95.34%.
A further ConvNet built **purely from raw tensor operations** (im2col conv,
unfold pooling, manual ReLU/cross-entropy; task 5b) reaches **85.51%** in 40 epochs.

### Part 2 — Batch Normalization
- VGG-A **78.6%** vs. VGG-A + BatchNorm **83.0%** best validation accuracy.
- BN gives a tighter/lower **loss landscape** (variation over learning rates) and more
  **predictive gradients** (p90 max gradient change 16.6 → 10.3), confirming it smooths
  the optimization landscape (Santurkar et al., 2018).

---

## Repository structure

```
PJ2/
├── report/
│   ├── report.tex            # full LaTeX report
│   ├── references.bib
│   └── report.pdf            # compiled report (11 pages)
├── codes/
│   ├── part1_cifar/          # Part 1 — CIFAR-10 classification
│   │   ├── data.py           #   loaders + standard / strong augmentation
│   │   ├── models.py         #   ConvNet, ResNet-18, WideResNet-28-10
│   │   ├── optimizers.py     #   hand-written ManualSGD / ManualAdam
│   │   ├── engine.py         #   training / evaluation loop
│   │   ├── manual_net.py     #   task 5(b): ConvNet from raw tensor ops only
│   │   ├── run_best.py       #   ResNet-18 baseline trainer
│   │   ├── run_ablations.py  #   width / activation / loss / optimizer ablations
│   │   ├── run_strong.py     #   accuracy-push trainer (AutoAug+Mixup+EMA+TTA)
│   │   ├── visualize.py      #   filters, confusion, t-SNE, loss surface
│   │   ├── plot_part1.py     #   training curves + ablation figures
│   │   ├── plot_strong.py    #   accuracy-push figures
│   │   └── test_optimizers.py#   validates ManualSGD/ManualAdam vs torch.optim
│   └── VGG_BatchNorm/        # Part 2 — Batch Normalization
│       ├── models/vgg.py     #   VGG_A and VGG_A_BatchNorm
│       ├── data/loaders.py
│       ├── VGG_Loss_Landscape.py   # train VGG-A ± BN, loss landscape
│       └── bn_gradient_analysis.py # gradient predictiveness / β-smoothness
├── results/
│   ├── figures/              # all report figures (.png)
│   ├── logs/                 # per-run metrics (.json) + console logs (.out)
│   └── models/               # trained weights (.pth)  ← upload these to a drive
└── data/                     # CIFAR-10 (auto-downloaded on first run)
```

---

## Setup

```bash
# Python 3.10+, a CUDA GPU recommended.
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

CIFAR-10 downloads automatically to `./data` on the first run (≈170 MB).

---

## Reproduce

All commands are run from the repository root.

### Part 1
```bash
# Baseline ResNet-18 (95.34%)
python codes/part1_cifar/run_best.py --model resnet18 --optimizer sgd       --epochs 150 --tag best
# Same model trained with the hand-written optimizer (task 5c)
python codes/part1_cifar/run_best.py --model resnet18 --optimizer manual_sgd --epochs 150 --tag manual

# Ablations (tasks 4 & 5a)
for g in width activation loss optimizer optional optional_arch optional_training; do
  python codes/part1_cifar/run_ablations.py --group $g --epochs 40
done

# Accuracy push — WideResNet-28-10 (98.02%)
python codes/part1_cifar/run_strong.py --model wrn28_10 --mix 1 --epochs 300 --tag wrn_mix300

# Hand-written optimizer validation, and figures
python codes/part1_cifar/test_optimizers.py

# Raw-tensor ConvNet (task 5b): self-test, then 40-epoch training (85.51%)
python codes/part1_cifar/manual_net.py --selftest
python codes/part1_cifar/manual_net.py --epochs 40
python codes/part1_cifar/visualize.py --model wrn28_10 --ckpt wrn_mix300
python codes/part1_cifar/plot_part1.py
python codes/part1_cifar/plot_strong.py
```

### Part 2
```bash
# VGG-A vs VGG-A+BN, loss landscape over learning rates [1e-3, 2e-3, 1e-4, 5e-4]
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --epochs 20
# Gradient predictiveness / effective beta-smoothness
python codes/VGG_BatchNorm/bn_gradient_analysis.py --epochs 3
```

Pin a GPU with `CUDA_VISIBLE_DEVICES=k` before any command.

### Build the report
```bash
cd report && latexmk -pdf report.tex
```

---

## Notes
- Training uses bfloat16 automatic mixed precision; experiments were run on a single
  NVIDIA H100 (ResNet-18 ≈ 9.7 s/epoch, WRN-28-10 ≈ 15 s/epoch).
- Trained checkpoints live in `results/models/` (`best.pth`, `manual.pth`,
  `wrn_mix300.pth`, `VGG_A.pth`, `VGG_A_BatchNorm.pth`). Upload these to a drive and
  paste the link above and in the report.

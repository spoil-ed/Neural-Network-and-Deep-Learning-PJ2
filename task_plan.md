# PJ2 — Neural Network & Deep Learning, Project 2

## Decisions
- Report language: **English** (LaTeX)
- Part 1 scope: **Comprehensive + hand-written optimizer**
- Author: placeholders `[Your Name]` / `[Student ID]`
- Compute: 8× H100 80GB. Training cost is not a constraint.

## Part 1 — CIFAR-10 CNN (60%)
- [ ] data.py — loaders + augmentation (CIFAR mean/std, randcrop, hflip)
- [ ] models.py — ConvNet (configurable: width/act/bn/dropout) + ResNet-18 (CIFAR)
- [ ] optimizers.py — custom optimizer from scratch (Adam, manual) [task 5c]
- [ ] engine.py — train/eval/logging/checkpoint
- [ ] run_best.py — train best model (ResNet-18) → best test acc
- [ ] run_ablations.py — filters(width)/loss+reg/activation/optimizer ablations [tasks 4,5a]
- [ ] visualize.py — first-layer filters, confusion matrix, per-class acc, t-SNE, loss surface [task 6]

## Part 2 — Batch Normalization (30%)
- [ ] fix models/vgg.py import; add VGG_A_BatchNorm
- [ ] fix data/loaders.py root
- [ ] VGG_Loss_Landscape.py — train VGG-A vs VGG-A-BN; loss landscape (max/min over LRs);
      gradient predictiveness; beta-smoothness [2.2, 2.3]

## Report (10%)
- [ ] report/report.tex — full write-up with all figures, tables, GitHub/weights placeholders

## Results (final)
- **HEADLINE: WideResNet-28-10 + AutoAugment+Cutout + Mixup/CutMix + EMA + TTA, 300ep = 98.02% test acc (1.98% error), 36.5M params.**
  - Race: wrn_mix300 98.02 > wrn_mix200 97.73 > wrn_base300 97.54 > wrn_base200 97.38 > r18_strong 96.96 (all w/ TTA).
  - Weights: results/models/wrn_mix300.pth (146MB). Insights regenerated on this model (cat 94.6%).
- ResNet-18 baseline (crop+flip, 150ep): 95.34% test acc, 11.17M params, 9.7s/epoch.
- ManualSGD ResNet-18: **95.34%** (identical to torch.optim.SGD — validates from-scratch optimizer).
- Width: 0.25x 87.46 / 0.5x 89.36 / 1.0x 90.91 / 2.0x 92.04 (diminishing returns).
- Activation: leakyrelu 91.34 > gelu 91.08 > relu 90.91 > tanh 88.32 > sigmoid 84.39.
- Loss/reg: label-smooth 91.51 > L2 90.91 > dropout 90.71 > MSE 90.02 > no-wd 89.81.
- Optimizer: SGD 90.91 = ManualSGD 90.91 (exact); Adam 90.44, AdamW 90.40, ManualAdam 90.11, RMSprop 79.64.
- Opt validation: SGD exact (0), Adam/AdamW max|diff| 7.2e-7.
- BN: VGG-A 78.64 vs VGG-A+BN 83.04; loss band tighter+lower; grad-change p90 16.6 vs 10.3.

## Status: COMPLETE
- All code written + run on 8xH100; 14 figures + 4 weight files in results/.
- report/report.pdf compiles cleanly (9 pages, no undefined refs/citations/missing figures).
- TODO for user: fill [Your Name]/[Student ID]; push code to GitHub + upload results/models to Drive; paste links into abstract.

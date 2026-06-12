# Optional Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add architecture and training optional-choice ablations to the Project 2 CIFAR-10 report workflow.

**Architecture:** Extend the existing compact `ConvNet` knobs without changing its default behavior, then add two new ablation groups: `optional_arch` and `optional_training`. Reuse the current JSON log and plotting conventions so the report can include new tables/figures with minimal disruption.

**Tech Stack:** Python, PyTorch, torchvision transforms, matplotlib, pytest, LaTeX.

---

### File Structure

- Modify `codes/part1_cifar/models.py`: add `pooling`, `depth_variant`, `classifier_width`, `norm`, and `conv_dropout` options to `ConvNet`.
- Modify `codes/part1_cifar/data.py`: expose a loader option for train augmentation mode so training ablations can compare standard, none, and cutout.
- Modify `codes/part1_cifar/run_ablations.py`: add configs for `optional_arch` and `optional_training`; support config-level loader and scheduler choices.
- Modify `codes/part1_cifar/plot_part1.py`: include optional groups in generated figures and text summary.
- Create `codes/part1_cifar/test_optional_ablations.py`: test model knobs and config construction.
- Modify `report/report.tex`: add concise architecture/training optional ablation tables and references to the new figures.
- Update generated outputs only after smoke runs: `results/logs/ablation_optional_arch.json`, `results/logs/ablation_optional_training.json`, `results/figures/ablation_optional_arch.*`, `results/figures/ablation_optional_training.*`.

### Task 1: Tests for New Optional Knobs

**Files:**
- Create: `codes/part1_cifar/test_optional_ablations.py`
- Read: `codes/part1_cifar/models.py`
- Read: `codes/part1_cifar/run_ablations.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `ConvNet`, `count_parameters`, and `configs`.

```python
import torch

from models import ConvNet, count_parameters
from run_ablations import configs


def test_convnet_optional_knobs_preserve_output_shape():
    x = torch.randn(2, 3, 32, 32)
    variants = [
        ConvNet(pooling='avg'),
        ConvNet(depth_variant='deep'),
        ConvNet(classifier_width=256),
        ConvNet(classifier_width=1024),
        ConvNet(norm='group'),
        ConvNet(conv_dropout=0.1),
    ]
    for model in variants:
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 10)


def test_convnet_optional_knobs_change_parameterization():
    base = ConvNet()
    deep = ConvNet(depth_variant='deep')
    narrow_head = ConvNet(classifier_width=256)
    wide_head = ConvNet(classifier_width=1024)
    assert count_parameters(deep) > count_parameters(base)
    assert count_parameters(narrow_head) < count_parameters(base)
    assert count_parameters(wide_head) > count_parameters(base)


def test_new_optional_ablation_groups_are_defined():
    arch = configs('optional_arch')
    training = configs('optional_training')
    assert len(arch) >= 8
    assert len(training) >= 7
    assert all('model_kw' in cfg for _, cfg in arch)
    assert all('loader_kw' in cfg for _, cfg in training)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=codes/part1_cifar pytest codes/part1_cifar/test_optional_ablations.py -q
```

Expected: FAIL because `ConvNet` does not yet accept the new keyword arguments and the new ablation groups do not exist.

### Task 2: Extend ConvNet and Data Loader Options

**Files:**
- Modify: `codes/part1_cifar/models.py`
- Modify: `codes/part1_cifar/data.py`
- Test: `codes/part1_cifar/test_optional_ablations.py`

- [ ] **Step 1: Implement model knobs**

In `ConvNet.__init__`, add keyword arguments:

```python
pooling='max',
depth_variant='default',
classifier_width=512,
norm='batch',
conv_dropout=0.0,
```

Implement `norm='none'`, `norm='batch'`, and `norm='group'`. Use GroupNorm with at most 32 groups and a divisor of the channel count. Replace hard-coded `MaxPool2d` creation with a helper that returns `MaxPool2d` or `AvgPool2d`. For `depth_variant='deep'`, add one extra convolution to each stage. Insert `Dropout2d(conv_dropout)` after stage activations when `conv_dropout > 0`.

- [ ] **Step 2: Implement loader augmentation modes**

In `get_loaders`, add `augment='standard'`. Map values:

- `standard`: existing random crop and horizontal flip.
- `none`: no train augmentation before normalization.
- `cutout`: existing standard augmentation plus `RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))` after `ToTensor`.

- [ ] **Step 3: Run tests to verify green for model/data pieces**

Run:

```bash
PYTHONPATH=codes/part1_cifar pytest codes/part1_cifar/test_optional_ablations.py -q
```

Expected: remaining failure only for missing `optional_arch` and `optional_training` configs if Task 3 is not implemented yet.

### Task 3: Add Optional Ablation Configs and Plotting

**Files:**
- Modify: `codes/part1_cifar/run_ablations.py`
- Modify: `codes/part1_cifar/plot_part1.py`
- Test: `codes/part1_cifar/test_optional_ablations.py`

- [ ] **Step 1: Add ablation groups**

Add `optional_arch` configs:

- `Baseline BN ConvNet`
- `No normalization`
- `GroupNorm`
- `Average pooling`
- `Deeper stages`
- `Narrow classifier`
- `Wide classifier`
- `Conv dropout`
- `BN + Residual + Dropout`

Add `optional_training` configs:

- `Standard crop+flip`
- `No augmentation`
- `Cutout`
- `No weight decay`
- `Label smoothing`
- `Step LR`
- `Constant LR`
- `Combined regularization`

Each config should include `loader_kw` and `scheduler` keys with defaults, so the training loop can vary data augmentation and scheduler without special cases.

- [ ] **Step 2: Update CLI choices and main loop**

Extend `--group` choices to include `optional_arch` and `optional_training`. Build loaders per config only when `loader_kw` differs, otherwise reuse the standard loader. Use scheduler values:

- `cosine`: `CosineAnnealingLR`
- `step`: `MultiStepLR` with milestones at half and three-quarters of total epochs.
- `constant`: no scheduler.

- [ ] **Step 3: Plot new groups**

Add display names for the new configs and include `optional_arch` and `optional_training` in the groups loop used by `plot_part1.py`.

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=codes/part1_cifar pytest codes/part1_cifar/test_optional_ablations.py -q
```

Expected: PASS.

### Task 4: Smoke Runs, Figures, and Report

**Files:**
- Modify: `report/report.tex`
- Generated: `results/logs/ablation_optional_arch.json`
- Generated: `results/logs/ablation_optional_training.json`
- Generated: `results/figures/ablation_optional_arch.pdf`
- Generated: `results/figures/ablation_optional_training.pdf`

- [ ] **Step 1: Smoke-run new groups**

Run:

```bash
python codes/part1_cifar/run_ablations.py --group optional_arch --epochs 1 --device cpu
python codes/part1_cifar/run_ablations.py --group optional_training --epochs 1 --device cpu
```

Expected: both commands complete and write JSON logs with all configs.

- [ ] **Step 2: Regenerate figures**

Run:

```bash
python codes/part1_cifar/plot_part1.py
```

Expected: `ablation_optional_arch` and `ablation_optional_training` figures are created under `results/figures/`.

- [ ] **Step 3: Update report**

Revise `report/report.tex` optional-components subsection to describe architecture options and training options. Use smoke-run values only as smoke evidence unless full 40-epoch logs exist; if only smoke logs exist, state that the code now supports the full controlled experiments and keep existing full-result claims unchanged.

- [ ] **Step 4: Compile report if available**

Run:

```bash
cd report && latexmk -pdf report.tex
```

Expected: exit code 0 if `latexmk` is installed. If unavailable, report the toolchain gap and run a syntax-oriented check by scanning for missing figure paths.

### Task 5: Final Verification

**Files:**
- All touched files

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=codes/part1_cifar pytest codes/part1_cifar/test_optimizers.py codes/part1_cifar/test_optional_ablations.py -q
```

Expected: PASS.

- [ ] **Step 2: Inspect changed files**

Run:

```bash
find docs codes report results -maxdepth 4 -type f -newer docs/superpowers/specs/2026-06-11-optional-ablation-design.md
```

Expected: output lists only files related to this optional-ablation task.

- [ ] **Step 3: No commit step**

This workspace is not a git repository, so do not run `git add` or `git commit`.
Report the changed file list and verification results instead.

# Optional Ablation Design

## Goal

Expand the Project 2 report so the required optional components are compared
systematically, not only BatchNorm, Dropout, and Residual connections. The
scope covers both architecture choices and training choices, while preserving
the existing 40-epoch compact ConvNet ablation protocol.

## Context

The project already has:

- `codes/part1_cifar/run_ablations.py` for controlled ConvNet ablations.
- `codes/part1_cifar/models.py` with configurable width, activation,
  BatchNorm, dropout, and residual shortcuts.
- Existing logs for width, activation, loss, optimizer, and optional groups.
- Report text and tables for BatchNorm, Dropout, Residual, width, activation,
  loss/regularization, optimizer, and the strong WRN recipe.

The current gap is that the optional-components section focuses on three
architecture switches, while the report requirement is better served by a
broader optional-choice comparison spanning architecture and training options.

## Architecture Optional Ablations

Use the same compact ConvNet, dataset, seed, optimizer, learning-rate schedule,
batch size, and 40-epoch training length as the existing ablations.

Add supported knobs to `ConvNet` only where needed:

- Pooling type: max pooling vs average pooling.
- Depth: default stage depth vs a deeper stage layout.
- Classifier width: small, default, and wide hidden layer.
- Normalization type: none, BatchNorm, and GroupNorm.
- Dropout placement: classifier-only vs convolutional plus classifier dropout.

Keep the existing BatchNorm, Dropout, Residual, and all-three comparison. These
remain the core optional-component table.

## Training Optional Ablations

Add a separate training-options group on the same ConvNet protocol:

- Standard crop+flip augmentation as the baseline.
- No augmentation.
- Cutout/random erasing.
- Label smoothing.
- Weight decay.
- Combined regularization where the comparison is meaningful.
- Learning-rate schedule choices: cosine, step, and constant.

Do not mix WRN-only recipe items such as Mixup/CutMix, EMA, and TTA into the
compact ConvNet optional table unless the existing loaders and training loop
already support them cleanly. Those remain in the accuracy-push section,
because they are evaluated under a different model and longer schedule.

## Outputs

Produce JSON logs under `results/logs/`:

- `ablation_optional_arch.json`
- `ablation_optional_training.json`

Generate matching figures under `results/figures/`:

- `ablation_optional_arch.pdf/png`
- `ablation_optional_training.pdf/png`

Also update `part1_summary.txt` through the existing plotting script.

## Report Updates

Revise the optional-components section to include:

- A concise architecture optional table.
- A concise training optional table.
- A short interpretation of which choices help under the controlled ConvNet
  protocol and which choices are reserved for the strong WRN recipe.

Keep the existing width, activation, loss, optimizer, and Part 2 BatchNorm
sections intact, except for cross-references needed to avoid repetition.

## Testing and Verification

Before implementation, add tests that instantiate the new ConvNet options and
verify output shape, parameter-count differences, and valid training config
construction.

After implementation:

- Run the new focused tests.
- Smoke-run the new ablation groups for one epoch on CPU or CUDA if available.
- Regenerate figures from existing or newly produced logs.
- Compile the LaTeX report if the local TeX toolchain is available.

## Non-goals

- Re-running the 200/300-epoch WRN strong sweep.
- Changing headline accuracy claims.
- Replacing existing completed ablation results unless the new logs are
  actually regenerated.
- Refactoring unrelated training code.

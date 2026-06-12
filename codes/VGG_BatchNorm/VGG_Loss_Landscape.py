"""Batch-Normalization study (Project Part 2).

Trains VGG-A and VGG-A-BatchNorm on CIFAR-10 over a set of learning rates while
recording, at every optimisation step, the training loss and gradient
statistics. From these we produce:

  * Section 2.2  -- VGG-A vs VGG-A-BN training-loss / validation-accuracy curves.
  * Section 2.3  -- the loss landscape (max/min loss band over learning rates,
                    filled with ``plt.fill_between``) and gradient
                    predictiveness / effective beta-smoothness bands.

Run (pin the GPU from outside with CUDA_VISIBLE_DEVICES):
    python VGG_Loss_Landscape.py --epochs 20
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubstyle
pubstyle.apply()
import torch
from torch import nn
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# --------------------------------------------------------------------------- #
# this file lives at PJ2/codes/VGG_BatchNorm/ -> three levels up is the PJ2 root
HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(HOME, 'results')
FIG_DIR = os.path.join(RESULTS, 'figures')
LOG_DIR = os.path.join(RESULTS, 'logs')
MODEL_DIR = os.path.join(RESULTS, 'models')
for d in (FIG_DIR, LOG_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_random_seeds(seed_value=2020):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


@torch.no_grad()
def get_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_and_record(model, optimizer, criterion, train_loader, val_loader, epochs):
    """Train, recording per-step loss + last-layer gradient statistics."""
    model.to(device)
    step_losses = []        # training loss at every step
    grad_norms = []         # ||g_t||  of the final classifier weight
    grad_diffs = []         # ||g_t - g_{t-1}||  (change of the gradient)
    val_acc_curve = []
    train_loss_curve = []
    prev_grad = None

    # the final classification layer weight, used for the gradient measures
    ref_param = model.classifier[4].weight

    for epoch in tqdm(range(epochs), unit='epoch'):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            g = ref_param.grad.detach().reshape(-1)
            gn = g.norm().item()
            grad_norms.append(gn)
            grad_diffs.append((g - prev_grad).norm().item() if prev_grad is not None else np.nan)
            prev_grad = g.clone()

            optimizer.step()
            lv = loss.item()
            step_losses.append(lv)
            running += lv * y.size(0)
            n += y.size(0)

        train_loss_curve.append(running / n)
        val_acc_curve.append(get_accuracy(model, val_loader))
    return dict(step_losses=step_losses, grad_norms=grad_norms,
                grad_diffs=grad_diffs, val_acc_curve=val_acc_curve,
                train_loss_curve=train_loss_curve)


def run_model_family(model_cls, name, lrs, epochs, train_loader, val_loader):
    runs = {}
    best_acc, best_state = 0.0, None
    for lr in lrs:
        set_random_seeds(2020)
        model = model_cls()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        print(f'  [{name}] lr={lr}', flush=True)
        rec = train_and_record(model, opt, crit, train_loader, val_loader, epochs)
        runs[str(lr)] = rec
        acc = max(rec['val_acc_curve'])
        print(f'  [{name}] lr={lr} best_val_acc={acc:.2f}', flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, os.path.join(MODEL_DIR, f'{name}.pth'))
    return runs, best_acc


# --- aggregation: max/min band over learning rates -------------------------- #
def _smooth(x, w=15):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode='valid')


def minmax_band(runs, key, smooth=1):
    series = [_smooth(np.asarray(runs[lr][key], dtype=float), smooth) for lr in runs]
    L = min(len(s) for s in series)
    M = np.stack([s[:L] for s in series], axis=0)
    return np.nanmin(M, axis=0), np.nanmax(M, axis=0)


def plot_loss_landscape(std_runs, bn_runs, fname, smooth=15, ymax=2.6):
    std_min, std_max = minmax_band(std_runs, 'step_losses', smooth)
    bn_min, bn_max = minmax_band(bn_runs, 'step_losses', smooth)
    steps_std = np.arange(len(std_min))
    steps_bn = np.arange(len(bn_min))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.fill_between(steps_std, std_min, std_max, alpha=0.35,
                    color=pubstyle.GREEN, linewidth=0,
                    label='Standard VGG-A')
    ax.fill_between(steps_bn, bn_min, bn_max, alpha=0.35,
                    color=pubstyle.RED, linewidth=0,
                    label='VGG-A + BatchNorm')
    ax.plot(steps_std, std_max, color=pubstyle.GREEN, lw=0.9)
    ax.plot(steps_std, std_min, color=pubstyle.GREEN, lw=0.9)
    ax.plot(steps_bn, bn_max, color=pubstyle.RED, lw=0.9)
    ax.plot(steps_bn, bn_min, color=pubstyle.RED, lw=0.9)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.legend(); pubstyle.grid(ax)
    plt.tight_layout(pad=1)
    pubstyle.save(FIG_DIR, fname)


def plot_grad_predictiveness(std_runs, bn_runs, fname):
    std_min, std_max = minmax_band(std_runs, 'grad_diffs')
    bn_min, bn_max = minmax_band(bn_runs, 'grad_diffs')
    # skip first nan
    s = np.arange(len(std_min))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.fill_between(s, std_min, std_max, alpha=0.35, color=pubstyle.GREEN,
                    linewidth=0, label='Standard VGG-A')
    s2 = np.arange(len(bn_min))
    ax.fill_between(s2, bn_min, bn_max, alpha=0.35, color=pubstyle.RED,
                    linewidth=0, label='VGG-A + BatchNorm')
    ax.plot(s, std_max, color=pubstyle.GREEN, lw=0.9)
    ax.plot(s2, bn_max, color=pubstyle.RED, lw=0.9)
    ax.set_xlabel('Training step')
    ax.set_ylabel(r'$\|\nabla_t - \nabla_{t-1}\|$  (gradient change)')
    ax.legend(); pubstyle.grid(ax)
    plt.tight_layout(pad=1)
    pubstyle.save(FIG_DIR, fname)


def plot_22_comparison(std_runs, bn_runs, lr, fname):
    sr, br = std_runs[str(lr)], bn_runs[str(lr)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(sr['train_loss_curve'], color=pubstyle.GREEN, label='Standard VGG-A')
    ax[0].plot(br['train_loss_curve'], color=pubstyle.RED, label='VGG-A + BN')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Train loss')
    ax[0].set_title(f'Training loss (lr={lr})')
    ax[0].legend(); pubstyle.grid(ax[0])
    ax[1].plot(sr['val_acc_curve'], color=pubstyle.GREEN, label='Standard VGG-A')
    ax[1].plot(br['val_acc_curve'], color=pubstyle.RED, label='VGG-A + BN')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Val accuracy (%)')
    ax[1].set_title(f'Validation accuracy (lr={lr})')
    ax[1].legend(loc='lower right'); pubstyle.grid(ax[1])
    plt.tight_layout(pad=1)
    pubstyle.save(FIG_DIR, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-3, 2e-3, 1e-4, 5e-4])
    ap.add_argument('--replot', action='store_true',
                    help='skip training, regenerate figures from results/logs/bn_runs.json')
    args = ap.parse_args()

    if args.replot:
        d = json.load(open(os.path.join(LOG_DIR, 'bn_runs.json')))
        std_runs, bn_runs = d['std'], d['bn']
        plot_loss_landscape(std_runs, bn_runs, 'bn_loss_landscape.png')
        cmp_lr = 1e-3 if str(1e-3) in std_runs else float(list(std_runs)[0])
        plot_22_comparison(std_runs, bn_runs, cmp_lr, 'bn_vgg_comparison.png')
        print('replotted from bn_runs.json', flush=True)
        return

    print('device:', device, flush=True)
    train_loader = get_cifar_loader(train=True, num_workers=8)
    val_loader = get_cifar_loader(train=False, num_workers=8, shuffle=False)

    print('=== Standard VGG-A ===', flush=True)
    std_runs, std_best = run_model_family(VGG_A, 'VGG_A', args.lrs, args.epochs,
                                          train_loader, val_loader)
    print('=== VGG-A + BatchNorm ===', flush=True)
    bn_runs, bn_best = run_model_family(VGG_A_BatchNorm, 'VGG_A_BatchNorm',
                                        args.lrs, args.epochs, train_loader, val_loader)

    # persist raw curves
    with open(os.path.join(LOG_DIR, 'bn_runs.json'), 'w') as f:
        json.dump({'std': std_runs, 'bn': bn_runs,
                   'std_best': std_best, 'bn_best': bn_best,
                   'lrs': args.lrs, 'epochs': args.epochs}, f)

    # figures
    plot_loss_landscape(std_runs, bn_runs, 'bn_loss_landscape.png')
    plot_grad_predictiveness(std_runs, bn_runs, 'bn_grad_predictiveness.png')
    cmp_lr = 1e-3 if 1e-3 in args.lrs else args.lrs[0]
    plot_22_comparison(std_runs, bn_runs, cmp_lr, 'bn_vgg_comparison.png')

    print(f'DONE  std_best={std_best:.2f}  bn_best={bn_best:.2f}', flush=True)


if __name__ == '__main__':
    main()

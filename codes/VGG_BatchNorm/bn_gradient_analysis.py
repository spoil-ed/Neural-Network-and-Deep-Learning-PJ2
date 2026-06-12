"""Gradient predictiveness & effective beta-smoothness (Project Part 2, the
measures 2 and 3 of Section 2.3).

Following Santurkar et al. (2018): while training each model with a *single*
optimizer, at every step we probe how the loss and the gradient change as we
move along the current gradient direction by a range of step sizes eta:

    g  = grad L(theta)
    for eta in grid:  theta' = theta - eta * g
        gradient predictiveness  ->  || grad L(theta') - g ||
        effective beta-smoothness -> || grad L(theta') - g || / (eta * ||g||)

A model whose landscape is smoother shows a *smaller, tighter* band of gradient
change. We plot the band over training steps for VGG-A and VGG-A+BN.

Run:  CUDA_VISIBLE_DEVICES=k python bn_gradient_analysis.py --epochs 3
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubstyle
pubstyle.apply()
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(HOME, 'results', 'figures')
LOG_DIR = os.path.join(HOME, 'results', 'logs')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def flat_grad(model):
    return parameters_to_vector([p.grad for p in model.parameters()]).detach()


def probe(model, x, y, criterion, base_lr, etas):
    """Return (max_grad_diff, max_beta) along the gradient direction."""
    params = list(model.parameters())
    theta0 = parameters_to_vector(params).detach().clone()

    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    g = flat_grad(model)
    gnorm = g.norm() + 1e-12

    diffs, betas = [], []
    for eta in etas:
        vector_to_parameters(theta0 - eta * g, params)
        model.zero_grad()
        criterion(model(x), y).backward()
        g_eta = flat_grad(model)
        d = (g_eta - g).norm().item()
        diffs.append(d)
        betas.append(d / (eta * gnorm.item()))
    vector_to_parameters(theta0, params)
    return float(np.max(diffs)), float(np.min(diffs)), float(np.max(betas))


def run(model_cls, name, train_loader, epochs, base_lr, etas):
    torch.manual_seed(2020)
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    crit = nn.CrossEntropyLoss()
    max_band, min_band, beta_curve = [], [], []
    for _ in tqdm(range(epochs), unit='epoch', desc=name):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            dmax, dmin, bmax = probe(model, x, y, crit, base_lr, etas)
            max_band.append(dmax); min_band.append(dmin); beta_curve.append(bmax)
            # real training step (fresh gradient at the restored theta)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    return dict(max_band=max_band, min_band=min_band, beta_curve=beta_curve)


def _smooth(x, w=25):
    x = np.asarray(x, float)
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot(std, bn, fname, w=25):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    smin, smax = _smooth(std['min_band'], w), _smooth(std['max_band'], w)
    bmin, bmax = _smooth(bn['min_band'], w), _smooth(bn['max_band'], w)
    s = np.arange(len(smax)); s2 = np.arange(len(bmax))
    ax.fill_between(s, smin, smax, alpha=0.35, color=pubstyle.GREEN,
                    linewidth=0, label='Standard VGG-A')
    ax.fill_between(s2, bmin, bmax, alpha=0.35, color=pubstyle.RED,
                    linewidth=0, label='VGG-A + BatchNorm')
    ax.plot(s, smax, color=pubstyle.GREEN, lw=0.9)
    ax.plot(s2, bmax, color=pubstyle.RED, lw=0.9)
    ax.set_xlabel('Training step')
    ax.set_ylabel(r'$\|\nabla L(\theta-\eta g)-\nabla L(\theta)\|$')
    ax.legend(); pubstyle.grid(ax)
    plt.tight_layout(pad=1)
    pubstyle.save(FIG_DIR, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--base_lr', type=float, default=1e-3)
    ap.add_argument('--replot', action='store_true')
    args = ap.parse_args()
    etas = [args.base_lr * m for m in (0.5, 1, 2, 5, 10, 20)]

    if args.replot:
        d = json.load(open(os.path.join(LOG_DIR, 'bn_grad.json')))
        plot(d['std'], d['bn'], 'bn_grad_predictiveness.png')
        mdiff_std = float(np.percentile(d['std']['max_band'], 90))
        mdiff_bn = float(np.percentile(d['bn']['max_band'], 90))
        print(f'replotted. max grad-difference (p90): VGG-A={mdiff_std:.2f} '
              f'VGG-A+BN={mdiff_bn:.2f}', flush=True)
        return

    train_loader = get_cifar_loader(train=True, num_workers=8)
    print('=== VGG-A gradient analysis ===', flush=True)
    std = run(VGG_A, 'VGG_A', train_loader, args.epochs, args.base_lr, etas)
    print('=== VGG-A+BN gradient analysis ===', flush=True)
    bn = run(VGG_A_BatchNorm, 'VGG_A_BatchNorm', train_loader, args.epochs, args.base_lr, etas)

    beta_std = float(np.max(std['beta_curve']))
    beta_bn = float(np.max(bn['beta_curve']))
    with open(os.path.join(LOG_DIR, 'bn_grad.json'), 'w') as f:
        json.dump(dict(std=std, bn=bn, beta_std=beta_std, beta_bn=beta_bn,
                       etas=etas, epochs=args.epochs), f)
    plot(std, bn, 'bn_grad_predictiveness.png')
    print(f'DONE  effective_beta_smoothness:  VGG-A={beta_std:.2f}  VGG-A+BN={beta_bn:.2f}', flush=True)


if __name__ == '__main__':
    main()

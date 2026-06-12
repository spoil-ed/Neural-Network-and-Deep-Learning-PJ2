"""Model insights & visualisations for the headline model (Project task 6):
  * first-layer convolutional filters,
  * confusion matrix and per-class accuracy,
  * t-SNE of the penultimate-layer features,
  * a filter-normalised 2D loss surface around the trained minimum.

Run after run_best.py has produced results/models/best.pth.
"""
import copy
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.patheffects
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubstyle
pubstyle.apply()
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from data import get_loaders, CLASSES
from models import build_model
from engine import evaluate

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')
FIG = os.path.join(RESULTS, 'figures')
os.makedirs(FIG, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_best(model_name='resnet18', ckpt='best'):
    model = build_model(model_name)
    state = torch.load(os.path.join(RESULTS, 'models', f'{ckpt}.pth'), map_location='cpu')
    model.load_state_dict(state)
    return model.to(device).eval()


def plot_filters(model):
    w = model.conv1.weight.detach().cpu()        # (64,3,3,3)
    n = w.size(0)
    cols, rows = 8, n // 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        f = w[i]                                  # normalise each filter on its own
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f.permute(1, 2, 0).numpy(), interpolation='nearest')
        ax.axis('off')
    plt.subplots_adjust(wspace=0.08, hspace=0.08,
                        left=0.01, right=0.99, top=0.99, bottom=0.01)
    pubstyle.save(FIG, 'filters')


@torch.no_grad()
def gather_preds(model, loader, max_feats=4000):
    ys, preds, feats = [], [], []
    nf = 0
    for x, y in loader:
        x = x.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.features_forward(x)
            out = model.fc(f) if hasattr(model, 'fc') else model.classifier[-1](f)
        ys.append(y.numpy())
        preds.append(out.argmax(1).cpu().numpy())
        if nf < max_feats:
            feats.append(f.float().cpu().numpy())
            nf += x.size(0)
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(feats)


def plot_confusion_and_perclass(y, pred):
    cm = confusion_matrix(y, pred)
    cm_norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.spines.top.set_visible(True); ax.spines.right.set_visible(True)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{cm_norm[i,j]*100:.0f}', ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=9)
    cb = fig.colorbar(im, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(1.0)
    plt.tight_layout(pad=1)
    pubstyle.save(FIG, 'confusion')

    per_class = 100 * cm.diagonal() / cm.sum(1)
    worst = int(per_class.argmin())
    colors = [pubstyle.RED if i == worst else pubstyle.BLUE for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(CLASSES, per_class, width=0.6, color=colors,
                  edgecolor='black', linewidth=1.0)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(80, 100); ax.grid(False)
    pubstyle.bar_labels(ax, bars, fmt='{:.1f}', dy=0.25, fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=1)
    pubstyle.save(FIG, 'per_class')
    return per_class


def plot_tsne(feats, y, n=2500):
    y = np.asarray(y)[:len(feats)]              # feats hold the first N (unshuffled) samples
    n = min(n, len(feats))
    idx = np.random.RandomState(0).permutation(len(feats))[:n]
    emb = TSNE(n_components=2, init='pca', perplexity=30,
               random_state=0).fit_transform(feats[idx])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = plt.get_cmap('tab10')
    for c in range(10):
        m = y[idx] == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(c), alpha=0.75,
                   linewidths=0, rasterized=True)
        cx, cy = np.median(emb[m, 0]), np.median(emb[m, 1])
        ax.text(cx, cy, CLASSES[c], ha='center', va='center',
                fontweight='bold', color=cmap(c), fontsize=15,
                path_effects=[mpl.patheffects.withStroke(linewidth=2.5,
                                                         foreground='white')])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout(pad=1)
    pubstyle.save(FIG, 'tsne')


# ---- filter-normalised 2D loss surface (Li et al., 2018) ------------------- #
def _rand_dir_like(params, rng):
    return [torch.tensor(rng.standard_normal(p.shape), dtype=torch.float32, device=device)
            for p in params]


def _filter_normalize(direction, weights):
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.zero_()                              # ignore biases / BN params
        else:
            for i in range(d.size(0)):             # per-filter / per-neuron norm
                d[i].mul_(w[i].norm() / (d[i].norm() + 1e-10))


def plot_loss_surface(model, loader, span=1.0, grid=21, n_batches=8):
    model = copy.deepcopy(model)
    params = [p for p in model.parameters()]
    w0 = [p.detach().clone() for p in params]
    rng = np.random.default_rng(0)
    d1, d2 = _rand_dir_like(w0, rng), _rand_dir_like(w0, rng)
    _filter_normalize(d1, w0); _filter_normalize(d2, w0)

    batches = []
    for k, (x, y) in enumerate(loader):
        batches.append((x.to(device), y.to(device)))
        if k + 1 >= n_batches:
            break
    crit = nn.CrossEntropyLoss()

    alphas = np.linspace(-span, span, grid)
    Z = np.zeros((grid, grid))
    model.eval()
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(alphas):
                for p, w, e1, e2 in zip(params, w0, d1, d2):
                    p.copy_(w + a * e1 + b * e2)
                tot, n = 0.0, 0
                for x, y in batches:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        tot += crit(model(x), y).item() * y.size(0)
                    n += y.size(0)
                Z[i, j] = tot / n
    A, B = np.meshgrid(alphas, alphas)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(A, B, np.log(Z), levels=30, cmap='viridis')
    ax.contour(A, B, np.log(Z), levels=15, colors='white',
               linewidths=0.4, alpha=0.8)
    ax.plot(0, 0, marker='*', color='white', markeredgecolor='black',
            markeredgewidth=0.8, markersize=16)
    cb = fig.colorbar(cs, fraction=0.046, pad=0.03)
    cb.set_label('log loss')
    cb.outline.set_linewidth(1.0)
    ax.set_xlabel(r'$\alpha$ (direction 1)'); ax.set_ylabel(r'$\beta$ (direction 2)')
    ax.spines.top.set_visible(True); ax.spines.right.set_visible(True)
    plt.tight_layout(pad=1)
    pubstyle.save(FIG, 'loss_surface')
    np.save(os.path.join(RESULTS, 'logs', 'loss_surface_Z.npy'), Z)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--ckpt', default='best')
    args = ap.parse_args()
    model = load_best(args.model, args.ckpt)
    _, test_loader = get_loaders(batch_size=256, num_workers=8, augment=False)
    acc, _ = evaluate(model, test_loader, device)
    print(f'loaded best model, test_acc={acc:.2f}', flush=True)

    plot_filters(model)
    print('filters done', flush=True)
    y, pred, feats = gather_preds(model, test_loader)
    per_class = plot_confusion_and_perclass(y, pred)
    print('confusion/per-class done; worst:', CLASSES[int(per_class.argmin())],
          f'{per_class.min():.1f}%', flush=True)
    plot_tsne(feats, y)
    print('tsne done', flush=True)
    plot_loss_surface(model, test_loader)
    print('loss surface done', flush=True)


if __name__ == '__main__':
    main()

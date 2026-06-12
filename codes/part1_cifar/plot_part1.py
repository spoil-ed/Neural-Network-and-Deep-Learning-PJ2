"""Build Part-1 figures and a results summary from the saved JSON logs."""
import json
import os

import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubstyle
pubstyle.apply()

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')
LOG = os.path.join(RESULTS, 'logs')
FIG = os.path.join(RESULTS, 'figures')
os.makedirs(FIG, exist_ok=True)


def load(name):
    with open(os.path.join(LOG, name)) as f:
        return json.load(f)


# prose display names for figure labels (raw run names stay in the logs)
DISPLAY = {
    'SGD+nesterov': 'SGD + Nesterov',
    'ManualSGD (5c)': 'hand-written SGD',
    'ManualAdam (5c)': 'hand-written Adam',
    'CE (no wd)': 'CE only',
    'CE + L2 (wd=5e-4)': 'CE + weight decay',
    'CE + label-smooth 0.1': 'CE + label smoothing',
    'CE + dropout 0.5': 'CE + dropout',
    'act=relu': 'ReLU',
    'act=leaky_relu': 'LeakyReLU',
    'act=gelu': 'GELU',
    'act=tanh': 'Tanh',
    'act=sigmoid': 'Sigmoid',
    'width x0.25': 'width 0.25',
    'width x0.5': 'width 0.5',
    'width x1.0': 'width 1.0',
    'width x2.0': 'width 2.0',
    'None (plain)': 'plain',
    '+ BatchNorm': '+BN',
    '+ Dropout': '+Dropout',
    '+ Residual': '+Residual',
    '+ All three': '+All',
    'Baseline BN ConvNet': 'baseline BN',
    'No normalization': 'no norm',
    'GroupNorm': 'GroupNorm',
    'Average pooling': 'avg pool',
    'Deeper stages': 'deeper',
    'Narrow classifier': 'narrow head',
    'Wide classifier': 'wide head',
    'Conv dropout': 'conv dropout',
    'BN + Residual + Dropout': 'BN+res+drop',
    'Standard crop+flip': 'crop+flip',
    'No augmentation': 'no aug',
    'Cutout': 'cutout',
    'No weight decay': 'no wd',
    'Label smoothing': 'label smooth',
    'Step LR': 'step LR',
    'Constant LR': 'constant LR',
    'Combined regularization': 'combined',
}

ABLATION_GROUPS = ['width', 'activation', 'loss', 'optimizer',
                   'optional', 'optional_arch', 'optional_training']


def disp(name):
    return DISPLAY.get(name, name)


def training_curves(best):
    ep = range(len(best['train_acc']))
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3))
    ax[0].plot(ep, best['train_loss'], color=pubstyle.BLUE, label='train')
    ax[0].plot(ep, best['test_loss'], color=pubstyle.RED, label='test')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss')
    ax[0].legend(); pubstyle.grid(ax[0])
    ax[1].plot(ep, best['train_acc'], color=pubstyle.BLUE, label='train')
    ax[1].plot(ep, best['test_acc'], color=pubstyle.RED, label='test')
    ax[1].axhline(best['best_acc'], ls='--', c=pubstyle.GREY, lw=0.8,
                  label=f'best {best["best_acc"]:.2f}%')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy (%)')
    ax[1].legend(loc='lower right')
    pubstyle.grid(ax[1])
    plt.tight_layout(pad=0.5)
    pubstyle.save(FIG, 'best_training_curves')


def best_vs_manual(best, manual):
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.plot(best['test_acc'], color=pubstyle.BLUE, lw=1.5,
            label=f'built-in SGD (best {best["best_acc"]:.2f}%)')
    ax.plot(manual['test_acc'], color=pubstyle.ORANGE, lw=1.2, ls='--',
            label=f'hand-written SGD (best {manual["best_acc"]:.2f}%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test accuracy (%)')
    ax.legend(loc='lower right'); pubstyle.grid(ax)
    plt.tight_layout(pad=0.5)
    pubstyle.save(FIG, 'best_vs_manual')


def ablation_bar(group, ylabel='Best test acc (%)'):
    data = load(f'ablation_{group}.json')
    names = [disp(d['name']) for d in data]
    accs = [d['best_acc'] for d in data]
    best_i = int(np.argmax(accs))
    colors = [pubstyle.BLUE if i == best_i else pubstyle.LIGHTGREY
              for i in range(len(accs))]
    fig_w = max(6.2, 4.0 + 0.35 * len(names))
    fig_h = 2.4 if len(names) > 6 else 2.1
    fig, ax = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                           gridspec_kw={'width_ratios': [1, 1.45]})
    bars = ax[0].bar(range(len(names)), accs, width=0.62, color=colors,
                     edgecolor='black', linewidth=0.8)
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=30, ha='right',
                          fontsize=6.5 if len(names) > 6 else 7)
    ax[0].set_ylabel(ylabel); ax[0].set_ylim(min(accs) - 3, max(accs) + 1.5)
    ax[0].grid(False)
    pubstyle.bar_labels(ax[0], bars, dy=0.12, fontsize=7)
    # visual hierarchy: best config thicker and on top, others lighter
    for i, d in enumerate(data):
        emph = i == best_i
        ax[1].plot(d['test_acc_curve'], lw=1.7 if emph else 1.0,
                   alpha=1.0 if emph else 0.75,
                   zorder=3 if emph else 2, label=disp(d['name']))
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Test acc (%)')
    ax[1].legend(fontsize=6.2 if len(names) > 6 else 6.5,
                 ncol=3 if len(names) > 6 else 2, loc='lower right')
    pubstyle.grid(ax[1])
    plt.tight_layout(pad=0.5)
    pubstyle.save(FIG, f'ablation_{group}')
    return data


def main():
    summary = {}
    try:
        best = load('best.json'); training_curves(best)
        summary['best'] = dict(best_acc=best['best_acc'], n_params=best['n_params'],
                               avg_epoch_time=float(np.mean(best['epoch_time'])))
    except FileNotFoundError:
        best = None
    try:
        manual = load('manual.json')
        if best: best_vs_manual(best, manual)
        summary['manual'] = dict(best_acc=manual['best_acc'])
    except FileNotFoundError:
        pass

    for g in ABLATION_GROUPS:
        try:
            summary[g] = ablation_bar(g)
        except FileNotFoundError:
            print('missing', g)

    # emit a compact text summary
    lines = []
    if 'best' in summary:
        lines.append(f"HEADLINE ResNet-18: best_test_acc={summary['best']['best_acc']:.2f}%  "
                     f"params={summary['best']['n_params']:,}  "
                     f"avg_epoch={summary['best']['avg_epoch_time']:.1f}s")
    if 'manual' in summary:
        lines.append(f"ManualSGD ResNet-18: best_test_acc={summary['manual']['best_acc']:.2f}%")
    for g in ABLATION_GROUPS:
        if g in summary and isinstance(summary[g], list):
            lines.append(f'--- {g} ---')
            for d in summary[g]:
                lines.append(f"  {d['name']:24s} params={d['n_params']:>10,}  "
                             f"best={d['best_acc']:.2f}%  final_train={d['final_train_acc']:.2f}%")
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(LOG, 'part1_summary.txt'), 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()

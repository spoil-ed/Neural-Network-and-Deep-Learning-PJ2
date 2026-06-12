"""Figures for the accuracy-push experiments: a bar chart comparing the five
strong configs and the winner's training curve."""
import json
import os

import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubstyle
pubstyle.apply()

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')
LOG, FIG = os.path.join(RESULTS, 'logs'), os.path.join(RESULTS, 'figures')

TAGS = ['r18_strong', 'wrn_base200', 'wrn_mix200', 'wrn_base300', 'wrn_mix300']
LABELS = ['ResNet-18\n+mix 300', 'WRN-28-10\n200ep', 'WRN-28-10\n+mix 200',
          'WRN-28-10\n300ep', 'WRN-28-10\n+mix 300']


def main():
    data = {t: json.load(open(os.path.join(LOG, f'{t}.json'))) for t in TAGS}
    accs = [max(data[t]['best_acc'], data[t]['tta_acc']) for t in TAGS]

    BASE = 95.34
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.4))
    best_i = max(range(len(accs)), key=lambda i: accs[i])
    colors = [pubstyle.BLUE if i == best_i else pubstyle.LIGHTGREY
              for i in range(len(accs))]
    bars = ax[0].bar(range(len(TAGS)), accs, width=0.62, color=colors,
                     edgecolor='black', linewidth=0.8)
    ax[0].axhline(BASE, ls='--', c=pubstyle.GREY, lw=0.8)
    ax[0].text(0.02, BASE + 0.07, f'baseline {BASE}%', fontsize=7,
               color=pubstyle.GREY, transform=ax[0].get_yaxis_transform())
    ax[0].set_xticks(range(len(TAGS)))
    ax[0].set_xticklabels(LABELS, fontsize=6.5)
    ax[0].set_ylabel('Best test acc (%)'); ax[0].set_ylim(95, 98.7)
    ax[0].grid(False)
    # annotate each bar with its gain over the baseline
    for b, v in zip(bars, accs):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.06,
                   f'+{v - BASE:.2f}', ha='center', va='bottom', fontsize=7)

    win = data['wrn_mix300']
    ax[1].plot(win['test_acc'], label='raw model', color=pubstyle.LIGHTGREY,
               lw=0.9)
    ax[1].plot(win['ema_acc'], label='EMA weights', color=pubstyle.RED, lw=1.5)
    ax[1].axhline(max(win['best_acc'], win['tta_acc']), ls='--',
                  c=pubstyle.GREY, lw=0.8,
                  label=f"best {max(win['best_acc'], win['tta_acc']):.2f}%")
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Test acc (%)')
    ax[1].set_ylim(80, 99)
    ax[1].legend(fontsize=7, loc='lower right'); pubstyle.grid(ax[1])
    plt.tight_layout(pad=0.5)
    pubstyle.save(FIG, 'strong_results')
    print('saved strong_results.png; accs:', dict(zip(TAGS, [round(a, 2) for a in accs])))


if __name__ == '__main__':
    main()

"""Shared publication style for all PJ2 figures.

NeurIPS/Nature-like defaults: colour-blind-safe Okabe-Ito palette, open
spines, subtle y-grids on line plots only, 300-dpi PNG + vector PDF export.
Import and call ``apply()`` once, before any plotting.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette (semantic roles)
BLUE = '#0072B2'      # primary / proposed
ORANGE = '#E69F00'    # secondary
GREEN = '#009E73'     # baseline A (standard VGG)
RED = '#D55E00'       # contrast / BN / EMA
PURPLE = '#CC79A7'
SKY = '#56B4E9'
YELLOW = '#F0E442'
GREY = '#7F7F7F'
LIGHTGREY = '#C4CCD4'
CYCLE = [BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY, YELLOW]


def apply():
    """Figures are drawn at their FINAL printed size (figsize == slot width in
    the report, scale 1:1), so in-figure text is true 8--9 pt, matching the
    document's footnote size like NeurIPS/Nature figures."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': mpl.cycler(color=CYCLE),
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'lines.linewidth': 1.4,
    })


def grid(ax):
    """Subtle horizontal grid for line plots (never on bar plots)."""
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig_dir, name):
    """Save 300-dpi PNG (for the report) plus a vector PDF twin."""
    base = os.path.join(fig_dir, name.rsplit('.', 1)[0])
    plt.savefig(base + '.png')
    plt.savefig(base + '.pdf')
    plt.close()


def bar_labels(ax, bars, fmt='{:.2f}', dy=0.08, fontsize=7.5):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha='center', va='bottom',
                fontsize=fontsize)

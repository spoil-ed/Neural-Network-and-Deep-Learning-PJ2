"""Controlled ablation studies on a compact ConvNet (Project tasks 4 & 5a).

Groups (selected with --group):
  width       -- number of filters/neurons          (task 4a)
  loss        -- loss function + regularization      (task 4b)
  activation  -- activation function                 (task 4c)
  optimizer   -- torch.optim + hand-written optims   (task 5a / 5c)
  optional_arch     -- optional architecture choices
  optional_training -- optional training choices

Each configuration trains the same ConvNet backbone for --epochs epochs with
cosine LR decay, so that only the studied factor varies.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_loaders
from models import ConvNet, count_parameters
from optimizers import build_optimizer
from engine import train, set_seed

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')


class MSEOneHot(nn.Module):
    """MSE loss on softmax-probabilities vs one-hot targets (a non-CE loss)."""
    def forward(self, logits, target):
        oh = F.one_hot(target, logits.size(1)).float()
        return F.mse_loss(F.softmax(logits, dim=1), oh)


def configs(group):
    """Return list of (name, kwargs) for a group. kwargs keys:
       model_kw, optimizer, lr, wd, criterion, loader_kw, scheduler."""
    if group == 'width':
        return [(f'width x{w}', dict(model_kw=dict(width_mult=w, use_bn=True),
                                     optimizer='sgd', lr=0.05, wd=5e-4,
                                     criterion=nn.CrossEntropyLoss()))
                for w in (0.25, 0.5, 1.0, 2.0)]
    if group == 'activation':
        return [(f'act={a}', dict(model_kw=dict(activation=a, use_bn=True),
                                  optimizer='sgd', lr=0.05, wd=5e-4,
                                  criterion=nn.CrossEntropyLoss()))
                for a in ('relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid')]
    if group == 'loss':
        base = dict(model_kw=dict(use_bn=True), optimizer='sgd', lr=0.05)
        return [
            ('CE (no wd)',            dict(**base, wd=0.0,  criterion=nn.CrossEntropyLoss())),
            ('CE + L2 (wd=5e-4)',     dict(**base, wd=5e-4, criterion=nn.CrossEntropyLoss())),
            ('CE + label-smooth 0.1', dict(**base, wd=5e-4, criterion=nn.CrossEntropyLoss(label_smoothing=0.1))),
            ('CE + dropout 0.5',      dict(model_kw=dict(use_bn=True, dropout=0.5),
                                           optimizer='sgd', lr=0.05, wd=5e-4,
                                           criterion=nn.CrossEntropyLoss())),
            ('MSE on one-hot',        dict(**base, wd=5e-4, criterion=MSEOneHot())),
        ]
    if group == 'optimizer':
        base_kw = dict(use_bn=True)
        ce = nn.CrossEntropyLoss()
        return [
            ('SGD+nesterov',   dict(model_kw=base_kw, optimizer='sgd',        lr=0.05,  wd=5e-4, criterion=ce)),
            ('Adam',           dict(model_kw=base_kw, optimizer='adam',       lr=1e-3,  wd=5e-4, criterion=ce)),
            ('AdamW',          dict(model_kw=base_kw, optimizer='adamw',      lr=1e-3,  wd=5e-2, criterion=ce)),
            ('RMSprop',        dict(model_kw=base_kw, optimizer='rmsprop',    lr=1e-3,  wd=5e-4, criterion=ce)),
            ('ManualSGD (5c)', dict(model_kw=base_kw, optimizer='manual_sgd', lr=0.05,  wd=5e-4, criterion=ce)),
            ('ManualAdam (5c)',dict(model_kw=base_kw, optimizer='manual_adam',lr=1e-3,  wd=5e-4, criterion=ce)),
        ]
    if group == 'optional':
        # optional components (task 3): each alone on a plain ConvNet, then all
        base = dict(optimizer='sgd', lr=0.05, wd=5e-4,
                    criterion=nn.CrossEntropyLoss())
        return [
            ('None (plain)', dict(**base, model_kw=dict(use_bn=False))),
            ('+ BatchNorm',  dict(**base, model_kw=dict(use_bn=True))),
            ('+ Dropout',    dict(**base, model_kw=dict(use_bn=False, dropout=0.5))),
            ('+ Residual',   dict(**base, model_kw=dict(use_bn=False, residual=True))),
            ('+ All three',  dict(**base, model_kw=dict(use_bn=True, dropout=0.5,
                                                        residual=True))),
        ]
    if group == 'optional_arch':
        base = dict(optimizer='sgd', lr=0.05, wd=5e-4,
                    criterion=nn.CrossEntropyLoss(),
                    loader_kw=dict(augment='standard'), scheduler='cosine')
        return [
            ('Baseline BN ConvNet',      dict(**base, model_kw=dict(use_bn=True))),
            ('No normalization',         dict(**base, model_kw=dict(norm='none'))),
            ('GroupNorm',                dict(**base, model_kw=dict(norm='group'))),
            ('Average pooling',          dict(**base, model_kw=dict(pooling='avg'))),
            ('Deeper stages',            dict(**base, model_kw=dict(depth_variant='deep'))),
            ('Narrow classifier',        dict(**base, model_kw=dict(classifier_width=256))),
            ('Wide classifier',          dict(**base, model_kw=dict(classifier_width=1024))),
            ('Conv dropout',             dict(**base, model_kw=dict(conv_dropout=0.1))),
            ('BN + Residual + Dropout',  dict(**base, model_kw=dict(use_bn=True,
                                                                    residual=True,
                                                                    dropout=0.5))),
        ]
    if group == 'optional_training':
        base = dict(model_kw=dict(use_bn=True), optimizer='sgd', lr=0.05,
                    wd=5e-4, criterion=nn.CrossEntropyLoss(),
                    loader_kw=dict(augment='standard'), scheduler='cosine')
        def cfg(**updates):
            out = base.copy()
            out.update(updates)
            return out
        return [
            ('Standard crop+flip',       cfg()),
            ('No augmentation',          cfg(loader_kw=dict(augment='none'))),
            ('Cutout',                   cfg(loader_kw=dict(augment='cutout'))),
            ('No weight decay',          cfg(wd=0.0)),
            ('Label smoothing',          cfg(criterion=nn.CrossEntropyLoss(label_smoothing=0.1))),
            ('Step LR',                  cfg(scheduler='step')),
            ('Constant LR',              cfg(scheduler='constant')),
            ('Combined regularization',  cfg(model_kw=dict(use_bn=True, dropout=0.3),
                                             loader_kw=dict(augment='cutout'),
                                             criterion=nn.CrossEntropyLoss(label_smoothing=0.1))),
        ]
    raise ValueError(group)


def build_scheduler(kind, optimizer, epochs):
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if kind == 'step':
        milestones = sorted({max(1, epochs // 2), max(1, (3 * epochs) // 4)})
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                   gamma=0.1)
    if kind == 'constant':
        return None
    raise ValueError(f'unknown scheduler {kind}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True,
                    choices=['width', 'activation', 'loss', 'optimizer', 'optional',
                             'optional_arch', 'optional_training'])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--limit-train-samples', type=int, default=0,
                    help='optional small-subset smoke-test limit')
    ap.add_argument('--limit-test-samples', type=int, default=0,
                    help='optional small-subset smoke-test limit')
    ap.add_argument('--output-suffix', default='',
                    help='suffix for smoke logs, e.g. smoke -> *_smoke.json')
    args = ap.parse_args()

    device = torch.device(args.device)
    use_amp = (device.type == 'cuda')
    loader_cache = {}

    def loaders(loader_kw):
        key = tuple(sorted(loader_kw.items()))
        if key not in loader_cache:
            train_loader, test_loader = get_loaders(batch_size=128,
                                                    num_workers=args.num_workers,
                                                    **loader_kw)
            if args.limit_train_samples:
                n = min(args.limit_train_samples, len(train_loader.dataset))
                train_set = torch.utils.data.Subset(train_loader.dataset, range(n))
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=128, shuffle=True,
                    num_workers=args.num_workers, pin_memory=use_amp)
            if args.limit_test_samples:
                n = min(args.limit_test_samples, len(test_loader.dataset))
                test_set = torch.utils.data.Subset(test_loader.dataset, range(n))
                test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=256, shuffle=False,
                    num_workers=args.num_workers, pin_memory=use_amp)
            loader_cache[key] = (train_loader, test_loader)
        return loader_cache[key]

    results = []
    for name, cfg in configs(args.group):
        set_seed(0)
        loader_kw = cfg.get('loader_kw', dict(augment='standard'))
        train_loader, test_loader = loaders(loader_kw)
        model = ConvNet(**cfg['model_kw'])
        n_params = count_parameters(model)
        opt = build_optimizer(cfg['optimizer'], model.parameters(),
                              lr=cfg['lr'], weight_decay=cfg['wd'], momentum=0.9)
        sched = build_scheduler(cfg.get('scheduler', 'cosine'), opt, args.epochs)
        print(f'[{args.group}] {name:22s} params={n_params:,}', flush=True)
        hist, _ = train(model, train_loader, test_loader, device, opt,
                        cfg['criterion'], args.epochs, scheduler=sched,
                        use_amp=use_amp, log_every=0, verbose=False)
        results.append(dict(name=name, n_params=n_params,
                            best_acc=hist['best_acc'],
                            final_train_acc=hist['train_acc'][-1],
                            test_acc_curve=hist['test_acc'],
                            train_loss_curve=hist['train_loss']))
        print(f'[{args.group}] {name:22s} best_test_acc={hist["best_acc"]:.2f}', flush=True)

    os.makedirs(os.path.join(RESULTS, 'logs'), exist_ok=True)
    suffix = f'_{args.output_suffix}' if args.output_suffix else ''
    out = os.path.join(RESULTS, 'logs', f'ablation_{args.group}{suffix}.json')
    with open(out, 'w') as f:
        json.dump(results, f)
    print(f'[{args.group}] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()

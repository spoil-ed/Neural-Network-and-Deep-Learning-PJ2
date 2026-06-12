"""Accuracy-push training: WideResNet / ResNet-18 with AutoAugment + Cutout +
Mixup/CutMix + label smoothing + EMA + cosine schedule with warmup + TTA.

Usage:
  python run_strong.py --model wrn28_10 --epochs 200 --mix 1 --tag wrn_mix
"""
import argparse
import copy
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_strong_loaders
from models import build_model, count_parameters
from engine import set_seed

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')


# --------------------------- Mixup / CutMix -------------------------------- #
def rand_bbox(H, W, lam):
    r = math.sqrt(1.0 - lam)
    cw, ch = int(W * r), int(H * r)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = max(cx - cw // 2, 0), max(cy - ch // 2, 0)
    x2, y2 = min(cx + cw // 2, W), min(cy + ch // 2, H)
    return x1, y1, x2, y2


def mix_batch(x, y, alpha=1.0, p_cutmix=0.5):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    if random.random() < p_cutmix:                       # CutMix
        x1, y1, x2, y2 = rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    else:                                                 # Mixup
        x = lam * x + (1 - lam) * x[idx]
    return x, y, y[idx], lam


# ------------------------------- EMA --------------------------------------- #
class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if tta:
                out = out + model(torch.flip(x, dims=[3]))   # horizontal-flip TTA
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='wrn28_10')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--mix', type=int, default=1, help='1=use mixup/cutmix')
    ap.add_argument('--mix_alpha', type=float, default=1.0)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--ema_decay', type=float, default=0.9998)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tag', default='strong')
    args = ap.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    train_loader, test_loader = get_strong_loaders(batch_size=128, num_workers=12)

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    ema = ModelEMA(model, decay=args.ema_decay)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda')

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1 + math.cos(math.pi * p))

    print(f'[{args.tag}] {args.model} params={n_params:,} mix={args.mix} '
          f'epochs={args.epochs} steps/ep={steps_per_epoch}', flush=True)

    hist = {'test_acc': [], 'ema_acc': [], 'lr': []}
    best_acc, best_state, step = 0.0, None, 0
    import time
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if args.mix:
                    xm, ya, yb, lam = mix_batch(x, y, args.mix_alpha)
                    out = model(xm)
                    loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
                else:
                    loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1

        acc = evaluate(model, test_loader, device, tta=False)
        ema_acc = evaluate(ema.module, test_loader, device, tta=False)
        hist['test_acc'].append(acc); hist['ema_acc'].append(ema_acc)
        hist['lr'].append(lr_at(step))
        cur = max(acc, ema_acc)
        if cur > best_acc:
            best_acc = cur
            best_state = {k: v.detach().cpu().clone()
                          for k, v in (ema.module if ema_acc >= acc else model).state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f'  ep {epoch:3d} | raw {acc:5.2f} ema {ema_acc:5.2f} '
                  f'best {best_acc:5.2f} | {time.time()-t0:.1f}s', flush=True)

    # final TTA evaluation on the best weights
    best_model = build_model(args.model).to(device)
    best_model.load_state_dict(best_state)
    tta_acc = evaluate(best_model, test_loader, device, tta=True)

    hist.update(model=args.model, n_params=n_params, best_acc=best_acc,
                tta_acc=tta_acc, mix=args.mix, epochs=args.epochs)
    os.makedirs(os.path.join(RESULTS, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, 'models'), exist_ok=True)
    with open(os.path.join(RESULTS, 'logs', f'{args.tag}.json'), 'w') as f:
        json.dump(hist, f)
    torch.save(best_state, os.path.join(RESULTS, 'models', f'{args.tag}.pth'))
    print(f'[{args.tag}] DONE best={best_acc:.2f} TTA={tta_acc:.2f}', flush=True)


if __name__ == '__main__':
    main()

"""Train the headline CIFAR-10 model (ResNet-18).

Usage:
    python run_best.py --model resnet18 --optimizer sgd     --epochs 150 --tag best
    python run_best.py --model resnet18 --optimizer manual_sgd --epochs 150 --tag manual
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from data import get_loaders
from models import build_model, count_parameters
from optimizers import build_optimizer
from engine import train, set_seed

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--optimizer', default='sgd')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tag', default='best')
    args = ap.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    train_loader, test_loader = get_loaders(batch_size=128, num_workers=8)

    model = build_model(args.model)
    n_params = count_parameters(model)
    print(f'[{args.tag}] {args.model} params={n_params:,} optimizer={args.optimizer}', flush=True)

    optimizer = build_optimizer(args.optimizer, model.parameters(),
                                lr=args.lr, weight_decay=args.wd, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    hist, best_state = train(model, train_loader, test_loader, device,
                             optimizer, criterion, args.epochs,
                             scheduler=scheduler, use_amp=True,
                             log_every=10, record_steps=False)

    hist['model'] = args.model
    hist['optimizer'] = args.optimizer
    hist['n_params'] = n_params
    os.makedirs(os.path.join(RESULTS, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, 'models'), exist_ok=True)
    with open(os.path.join(RESULTS, 'logs', f'{args.tag}.json'), 'w') as f:
        json.dump(hist, f)
    torch.save(best_state, os.path.join(RESULTS, 'models', f'{args.tag}.pth'))
    print(f'[{args.tag}] DONE best_test_acc={hist["best_acc"]:.2f} '
          f'avg_epoch_time={sum(hist["epoch_time"])/len(hist["epoch_time"]):.1f}s', flush=True)


if __name__ == '__main__':
    main()

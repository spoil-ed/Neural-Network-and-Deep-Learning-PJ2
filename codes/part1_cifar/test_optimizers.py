"""Validate the hand-written optimizers against torch.optim.

We train a tiny MLP on a fixed synthetic problem with both the manual and the
reference optimizer from identical initial weights, and check the loss
trajectories coincide to numerical precision. Prints a small table consumed by
the report.
"""
import copy
import torch
import torch.nn as nn

from optimizers import ManualSGD, ManualAdam


def make_problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(512, 20, generator=g)
    w = torch.randn(20, 1, generator=g)
    y = (X @ w + 0.1 * torch.randn(512, 1, generator=g))
    net = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 1))
    return X, y, net


def run(opt_factory, net, X, y, steps=200):
    net = copy.deepcopy(net)
    opt = opt_factory(net.parameters())
    lossf = nn.MSELoss()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(net(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    X, y, net = make_problem()
    cases = [
        ('SGD+mom+nesterov',
         lambda p: ManualSGD(p, lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True),
         lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)),
        ('Adam',
         lambda p: ManualAdam(p, lr=1e-2, weight_decay=0.0),
         lambda p: torch.optim.Adam(p, lr=1e-2)),
        ('AdamW (decoupled wd)',
         lambda p: ManualAdam(p, lr=1e-2, weight_decay=1e-2, decoupled=True),
         lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=1e-2)),
    ]
    print(f'{"case":24s} {"manual_final":>14s} {"torch_final":>14s} {"max|diff|":>12s}')
    for name, manual, ref in cases:
        lm = run(manual, net, X, y)
        lr = run(ref, net, X, y)
        max_diff = max(abs(a - b) for a, b in zip(lm, lr))
        print(f'{name:24s} {lm[-1]:14.6e} {lr[-1]:14.6e} {max_diff:12.3e}')


if __name__ == '__main__':
    main()

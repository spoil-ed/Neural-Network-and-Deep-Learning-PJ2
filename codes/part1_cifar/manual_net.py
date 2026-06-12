"""Task 5(b): a ConvNet built from raw tensor operations only.

The network uses none of torch.nn: convolution is im2col (Tensor.unfold)
followed by matmul, pooling is unfold + amax, the fully connected layers are
matmul + add, the activation is an elementwise maximum against zero, and the
cross-entropy loss is written with logsumexp. Parameters are plain
requires_grad tensors handed to torch.optim.SGD, as the task requires.

Run with --selftest to check every manual operation against the torch.nn
reference implementation before training.
"""
import argparse
import json
import math
import os
import time

import torch

from data import get_loaders

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results', 'logs')


# ----------------------------------------------------------------------------
# Raw-tensor building blocks (no torch.nn anywhere)
# ----------------------------------------------------------------------------
def conv2d(x, w, b, pad=1):
    """3x3 convolution as im2col + matmul. x: (N,C,H,W), w: (O,C,k,k)."""
    n, c, h, _ = x.shape
    o, _, k, _ = w.shape
    if pad:
        xp = torch.zeros(n, c, h + 2 * pad, h + 2 * pad,
                         device=x.device, dtype=x.dtype)
        xp[:, :, pad:h + pad, pad:h + pad] = x
    else:
        xp = x
    ho = xp.shape[2] - k + 1
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)          # N,C,Ho,Wo,k,k
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n, ho * ho, c * k * k)
    out = torch.matmul(patches, w.reshape(o, -1).t()) + b  # N,Ho*Wo,O
    return out.permute(0, 2, 1).reshape(n, o, ho, ho)


def maxpool2(x):
    """2x2 max-pooling with stride 2 via unfold + amax."""
    return x.unfold(2, 2, 2).unfold(3, 2, 2).amax(dim=(-1, -2))


def relu(x):
    return torch.maximum(x, torch.zeros((), device=x.device, dtype=x.dtype))


def cross_entropy(logits, y):
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    lse = torch.logsumexp(logits, dim=1)
    true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    return (lse - true_logit).mean()


class ManualConvNet:
    """Conv(3->64) - pool - Conv(64->128) - pool - Conv(128->256) - pool
    - FC(4096->256) - ReLU - FC(256->10), all from raw tensor ops."""

    def __init__(self, device):
        def kaiming(*shape, fan_in):
            return torch.randn(*shape, device=device) * math.sqrt(2.0 / fan_in)

        self.w1 = kaiming(64, 3, 3, 3, fan_in=3 * 9)
        self.w2 = kaiming(128, 64, 3, 3, fan_in=64 * 9)
        self.w3 = kaiming(256, 128, 3, 3, fan_in=128 * 9)
        self.wf1 = kaiming(256 * 4 * 4, 256, fan_in=256 * 4 * 4)
        self.wf2 = kaiming(256, 10, fan_in=256)
        self.b1 = torch.zeros(64, device=device)
        self.b2 = torch.zeros(128, device=device)
        self.b3 = torch.zeros(256, device=device)
        self.bf1 = torch.zeros(256, device=device)
        self.bf2 = torch.zeros(10, device=device)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                       self.wf1, self.bf1, self.wf2, self.bf2]
        for p in self.params:
            p.requires_grad_(True)

    def __call__(self, x):
        x = maxpool2(relu(conv2d(x, self.w1, self.b1)))   # 32 -> 16
        x = maxpool2(relu(conv2d(x, self.w2, self.b2)))   # 16 -> 8
        x = maxpool2(relu(conv2d(x, self.w3, self.b3)))   # 8  -> 4
        x = x.reshape(x.shape[0], -1)
        x = relu(torch.matmul(x, self.wf1) + self.bf1)
        return torch.matmul(x, self.wf2) + self.bf2

    def n_params(self):
        return sum(p.numel() for p in self.params)


# ----------------------------------------------------------------------------
# Self-test: every manual op must match the torch.nn reference
# ----------------------------------------------------------------------------
def selftest():
    import torch.nn.functional as F  # reference implementation only
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8, dtype=torch.float64)
    w = torch.randn(5, 3, 3, 3, dtype=torch.float64)
    b = torch.randn(5, dtype=torch.float64)
    checks = {
        'conv2d': (conv2d(x, w, b), F.conv2d(x, w, b, padding=1)),
        'maxpool2': (maxpool2(x), F.max_pool2d(x, 2)),
        'relu': (relu(x), F.relu(x)),
        'cross_entropy': (cross_entropy(torch.randn(7, 10), torch.arange(7) % 10),
                          None),
    }
    logits = torch.randn(7, 10, dtype=torch.float64)
    y = torch.arange(7) % 10
    checks['cross_entropy'] = (cross_entropy(logits, y),
                               F.cross_entropy(logits, y))
    for name, (ours, ref) in checks.items():
        err = (ours - ref).abs().max().item()
        print(f'  {name:14s} max |diff| = {err:.2e}')
        assert err < 1e-12, name
    print('self-test passed: all manual ops match torch.nn references')


# ----------------------------------------------------------------------------
# Training (torch.optim.SGD on the raw parameter tensors)
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, loader, device):
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main(epochs, lr):
    device = 'cuda'
    torch.manual_seed(0)
    train_loader, test_loader = get_loaders()
    net = ManualConvNet(device)
    print(f'ManualConvNet parameters: {net.n_params():,}')
    opt = torch.optim.SGD(net.params, lr=lr, momentum=0.9, nesterov=True,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {'train_loss': [], 'train_acc': [], 'test_acc': [],
            'epoch_time': [], 'best_acc': 0.0}
    for epoch in range(epochs):
        t0 = time.time()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = net(x)
            loss = cross_entropy(out, y)
            loss.backward()
            # manual gradient clipping (raw tensor ops): without normalization
            # layers the early gradients are large enough to diverge otherwise
            gnorm = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.params))
            if gnorm > 5.0:
                for p in net.params:
                    p.grad.mul_(5.0 / gnorm)
            opt.step()
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()
        test_acc = evaluate(net, test_loader, device)
        hist['train_loss'].append(run_loss / total)
        hist['train_acc'].append(100.0 * correct / total)
        hist['test_acc'].append(test_acc)
        hist['epoch_time'].append(time.time() - t0)
        hist['best_acc'] = max(hist['best_acc'], test_acc)
        print(f'  epoch {epoch:3d} | train_loss {hist["train_loss"][-1]:.3f} '
              f'train_acc {hist["train_acc"][-1]:5.2f} test_acc {test_acc:5.2f} '
              f'best {hist["best_acc"]:5.2f} | {hist["epoch_time"][-1]:.1f}s',
              flush=True)

    hist['n_params'] = net.n_params()
    os.makedirs(LOG, exist_ok=True)
    with open(os.path.join(LOG, 'manual_net.json'), 'w') as f:
        json.dump(hist, f)
    print(f'best test acc: {hist["best_acc"]:.2f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        main(args.epochs, args.lr)

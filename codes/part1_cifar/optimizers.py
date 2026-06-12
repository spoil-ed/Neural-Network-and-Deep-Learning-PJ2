"""Hand-written optimizers (Project task 5c: 'implement an optimizer for your
full model by yourself').

The parameter-update mathematics below is written out explicitly with raw
tensor operations -- we do *not* call any of torch.optim's update routines.
We subclass ``torch.optim.Optimizer`` only to reuse its parameter-group /
per-parameter state bookkeeping, which is just a dictionary container.

Both optimizers are validated against their torch.optim counterparts in
``test_optimizers.py`` (the curves overlap to numerical precision).
"""
import torch
from torch.optim.optimizer import Optimizer


class ManualSGD(Optimizer):
    """SGD with momentum / Nesterov / L2 weight decay, written by hand."""

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0,
                 nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group['lr'], group['momentum']
            wd, nesterov = group['weight_decay'], group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)          # L2 regularization
                if mom != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = torch.clone(g).detach()
                        state['momentum_buffer'] = buf
                    else:
                        buf.mul_(mom).add_(g)        # v <- mom*v + g
                    g = g.add(buf, alpha=mom) if nesterov else buf
                p.add_(g, alpha=-lr)                 # theta <- theta - lr*g
        return loss


class ManualAdam(Optimizer):
    """Adam (Kingma & Ba, 2015) with optional decoupled weight decay (AdamW),
    written out explicitly."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, decoupled=decoupled)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2) = group['lr'], group['betas']
            eps, wd, decoupled = group['eps'], group['weight_decay'], group['decoupled']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0 and not decoupled:
                    g = g.add(p, alpha=wd)           # coupled L2

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                m.mul_(b1).add_(g, alpha=1 - b1)         # 1st moment
                v.mul_(b2).addcmul_(g, g, value=1 - b2)  # 2nd moment
                m_hat = m / (1 - b1 ** t)                # bias correction
                v_hat = v / (1 - b2 ** t)
                update = m_hat / (v_hat.sqrt() + eps)

                if wd != 0 and decoupled:
                    p.add_(p, alpha=-lr * wd)            # decoupled decay (AdamW)
                p.add_(update, alpha=-lr)
        return loss


def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    """Factory used by the ablation runner. Names prefixed 'manual_' use the
    hand-written implementations above; the rest use torch.optim."""
    name = name.lower()
    if name == 'manual_sgd':
        return ManualSGD(params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=True)
    if name == 'manual_adam':
        return ManualAdam(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=True)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay,
                                   momentum=momentum)
    raise ValueError(f'unknown optimizer {name}')

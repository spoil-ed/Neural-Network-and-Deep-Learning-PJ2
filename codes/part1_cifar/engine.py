"""Training / evaluation engine shared by all Part-1 experiments."""
import time
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    amp_enabled = (device.type == 'cuda')
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            out = model(x)
            if criterion is not None:
                loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / total
    return acc, (loss_sum / total if criterion is not None else None)


def train(model, train_loader, test_loader, device,
          optimizer, criterion, epochs, scheduler=None,
          use_amp=True, log_every=0, record_steps=False, verbose=True):
    """Train ``model`` and return a history dict.

    If ``record_steps`` is True we also store the per-step training loss, which
    the Part-2 loss-landscape analysis consumes.
    """
    model.to(device)
    hist = {'train_loss': [], 'train_acc': [], 'test_acc': [],
            'test_loss': [], 'lr': [], 'step_loss': [], 'epoch_time': []}
    best_acc, best_state = 0.0, None
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            if record_steps:
                hist['step_loss'].append(loss.item())
        if scheduler is not None:
            scheduler.step()

        train_loss = run_loss / total
        train_acc = 100.0 * correct / total
        test_acc, test_loss = evaluate(model, test_loader, device, criterion)
        hist['train_loss'].append(train_loss)
        hist['train_acc'].append(train_acc)
        hist['test_acc'].append(test_acc)
        hist['test_loss'].append(test_loss)
        hist['lr'].append(optimizer.param_groups[0]['lr'])
        hist['epoch_time'].append(time.time() - t0)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose and (log_every and (epoch % log_every == 0 or epoch == epochs - 1)):
            print(f'  epoch {epoch:3d} | train_loss {train_loss:.3f} '
                  f'train_acc {train_acc:5.2f} test_acc {test_acc:5.2f} '
                  f'best {best_acc:5.2f} | {hist["epoch_time"][-1]:.1f}s',
                  flush=True)

    hist['best_acc'] = best_acc
    return hist, best_state

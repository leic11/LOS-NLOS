from __future__ import annotations

import torch
from tqdm import tqdm


def _update_binary_confusion(labels, preds):
    tp = fp = tn = fn = 0
    for t, p in zip(labels, preds):
        t = int(t.item())
        p = int(p.item())
        if t == 1 and p == 1:
            tp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _metrics_from_confusion(tp, fp, tn, fn, correct, total_samples):
    acc = correct / max(1, total_samples)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1


def train_one_epoch(model, dataset_loader, optimizer, criterion, device, lambda_reg=0.009):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    tp = fp = tn = fn = 0

    pbar = tqdm(dataset_loader, desc="Training", leave=False)
    for batch in pbar:
        time_data = batch["time_history"].to(device)
        space_data = batch["space_global"].to(device)
        labels = batch["labels"].to(device)
        space_mask = space_data.abs().sum(dim=2) == 0

        time_data = time_data + torch.randn_like(time_data) * 0.01
        space_data = space_data + torch.randn_like(space_data) * 0.01

        optimizer.zero_grad()
        logits, reg_loss = model(time_data, space_data, space_mask)
        loss = criterion(logits, labels) + lambda_reg * reg_loss
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        d_tp, d_fp, d_tn, d_fn = _update_binary_confusion(labels, preds)
        tp += d_tp
        fp += d_fp
        tn += d_tn
        fn += d_fn
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, total_samples)
    acc, prec, rec, f1 = _metrics_from_confusion(tp, fp, tn, fn, correct, total_samples)
    return avg_loss, acc, prec, rec, f1


def evaluate(model, dataset_loader, criterion, device, lambda_reg=0.005):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        pbar = tqdm(dataset_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            time_data = batch["time_history"].to(device)
            space_data = batch["space_global"].to(device)
            labels = batch["labels"].to(device)
            space_mask = space_data.abs().sum(dim=2) == 0

            logits, reg_loss = model(time_data, space_data, space_mask)
            loss = criterion(logits, labels) + lambda_reg * reg_loss

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            d_tp, d_fp, d_tn, d_fn = _update_binary_confusion(labels, preds)
            tp += d_tp
            fp += d_fp
            tn += d_tn
            fn += d_fn
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, total_samples)
    acc, prec, rec, f1 = _metrics_from_confusion(tp, fp, tn, fn, correct, total_samples)
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    return avg_loss, acc, prec, rec, f1

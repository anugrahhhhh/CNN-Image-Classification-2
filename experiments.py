# ConvNet architecture experiments on CIFAR-10
# Follows the pipeline of "colab img_clf.ipynb": same data split (train=data_batch_1,
# val=data_batch_2, test=test_batch), sigmoid output + BCE loss, SGD(lr=0.0025, momentum=0.9),
# batch size 10. Only the ConvNet architecture is changed between experiments.
import json
import os
import pickle as pk
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar-10-batches-py") + "/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/"
EPOCHS = 40
SEED = 0
torch.set_num_threads(8)


def unpickle(file):
    with open(file, "rb") as fo:
        return pk.load(fo, encoding="bytes")


def load(batch_files, red_only=False):
    labels, data = [], []
    for f in batch_files:
        d = unpickle(ROOT + f)
        labels.append(np.array(d[b"labels"]))
        data.append(d[b"data"])
    labels = np.concatenate(labels)
    data = np.concatenate(data).reshape(-1, 3, 32, 32)  # C x H x W (R, G, B planes)
    if red_only:  # reproduces the notebook's myData bug: R plane copied into all 3 channels
        data = np.repeat(data[:, :1], 3, axis=1)
    x = torch.from_numpy(data.copy()).float()
    y = torch.from_numpy(labels).long()
    return x, y


class ConvNet(nn.Module):
    """Generalised version of the notebook ConvNet: N x conv_block -> 3 x fc_block."""

    def __init__(self, channels=(16, 32, 64), kernel=3, padding=0, bn=False,
                 dropout=0.0, head="gap", fc=(128, 64)):
        super().__init__()
        layers, c_in, size = [], 3, 32
        for c in channels:
            layers.append(nn.Conv2d(c_in, c, kernel_size=kernel, padding=padding))
            if bn:
                layers.append(nn.BatchNorm2d(c))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            size = (size - kernel + 1 + 2 * padding) // 2
            c_in = c
        self.features = nn.Sequential(*layers)
        if head == "gap":
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            n_in = c_in
        else:
            self.pool = nn.Flatten()
            n_in = c_in * size * size
        fcs = []
        for h in fc:
            fcs += [nn.Linear(n_in, h), nn.ReLU()]
            if dropout:
                fcs.append(nn.Dropout(dropout))
            n_in = h
        fcs += [nn.Linear(n_in, 10), nn.Sigmoid()]
        self.classifier = nn.Sequential(*fcs)

    def forward(self, x):
        x = x / 255
        return self.classifier(self.pool(self.features(x)))


EXPERIMENTS = {
    "E0_notebook_bug":  dict(arch={}, red_only=True,
                             desc="Notebook as-is (loader copies R channel to G,B)"),
    "E1_baseline":      dict(arch={}, desc="Baseline 3 conv (16-32-64) k3, GAP, FC 128-64-10"),
    "E2_wider":         dict(arch=dict(channels=(32, 64, 128)), desc="Wider: 32-64-128 filters"),
    "E3_deeper":        dict(arch=dict(channels=(16, 32, 64, 128), padding=1),
                             desc="Deeper: 4 conv blocks (16-32-64-128), padding=1"),
    "E4_kernel5":       dict(arch=dict(kernel=5, padding=2), desc="Kernel 5x5 (padding=2)"),
    "E5_flatten":       dict(arch=dict(padding=1, head="flatten"),
                             desc="Flatten 64x4x4 instead of global avg pool"),
    "E6_batchnorm":     dict(arch=dict(bn=True), desc="Baseline + BatchNorm"),
    "E7_bn_wide_drop":  dict(arch=dict(channels=(32, 64, 128), padding=1, bn=True,
                                       head="flatten", dropout=0.3),
                             desc="Wide + pad + BN + Flatten + Dropout 0.3"),
}


def run(name, cfg):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    red_only = cfg.get("red_only", False)
    xtr, ytr = load(["data_batch_1"], red_only)
    xva, yva = load(["data_batch_2"], red_only)
    xte, yte = load(["test_batch"], red_only)
    g = torch.Generator().manual_seed(SEED)
    dl_tr = DataLoader(TensorDataset(xtr, ytr), batch_size=10, shuffle=True, generator=g)
    dl_va = DataLoader(TensorDataset(xva, yva), batch_size=100)
    dl_te = DataLoader(TensorDataset(xte, yte), batch_size=100)

    model = ConvNet(**cfg["arch"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = optim.SGD(model.parameters(), lr=0.0025, momentum=0.9)

    def evaluate(dl):
        model.eval()
        tot_loss, correct, preds = 0.0, 0, []
        with torch.no_grad():
            for x, y in dl:
                p = model(x)
                tot_loss += F.binary_cross_entropy(p, F.one_hot(y, 10).float(), reduction="sum").item() / 10
                pr = p.argmax(1)
                correct += (pr == y).sum().item()
                preds.append(pr)
        return tot_loss / len(dl.dataset), correct / len(dl.dataset), torch.cat(preds)

    log, best_acc, best_state, t_train = [], -1, None, 0.0
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        tl, n = 0.0, 0
        for x, y in dl_tr:
            p = model(x)
            loss = F.binary_cross_entropy(p, F.one_hot(y, 10).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            n += len(y)
        dt = time.time() - t0
        t_train += dt
        _, tr_acc, _ = evaluate(DataLoader(TensorDataset(xtr, ytr), batch_size=100))
        vl, va, _ = evaluate(dl_va)
        log.append(dict(epoch=ep + 1, train_loss=tl / n, train_acc=tr_acc, val_loss=vl, val_acc=va, time=dt))
        print(f"{name} ep{ep+1} train_loss={tl/n:.4f} train_acc={tr_acc:.3f} val_loss={vl:.4f} val_acc={va:.3f} {dt:.1f}s", flush=True)
        if va > best_acc:
            best_acc, best_ep = va, ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te_loss, te_acc, preds = evaluate(dl_te)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(yte.numpy(), preds.numpy()):
        cm[t, p] += 1
    # single-image inference latency (batch=1, as in the notebook test loop)
    model.eval()
    with torch.no_grad():
        for i in range(20):
            model(xte[i:i + 1])
        t0 = time.time()
        for i in range(500):
            model(xte[i:i + 1])
        lat_ms = (time.time() - t0) / 500 * 1000

    res = dict(name=name, desc=cfg["desc"], params=n_params, best_epoch=best_ep, best_val_acc=best_acc,
               test_acc=te_acc, test_loss=te_loss, train_time_s=t_train, latency_ms=lat_ms,
               per_class=(cm.diagonal() / cm.sum(1)).tolist(), cm=cm.tolist(), log=log)
    with open(OUT + name + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"== {name}: params={n_params} best_ep={best_ep} val={best_acc:.4f} test={te_acc:.4f} "
          f"train_time={t_train:.0f}s latency={lat_ms:.2f}ms", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(EXPERIMENTS)
    for nm in names:
        run(nm, EXPERIMENTS[nm])

# Builds report figures from results/*.json
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json")))}
FIG = os.path.join(HERE, "results", "fig")
os.makedirs(FIG, exist_ok=True)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2 = "#9b9a94", "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ebeae6",
                     "grid.linewidth": 0.6, "legend.frameon": False})

best = max((k for k in R if k != "E0_notebook_bug"), key=lambda k: R[k]["test_acc"])
short = {k: k.split("_", 1)[0] for k in R}

# Fig 1: validation accuracy curves (4 key runs) + test accuracy bars (all runs)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
sel = [("E0_notebook_bug", GRAY, "E0 notebook (R-only bug)"), ("E1_baseline", BLUE, "E1 baseline"),
       ("E6_batchnorm", AQUA, "E6 +BatchNorm"), (best, ORANGE, f"{short[best]} best")]
seen = set()
for k, c, lab in sel:
    if k in seen or k not in R:
        continue
    seen.add(k)
    ep = [l["epoch"] for l in R[k]["log"]]
    a1.plot(ep, [100 * l["val_acc"] for l in R[k]["log"]], color=c, lw=1.6, label=lab)
    a1.plot(ep, [100 * l["train_acc"] for l in R[k]["log"]], color=c, lw=1.0, ls=":")
a1.set_xlabel("epoch"); a1.set_ylabel("accuracy (%)")
a1.set_title("Validation (solid) / training (dotted) accuracy", fontsize=8.5, color=INK, loc="left")
a1.legend(fontsize=7, loc="upper left")
names = list(R)
vals = [100 * R[k]["test_acc"] for k in names]
cols = [ORANGE if k == best else (GRAY if k == "E0_notebook_bug" else BLUE) for k in names]
y = np.arange(len(names))[::-1]
a2.barh(y, vals, color=cols, height=0.62)
for yi, v in zip(y, vals):
    a2.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=7, color=INK)
a2.set_yticks(y, [short[k] for k in names]); a2.set_xlim(0, max(vals) + 9)
a2.set_xlabel("test accuracy (%)"); a2.grid(axis="y", visible=False)
a2.set_title("Test accuracy (best-val checkpoint)", fontsize=8.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1.png"), dpi=220)

# Fig 2: per-class accuracy baseline vs best + confusion matrix of best
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
x = np.arange(10)
for off, k, c, lab in [(-0.2, "E1_baseline", BLUE, "E1 baseline"), (0.2, best, ORANGE, f"{short[best]} best")]:
    b1.bar(x + off, [100 * v for v in R[k]["per_class"]], width=0.38, color=c, label=lab)
b1.set_xticks(x, CLASSES, rotation=40, ha="right"); b1.set_ylabel("test accuracy (%)")
b1.grid(axis="x", visible=False); b1.legend(fontsize=7, ncol=2, loc="upper left")
b1.set_ylim(0, 100)
b1.set_title("Per-class test accuracy", fontsize=8.5, color=INK, loc="left")
cm = np.array(R[best]["cm"]); cmn = cm / cm.sum(1, keepdims=True)
b2.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(10):
    for j in range(10):
        if cmn[i, j] >= 0.08:
            b2.text(j, i, f"{100*cmn[i,j]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if cmn[i, j] > 0.5 else INK)
b2.set_xticks(x, [c[:4] for c in CLASSES], rotation=90, fontsize=6.5)
b2.set_yticks(x, [c[:4] for c in CLASSES], fontsize=6.5); b2.grid(False)
b2.set_xlabel("predicted"); b2.set_ylabel("true")
b2.set_title(f"Confusion matrix {short[best]} (row %)", fontsize=8.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2.png"), dpi=220)
print("best:", best)

# Fig 3: validation loss curves, accuracy vs params, generalisation gap
fig, (c1, c2, c3) = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1, 0.9]))
for k, c, lab in [("E1_baseline", BLUE, "E1"), ("E5_flatten", YELLOW, "E5"), ("E6_batchnorm", AQUA, "E6"), (best, ORANGE, short[best])]:
    l = R[k]["log"]
    c1.plot([x["epoch"] for x in l], [x["val_loss"] for x in l], color=c, lw=1.5, label=lab)
    i = int(np.argmin([x["val_loss"] for x in l]))
    c1.plot(l[i]["epoch"], l[i]["val_loss"], "o", ms=4, color=c, mec="white", mew=0.8)
c1.set_xlabel("epoch"); c1.set_ylabel("validation BCE loss")
c1.set_title("Validation loss (dot = minimum)", fontsize=8.5, color=INK, loc="left")
c1.set_ylim(0.125, 0.335); c1.legend(fontsize=6.5, ncol=4, loc="lower left", handlelength=1.2, columnspacing=0.8)
for k in R:
    col = ORANGE if k == best else (GRAY if k == "E0_notebook_bug" else BLUE)
    c2.plot(R[k]["params"] / 1000, 100 * R[k]["test_acc"], "o", ms=5.5, color=col, mec="white", mew=0.8)
    dx, dy = (6, -1) if k not in ("E6_batchnorm",) else (6, 1)
    c2.annotate(short[k], (R[k]["params"] / 1000, 100 * R[k]["test_acc"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.5, color=INK2, va="center")
c2.set_xscale("log"); c2.set_xlabel("parameters (thousands, log)"); c2.set_ylabel("test accuracy (%)")
c2.set_title("Accuracy vs. model size", fontsize=8.5, color=INK, loc="left")
c2.set_xlim(30, 600); c2.set_xticks([50, 100, 200, 400], ["50", "100", "200", "400"]); c2.minorticks_off()
gaps = [100 * (R[k]["log"][-1]["train_acc"] - R[k]["log"][-1]["val_acc"]) for k in names]
c3.barh(y, gaps, color=[ORANGE if k == best else (GRAY if k == "E0_notebook_bug" else BLUE) for k in names], height=0.62)
for yi, v in zip(y, gaps):
    c3.text(max(v, 0) + 0.8, yi, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
c3.set_yticks(y, [short[k] for k in names]); c3.grid(axis="y", visible=False)
c3.set_xlim(0, max(gaps) + 8); c3.set_xlabel("train - val acc. at ep. 40 (pp)")
c3.set_title("Overfitting gap", fontsize=8.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3.png"), dpi=220)

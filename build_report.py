# Builds the A4 PDF report from results/*.json and results/fig/*.png
import glob
import json
import os

import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "Anugrah Putra Al Fatih"
NIM = "26/590120/PPA/07385"
OUT = os.path.join(HERE, "Report_ConvNet_Experiments_Anugrah_Putra_Al_Fatih.pdf")

R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json")))}
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=9.8, leading=11.9,
                      alignment=TA_JUSTIFY, spaceAfter=4)
h = ParagraphStyle("h", parent=body, fontName="Times-Bold", fontSize=11.5, leading=14, spaceBefore=6, spaceAfter=3)
title = ParagraphStyle("t", parent=body, fontName="Times-Bold", fontSize=14.5, leading=18, alignment=1, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, fontSize=9.5, alignment=1, textColor=colors.HexColor("#444444"))
cap = ParagraphStyle("c", parent=body, fontSize=8.8, leading=10.8, textColor=colors.HexColor("#333333"))
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=9.4, alignment=0, spaceAfter=0)
bullet = ParagraphStyle("bl", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=2.5)


def P(t, s=body):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, bullet, bulletText="•")


def pct(x):
    return f"{100 * x:.1f}"


def macro_f1(r):
    cmx = np.array(r["cm"]); tp = cmx.diagonal()
    p = tp / np.maximum(cmx.sum(0), 1); rc = tp / cmx.sum(1)
    return float(np.mean(2 * p * rc / np.maximum(p + rc, 1e-9)))


def header_footer(c, doc):
    c.saveState()
    w, hgt = A4
    c.setFont("Times-Roman", 8.5)
    c.setFillColor(colors.HexColor("#444444"))
    c.drawString(2 * cm, hgt - 1.25 * cm, f"{NAME}  |  {NIM}")
    c.drawRightString(w - 2 * cm, hgt - 1.25 * cm, "Image Classification - ConvNet Architecture Experiments")
    c.setStrokeColor(colors.HexColor("#999999")); c.setLineWidth(0.4)
    c.line(2 * cm, hgt - 1.4 * cm, w - 2 * cm, hgt - 1.4 * cm)
    c.line(2 * cm, 1.4 * cm, w - 2 * cm, 1.4 * cm)
    c.drawString(2 * cm, 1.0 * cm, f"Name: {NAME}   Student number: {NIM}")
    c.drawRightString(w - 2 * cm, 1.0 * cm, f"Page {doc.page}")
    c.restoreState()


e = R
s = []
s.append(P("Image Classification on CIFAR-10: Effect of ConvNet Architecture", title))
s.append(P(f"{NAME} ({NIM})", sub))
s.append(P('GitHub: <link href="https://github.com/anugrahhhhh/CNN-Image-Classification-2" color="#1a5fb4">'
           'https://github.com/anugrahhhhh/CNN-Image-Classification-2</link>', sub))
s.append(Spacer(1, 4))

s.append(P("1. Objective and Setup", h))
s.append(P(
    "The provided notebook (<i>colab img_clf.ipynb</i>) was executed and its <b>ConvNet</b> "
    "(3 conv blocks + 3 fully connected blocks) was used as the starting point. Several architectural variants were "
    "then trained under <b>identical</b> conditions so that only the architecture differs between runs. Settings kept "
    "from the notebook: CIFAR-10, training set = <i>data_batch_1</i> (10,000 images), validation = <i>data_batch_2</i> "
    "(10,000), test = <i>test_batch</i> (10,000); input scaled by /255; sigmoid output with binary cross-entropy on "
    "one-hot labels; SGD (lr = 0.0025, momentum = 0.9); batch size 10. Changes for a fair comparison: "
    "(i) <b>40 epochs</b> instead of 1, because the from-scratch ConvNet barely moves away from chance in the "
    "first epochs; (ii) the checkpoint with the best validation accuracy is evaluated on the test set; (iii) fixed seed (0); "
    "(iv) the code runs locally on CPU (PyTorch CPU build, 8 threads) instead of the Colab GPU."))
s.append(P(
    "<b>Bug found in the notebook data loader.</b> In <i>myData.__getitem__</i> all three channels are sliced "
    "with <i>[0:1024]</i>, so the network receives the <u>red channel copied three times</u> instead of RGB. "
    "Run E0 reproduces this bug; all other runs use the corrected slices (R = 0:1024, G = 1024:2048, B = 2048:3072)."))

s.append(P("2. Architectures Compared", h))
s.append(P(
    "The notebook ConvNet follows the pattern <i>3 &#215; conv_block &#8594; 3 &#215; fc_block</i>, where "
    "conv_block = Conv + ReLU + MaxPool(2) and fc_block = Linear + ReLU (sigmoid at the output). Table 1 lists the "
    "layers of the baseline (E1) next to the best variant (E7). Each experiment changes <b>one</b> design factor so "
    "its effect can be isolated: <b>width</b> (E2, more filters per layer), <b>depth</b> (E3, a 4th conv block; "
    "padding = 1 so the map does not shrink to zero), <b>receptive field</b> (E4, 5&#215;5 kernels), the <b>"
    "feature aggregation head</b> (E5, flatten the spatial map instead of global average pooling) and <b>"
    "normalisation</b> (E6, BatchNorm after every conv). E7 then combines the factors that helped, plus Dropout "
    "0.3 in the FC layers to fight overfitting."))
L = [["Block", "E1 baseline layer", "Output", "Params", "E7 best layer", "Output", "Params"],
     ["Input", "-", "3&#215;32&#215;32", "-", "-", "3&#215;32&#215;32", "-"],
     ["Conv 1", "Conv3&#215;3 3&#8594;16, ReLU, MaxPool", "16&#215;15&#215;15", "448", "Conv3&#215;3 p1 3&#8594;32, BN, ReLU, Pool", "32&#215;16&#215;16", "960"],
     ["Conv 2", "Conv3&#215;3 16&#8594;32, ReLU, MaxPool", "32&#215;6&#215;6", "4,640", "Conv3&#215;3 p1 32&#8594;64, BN, ReLU, Pool", "64&#215;8&#215;8", "18,624"],
     ["Conv 3", "Conv3&#215;3 32&#8594;64, ReLU, MaxPool", "64&#215;2&#215;2", "18,496", "Conv3&#215;3 p1 64&#8594;128, BN, ReLU, Pool", "128&#215;4&#215;4", "74,112"],
     ["Head", "Global average pool", "64", "0", "Flatten", "2048", "0"],
     ["FC 1", "Linear 64&#8594;128, ReLU", "128", "8,320", "Linear 2048&#8594;128, ReLU, Dropout", "128", "262,272"],
     ["FC 2", "Linear 128&#8594;64, ReLU", "64", "8,256", "Linear 128&#8594;64, ReLU, Dropout", "64", "8,256"],
     ["FC 3", "Linear 64&#8594;10, Sigmoid", "10", "650", "Linear 64&#8594;10, Sigmoid", "10", "650"],
     ["Total", "", "", "40,810", "", "", "364,874"]]
hdr0 = ParagraphStyle("hdr0", parent=cell, fontName="Times-Bold", alignment=1)
c0 = ParagraphStyle("c0", parent=cell, alignment=1)
L = [[Paragraph(x, hdr0 if i == 0 else (cell if j in (1, 4) else c0)) for j, x in enumerate(r)] for i, r in enumerate(L)]
lt = Table(L, colWidths=[1.3 * cm, 4.6 * cm, 1.7 * cm, 1.4 * cm, 4.9 * cm, 1.7 * cm, 1.4 * cm])
lt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f3f3f1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ("LINEAFTER", (3, 0), (3, -1), 0.4, colors.HexColor("#999999")),
    ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
]))
s.append(lt)
s.append(P("<b>Table 1.</b> Layer-by-layer comparison of the baseline ConvNet (E1) and the best variant (E7). "
           "Most E7 parameters sit in FC1 because flattening keeps the 128&#215;4&#215;4 spatial map.", cap))

s.append(P("3. Results", h))
hdr = ParagraphStyle("hdr", parent=cell, fontName="Times-Bold", alignment=1)
rows = [[Paragraph(x, hdr) for x in ["ID", "Architecture change (vs. E1)", "Params", "Best epoch", "Val acc (%)",
                                     "Test acc (%)", "Macro F1", "Train time (s)", "Latency (ms/img)"]]]
for k, r in e.items():
    rows.append([k.split("_")[0], Paragraph(r["desc"], cell), f"{r['params']:,}", str(r["best_epoch"]),
                 pct(r["best_val_acc"]), pct(r["test_acc"]), f"{macro_f1(r):.3f}", f"{r['train_time_s']:.0f}", f"{r['latency_ms']:.2f}"])
best = max(e, key=lambda k: e[k]["test_acc"])
bi = list(e).index(best) + 1
t = Table(rows, colWidths=[0.8 * cm, 5.6 * cm, 1.5 * cm, 1.2 * cm, 1.5 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.9 * cm], repeatRows=1)
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Times-Roman", 8.6),
    ("FONT", (0, 0), (-1, 0), "Times-Bold", 8.4),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
    ("BACKGROUND", (0, bi), (-1, bi), colors.HexColor("#fdeee6")),
    ("FONT", (5, bi), (6, bi), "Times-Bold", 8.6),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
s.append(t)
s.append(P(
    "<b>Table 2.</b> Results of all runs (40 epochs, same optimizer, loss, data and seed). Val/test accuracy use the "
    "best-validation checkpoint. Macro F1 = unweighted mean of per-class F1 on the test set. 'Flatten' (E5) flattens the "
    "64&#215;4&#215;4 map (padding = 1). Latency = single-image forward pass on CPU (batch 1, as in the notebook test loop).", cap))

s.append(Image(os.path.join(HERE, "results", "fig", "fig1.png"), width=16 * cm, height=16 * cm * 2.5 / 7.2))
s.append(P(
    "<b>Figure 1.</b> Left: learning curves for the notebook version (E0), the corrected baseline (E1), baseline + "
    "BatchNorm (E6) and the best model (E7). Right: test accuracy of all variants.", cap))

s.append(P("4. Analysis", h))
b0, b1 = e["E0_notebook_bug"], e["E1_baseline"]
s.append(P("<b>4.1 Effect of each architectural change</b>", body))
s.append(B(
    f"<b>Correct input data matters before architecture.</b> Fixing the RGB loader alone raised test accuracy from "
    f"{pct(b0['test_acc'])}% (E0) to {pct(b1['test_acc'])}% (E1) with the same network. With the bug, colour cues "
    "(blue sky/sea for airplane/ship, green background for frog/deer) are lost."))
s.append(B(
    f"<b>The baseline underfits.</b> E1 stays at ~10% (chance) for about 12 epochs and ends at only "
    f"{pct(b1['log'][-1]['train_acc'])}% <i>training</i> accuracy, the same as its validation accuracy. So the problem is "
    "optimisation and capacity, not overfitting. A sigmoid + BCE output with a small learning rate gives weak "
    "gradients at the start (all outputs near 0.1). The tiny 2&#215;2 final map is also averaged into only 64 features."))
s.append(B(
    f"<b>Width vs. depth vs. kernel size.</b> Doubling the filters (E2, 2.9&#215; params) gave only "
    f"+{100*(e['E2_wider']['test_acc']-b1['test_acc']):.1f} pp. Adding a 4th conv block with padding (E3) gave "
    f"+{100*(e['E3_deeper']['test_acc']-b1['test_acc']):.1f} pp, and 5&#215;5 kernels (E4) "
    f"+{100*(e['E4_kernel5']['test_acc']-b1['test_acc']):.1f} pp. Larger receptive fields and keeping "
    "spatial resolution (padding) help more than simply adding filters."))
s.append(B(
    f"<b>GAP vs. Flatten.</b> Replacing global average pooling with flattening (E5) gave the largest single gain "
    f"among the 'plain' changes: {pct(e['E5_flatten']['test_acc'])}% (+{100*(e['E5_flatten']['test_acc']-b1['test_acc']):.1f} pp). "
    "GAP throws away <i>where</i> a feature appears. That is costly for a shallow 3-layer network whose features "
    "are not yet very abstract. The price is 4&#215; more parameters (mostly in FC1) and the first sign of overfitting "
    f"(train-val gap {100*(e['E5_flatten']['log'][-1]['train_acc']-e['E5_flatten']['log'][-1]['val_acc']):.0f} pp)."))
s.append(B(
    f"<b>BatchNorm is the most effective change.</b> Adding BN (E6) costs only 224 extra parameters but raised "
    f"test accuracy to {pct(e['E6_batchnorm']['test_acc'])}% (+{100*(e['E6_batchnorm']['test_acc']-b1['test_acc']):.1f} pp). "
    "It passed 25% validation accuracy at epoch 2 instead of epoch 40. BN normalises each layer's activations, "
    "which removes the poor-scaling problem of the un-centred /255 input and lets SGD make real progress. "
    f"However, E6 overfits (train {pct(e['E6_batchnorm']['log'][-1]['train_acc'])}% vs. val "
    f"{pct(e['E6_batchnorm']['log'][-1]['val_acc'])}% at epoch 40), and its validation curve is noisy."))
b7 = e["E7_bn_wide_drop"]
s.append(B(
    f"<b>Best model (E7)</b> combines the helpful changes (wider 32-64-128 + padding + BN + Flatten + Dropout 0.3). "
    f"It reaches <b>{pct(b7['test_acc'])}%</b> test accuracy, +{100*(b7['test_acc']-b1['test_acc']):.1f} pp over the "
    f"baseline and +{100*(b7['test_acc']-b0['test_acc']):.1f} pp over the original notebook. Its training accuracy reaches "
    f"{pct(b7['log'][-1]['train_acc'])}% while validation plateaus at ~65% from epoch ~20 (best epoch {b7['best_epoch']}). "
    "With only 10,000 training images and 365k parameters, dropout alone is not enough regularisation."))

s.append(Image(os.path.join(HERE, "results", "fig", "fig2.png"), width=16 * cm, height=16 * cm * 2.7 / 7.2))
s.append(P(
    "<b>Figure 2.</b> Left: per-class test accuracy of the baseline (E1) and best model (E7). Right: row-normalised "
    "confusion matrix of E7 (only cells &#8805; 8% are labelled).", cap))

pc1, pc7 = b1["per_class"], b7["per_class"]
s.append(P("<b>4.2 Per-class behaviour</b>", body))
s.append(B(
    f"<b>Vehicles are easiest</b> (automobile {pc7[1]*100:.0f}%, airplane {pc7[0]*100:.0f}%, ship {pc7[8]*100:.0f}%, "
    f"truck {pc7[9]*100:.0f}%). <b>Animals are hardest</b> (cat {pc7[3]*100:.0f}%, dog {pc7[5]*100:.0f}%, bird {pc7[2]*100:.0f}%). "
    "The main confusions are between visually similar classes: dog&#8594;cat (18% of dogs), cat&#8594;bird/deer/dog (11-12%), "
    "bird&#8594;deer (14%), ship&#8594;airplane (13%) and automobile&#8594;truck (9%). These are classes with similar shape "
    "and background at 32&#215;32 resolution."))

CL = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
pr = [[Paragraph(x, hdr0) for x in ["ID", "plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "Mean"]]]
colbest = [max(range(len(e)), key=lambda i: list(e.values())[i]["per_class"][j]) for j in range(10)]
for i, (k, r) in enumerate(e.items()):
    pr.append([k.split("_")[0]] + [f"{100*v:.0f}" for v in r["per_class"]] + [f"{100*np.mean(r['per_class']):.1f}"])
pt = Table(pr, colWidths=[1.0 * cm] + [1.42 * cm] * 10 + [1.4 * cm])
pst = [("FONT", (0, 0), (-1, -1), "Times-Roman", 8.4), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
       ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
       ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
       ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]
for j, i in enumerate(colbest):
    pst.append(("FONT", (j + 1, i + 1), (j + 1, i + 1), "Times-Bold", 8.4))
for i, r in enumerate(e.values()):
    for j, v in enumerate(r["per_class"]):
        if v < 0.05:
            pst.append(("TEXTCOLOR", (j + 1, i + 1), (j + 1, i + 1), colors.HexColor("#c0392b")))
pt.setStyle(TableStyle(pst))
s.append(pt)
s.append(P("<b>Table 3.</b> Per-class test recall (%) of every variant. Bold = best per class; red = class almost never "
           "predicted correctly (&lt; 5%).", cap))
s.append(B(
    "Table 3 shows that the improvements are <b>broad, not class-specific</b>. The baseline E1 has collapsed onto a few "
    "classes (bird, cat, deer &lt; 1% recall); from E3/E4 onward these classes start to be recognised. BatchNorm (E6) improves <i>every</i> class over E1, and E7 is the best "
    "model on 7 of 10 classes (E6 wins cat and ship). <i>Cat</i> and <i>dog</i> are the two weakest classes of both "
    "strong models (E6, E7) and are the main source of the remaining error. E0's high dog recall (54%) is an artefact: "
    "without colour the model predicts 'dog' for a large share of all images."))

s.append(Image(os.path.join(HERE, "results", "fig", "fig3.png"), width=16 * cm, height=16 * cm * 2.3 / 7.2))
s.append(P(
    "<b>Figure 3.</b> Left: validation BCE loss (dot = epoch of minimum loss). Middle: test accuracy vs. number of "
    "parameters (log scale). Right: overfitting gap = training minus validation accuracy at the last epoch.", cap))
s.append(P("<b>4.3 Training dynamics, overfitting and calibration</b>", body))
b5, b6 = e["E5_flatten"], e["E6_batchnorm"]
s.append(B(
    "The <b>validation loss</b> separates the models better than accuracy. E1 and E5 are still decreasing at "
    "epoch 40 (more epochs would help). E6 and E7 reach their minimum early (epoch "
    f"{1+int(np.argmin([x['val_loss'] for x in b6['log']]))} and "
    f"{1+int(np.argmin([x['val_loss'] for x in b7['log']]))}) and then <i>rise</i> again "
    f"(E7: {min(x['val_loss'] for x in b7['log']):.3f} &#8594; {b7['log'][-1]['val_loss']:.3f}). At the same time their "
    "validation accuracy stays flat. The networks keep becoming <b>more confident on the wrong predictions</b> "
    "(worse calibration) even though the ranking of classes barely changes. Early stopping on validation loss would "
    "pick a better-calibrated model."))
s.append(B(
    "The <b>overfitting gap</b> grows with capacity and ease of optimisation: &#8776;0 pp for E0-E4 (underfitting), "
    f"{100*(b5['log'][-1]['train_acc']-b5['log'][-1]['val_acc']):.0f} pp for E5, "
    f"{100*(b6['log'][-1]['train_acc']-b6['log'][-1]['val_acc']):.0f} pp for E6 and "
    f"{100*(b7['log'][-1]['train_acc']-b7['log'][-1]['val_acc']):.0f} pp for E7. Accuracy is <b>not</b> a simple function "
    "of parameter count (Fig. 3, middle). E6 (41k) beats E2, E3 and E5, which have 3-4&#215; more parameters. "
    "<i>How</i> the parameters are used (normalisation, spatial head) matters more than how many there are."))

s.append(P("<b>4.4 Computational cost and efficiency</b>", body))
s.append(B(
    f"Training time on CPU grew from {b1['train_time_s']:.0f} s (E1) to {b7['train_time_s']:.0f} s (E7), and single-image latency "
    f"from {b1['latency_ms']:.2f} to {b7['latency_ms']:.2f} ms (all far below 1 ms per image). E6 gives the best "
    "accuracy-per-parameter trade-off: it matches the baseline size (41k params) at more than double the accuracy. "
    "For reference, the notebook's EfficientNet-B0 transfer-learning run (Colab GPU, 1 epoch, same RGB bug) reported "
    "27.9% test accuracy with ~4M parameters."))

s.append(P("5. Limitations", h))
s.append(P(
    "Each configuration was trained once (seed 0), so differences of 1-3 pp (e.g. E2 vs. E4) are within the "
    "range that run-to-run noise could produce. "
    "Only one training batch (10k images) is used, as in the notebook, which exaggerates overfitting for the larger models. "
    "Hyper-parameters were not re-tuned per architecture, which disadvantages slow learners such as E1 "
    "(still improving at epoch 40)."))

s.append(P("6. Conclusion", h))
s.append(P(
    f"Changing the ConvNet architecture changed test accuracy from {pct(b1['test_acc'])}% to {pct(b7['test_acc'])}% "
    "under an identical training budget. Ranked by impact: (1) <b>Batch Normalisation</b> (fixes the slow, unstable "
    "optimisation); (2) <b>keeping spatial information</b> (Flatten instead of GAP, padding); (3) <b>depth / larger "
    "receptive field</b>; (4) <b>width</b> (least gain per parameter). Because the larger "
    "models overfit the 10k-image training set, next steps would be data augmentation (random crop/flip), using all "
    "5 training batches, weight decay, a learning-rate schedule, and softmax + cross-entropy instead of sigmoid + BCE."))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                        bottomMargin=1.6 * cm, title="ConvNet Architecture Experiments on CIFAR-10",
                        author=f"{NAME} ({NIM})")
doc.build(s, onFirstPage=header_footer, onLaterPages=header_footer)
print(OUT)

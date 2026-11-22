"""Evaluate the saved test split and plot run metrics."""

import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader

from data import load_test_split
from model import load_model

sns.set_context("talk")
sns.set_style("whitegrid")

def smooth(y, k=3):
    if k <= 1: return y
    return pd.Series(y).rolling(k, min_periods=1, center=True).mean().values

@torch.no_grad()
def evaluate_test(out_dir="outputs", data_root="EuroSAT", batch_size=64, workers=4):
    model, classes, device = load_model(out_dir)
    test = load_test_split(data_root, out_dir, classes)
    dl = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device == "cuda",
    )

    y_true, y_pred = [], []
    for x, y in dl:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean() * 100.0


    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar_kws={"label": "Proportion"}, ax=ax
    )
    ax.set_title(f"Confusion Matrix (Test Acc = {acc:.2f}%)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=300); plt.close(fig)
    print(f"Saved {cm_path}")

    print(f"Test accuracy: {acc:.2f}%")
    print("\nClassification report (TEST):\n")
    print(classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, digits=4, zero_division=0))

def plot_history(out_dir="outputs", smooth_k=3):
    hist_csv = os.path.join(out_dir, "history.csv")
    if not os.path.exists(hist_csv):
        print(f"[warn] {hist_csv} not found. Did you run the updated train.py?")
        return

    df = pd.read_csv(hist_csv)
    ep  = df["epoch"].values
    tl, ta = df["train_loss"].values, df["train_acc"].values
    vl, va = df["val_loss"].values, df["val_acc"].values
    pr, rc, f1 = df["val_precision_macro"].values, df["val_recall_macro"].values, df["val_f1_macro"].values
    lr  = df["lr"].values


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, tl, label="train"); axes[0].plot(ep, smooth(vl, smooth_k), "--", label="val(smooth)")
    axes[0].plot(ep, vl, alpha=.35, color="C1")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(ep, ta, label="train"); axes[1].plot(ep, smooth(va, smooth_k), "--", label="val(smooth)")
    axes[1].plot(ep, va, alpha=.35, color="C1")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    fig.tight_layout()
    p = os.path.join(out_dir, "curves_loss_acc.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"Saved {p}")


    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax = axes[0,0]; ax.plot(ep, tl, label="train"); ax.plot(ep, vl, label="val"); ax.set_title("Loss"); ax.legend(); ax.grid(True)
    ax = axes[0,1]; ax.plot(ep, ta, label="train"); ax.plot(ep, va, label="val"); ax.set_title("Accuracy"); ax.legend(); ax.grid(True)
    ax = axes[0,2]; ax.plot(ep, pr, label="val precision"); ax.set_ylim(0,1.02); ax.grid(True)
    ax.set_title("Val Precision (macro)")
    ax = axes[1,0]; ax.plot(ep, rc, label="val recall"); ax.set_ylim(0,1.02); ax.grid(True)
    ax.set_title("Val Recall (macro)")
    ax = axes[1,1]; ax.plot(ep, f1, label="val F1"); ax.set_ylim(0,1.02); ax.grid(True)
    ax.set_title("Val F1 (macro)")
    ax = axes[1,2]; ax.plot(ep, lr, label="lr"); ax.set_title("Learning Rate"); ax.grid(True)

    for ax in axes.ravel(): ax.set_xlabel("Epoch")

    fig.suptitle("Training / Validation Metrics")
    fig.tight_layout(rect=[0,0,1,0.96])
    p = os.path.join(out_dir, "metrics_grid.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"Saved {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--data_root", default="EuroSAT")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--smooth_k", type=int, default=3, help="rolling window for dashed smoothing")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()


    plot_history(out_dir=args.out_dir, smooth_k=args.smooth_k)

    evaluate_test(out_dir=args.out_dir, data_root=args.data_root, batch_size=args.batch_size, workers=args.workers)

if __name__ == "__main__":
    main()

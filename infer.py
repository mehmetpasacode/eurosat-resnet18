"""Generate predictions for the held-out split or local RGB images."""

import os, glob, csv, argparse, numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from data import load_test_split
from model import load_model



@torch.no_grad()
def predict_tensor(model, x, device):
    x = x.to(device).unsqueeze(0)
    logits = model(x)
    prob = F.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(prob).item())
    conf = float(prob[pred].item())
    return pred, conf

def load_image_64(path):
    img = Image.open(path).convert("RGB").resize((64, 64))
    tf = transforms.ToTensor()
    return tf(img)

def draw_prediction_grid(paths, y_true_idx, y_pred_idx, y_conf, classes, save_path, ncols=4):

    n = len(paths)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(3.2*ncols, 3.2*nrows))
    for i, p in enumerate(paths):
        ax = plt.subplot(nrows, ncols, i+1)
        with Image.open(p) as image:
            img = image.convert("RGB").resize((128, 128))
        ax.imshow(img)
        true_lbl = classes[y_true_idx[i]]
        pred_lbl = classes[y_pred_idx[i]]
        conf = y_conf[i]
        ok = (y_true_idx[i] == y_pred_idx[i])
        # Title color by correctness
        title = f"Pred: {pred_lbl}\nTrue: {true_lbl}\nConf: {conf:.2f}"
        ax.set_title(title, color=("green" if ok else "red"), fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def infer_test_split(data_root="EuroSAT", out_dir="outputs", grid_samples=16):
    """Run inference on the saved TEST split, save CSV and prediction grids."""
    model, classes, device = load_model(out_dir)
    test = load_test_split(data_root, out_dir, classes)

    y_true, y_pred, y_conf, paths = [], [], [], []
    rows = [("filepath", "true_class", "pred_class", "confidence")]

    for i in range(len(test)):
        x, true_idx = test[i]
        path, _ = test.dataset.samples[test.indices[i]]
        pred_idx, conf = predict_tensor(model, x, device)

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_conf.append(conf)
        paths.append(path)
        rows.append((path, classes[true_idx], classes[pred_idx], f"{conf:.4f}"))

    acc = 100.0 * np.mean(np.array(y_true) == np.array(y_pred))
    print(f"[Inference on TEST split] images={len(test)}  accuracy={acc:.2f}%")


    csv_path = os.path.join(out_dir, "preds_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"Saved test predictions -> {csv_path}")


    idx_all = np.arange(len(paths))
    if len(idx_all) > grid_samples:
        np.random.seed(42)
        idx_all = np.random.choice(idx_all, size=grid_samples, replace=False)

    grid_paths = [paths[i] for i in idx_all]
    grid_true  = [y_true[i] for i in idx_all]
    grid_pred  = [y_pred[i] for i in idx_all]
    grid_conf  = [y_conf[i] for i in idx_all]

    draw_prediction_grid(grid_paths, grid_true, grid_pred, grid_conf, classes,
                         save_path=os.path.join(out_dir, "pred_grid.png"), ncols=4)
    print("Saved prediction grid ->", os.path.join(out_dir, "pred_grid.png"))


    mis_idx = [i for i in range(len(paths)) if y_true[i] != y_pred[i]]
    if len(mis_idx) > 0:
        if len(mis_idx) > grid_samples:
            np.random.seed(42)
            mis_idx = np.random.choice(mis_idx, size=grid_samples, replace=False)
        mis_paths = [paths[i] for i in mis_idx]
        mis_true  = [y_true[i] for i in mis_idx]
        mis_pred  = [y_pred[i] for i in mis_idx]
        mis_conf  = [y_conf[i] for i in mis_idx]
        draw_prediction_grid(mis_paths, mis_true, mis_pred, mis_conf, classes,
                             save_path=os.path.join(out_dir, "pred_grid_miscls.png"), ncols=4)
        print("Saved misclassified grid ->", os.path.join(out_dir, "pred_grid_miscls.png"))
    else:
        print("No misclassifications in the sampled test set.")




def infer_custom_glob(images_glob, out_dir="outputs"):
    model, classes, device = load_model(out_dir)
    paths = sorted(glob.glob(images_glob))
    if not paths:
        print(f"No files matched: {images_glob}")
        return
    rows = [("filepath", "pred_class", "confidence")]
    for p in paths:
        x = load_image_64(p)
        pred_idx, conf = predict_tensor(model, x, device)
        rows.append((p, classes[pred_idx], f"{conf:.4f}"))
        print(f"{p} -> {classes[pred_idx]} (conf {conf:.2f})")
    csv_path = os.path.join(out_dir, "preds_custom.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"Saved custom predictions -> {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", type=str, default=None,
                    help='Optional glob for custom images, e.g. "test_images/*.jpg"')
    ap.add_argument("--data_root", type=str, default="EuroSAT")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--grid_samples", type=int, default=16, help="how many images to show in the grid")
    args = ap.parse_args()
    if args.grid_samples < 1:
        ap.error("grid_samples must be positive")

    if args.images_glob:
        infer_custom_glob(args.images_glob, out_dir=args.out_dir)
    else:
        infer_test_split(data_root=args.data_root, out_dir=args.out_dir, grid_samples=args.grid_samples)

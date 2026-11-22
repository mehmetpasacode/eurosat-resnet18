"""Train and select a checkpoint using validation accuracy."""

import argparse, os, random, csv, json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import training_splits
from model import resnet18_cifar

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_epoch(model,loader,opt,dev,crit,scaler):
    model.train(); tl,acc,cnt=0,0,0
    for x,y in tqdm(loader,leave=False,desc="train"):
        x,y=x.to(dev),y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=(dev=="cuda")):
            out=model(x); loss=crit(out,y)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        tl+=loss.item()*x.size(0); acc+=(out.argmax(1)==y).sum().item(); cnt+=x.size(0)
    return tl/cnt, acc/cnt

@torch.no_grad()
def evaluate(model,loader,dev,crit):

    model.eval(); tl,acc,cnt=0,0,0
    all_y, all_p = [], []
    for x,y in tqdm(loader,leave=False,desc="val"):
        x,y=x.to(dev),y.to(dev)
        out=model(x); l=crit(out,y)
        tl+=l.item()*x.size(0)
        p=out.argmax(1)
        acc+=(p==y).sum().item(); cnt+=x.size(0)
        all_y.extend(y.cpu().tolist()); all_p.extend(p.cpu().tolist())
    pr, rc, f1, _ = precision_recall_fscore_support(all_y, all_p, average="macro", zero_division=0)
    return tl/cnt, acc/cnt, pr, rc, f1

# ---------- main ----------
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--data_dir",required=True)
    a.add_argument("--epochs",type=int,default=80)
    a.add_argument("--batch_size",type=int,default=64)
    a.add_argument("--lr",type=float,default=1e-3)
    a.add_argument("--weight_decay",type=float,default=1e-4)
    a.add_argument("--width",type=float,default=1.0)
    a.add_argument("--dropout",type=float,default=0.2)
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--out_dir",default="outputs")
    a.add_argument("--no-augmentation", action="store_true")
    a.add_argument("--workers", type=int, default=4)
    args=a.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        a.error("epochs and batch size must be positive, and workers must be nonnegative")

    os.makedirs(args.out_dir,exist_ok=True)
    set_seed(args.seed)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print("device:",dev)

    train, val, (idx_tr, idx_va, idx_te), classes = training_splits(
        args.data_dir, args.seed, augment=not args.no_augmentation
    )
    np.savez(
        os.path.join(args.out_dir, "splits_80_10_10.npz"),
        train=idx_tr, val=idx_va, test=idx_te,
    )
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(idx_te)}")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": dev == "cuda",
    }
    dl_tr = DataLoader(train, shuffle=True, **loader_options)
    dl_va = DataLoader(val, shuffle=False, **loader_options)
    model = resnet18_cifar(len(classes), width=args.width, dropout=args.dropout).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))


    history = []
    log_csv = os.path.join(args.out_dir, "history.csv")
    log_json = os.path.join(args.out_dir, "history.json")

    best, best_ep = -1.0, -1
    for ep in range(1, args.epochs+1):
        print(f"\nEpoch {ep}/{args.epochs}")
        tr_l,tr_a = train_epoch(model, dl_tr, opt, dev, crit, scaler)
        va_l,va_a, va_p, va_r, va_f1 = evaluate(model, dl_va, dev, crit)
        sch.step()
        lr_now = opt.param_groups[0]['lr']
        print(f"train {tr_l:.4f}|{tr_a:.4f}  val {va_l:.4f}|{va_a:.4f}  "
              f"prec {va_p:.4f}  rec {va_r:.4f}  f1 {va_f1:.4f}  lr {lr_now:.6f}")

        # append and write logs every epoch
        row = {
            "epoch": ep,
            "lr": lr_now,
            "train_loss": tr_l, "train_acc": tr_a,
            "val_loss": va_l, "val_acc": va_a,
            "val_precision_macro": va_p,
            "val_recall_macro": va_r,
            "val_f1_macro": va_f1
        }
        history.append(row)

        # CSV
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)
        # JSON
        with open(log_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if va_a > best:
            best, best_ep = va_a, ep
            torch.save({"model": model.state_dict(), "classes": classes,
                        "width": args.width, "dropout": args.dropout},
                       os.path.join(args.out_dir,"best.pt"))
            print("Saved best model")

    print(f"\nBest val acc {best:.4f} (epoch {best_ep})")
    torch.save(model.state_dict(), os.path.join(args.out_dir,"last.pt"))

if __name__=="__main__":
    main()

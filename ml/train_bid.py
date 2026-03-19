#!/usr/bin/env python3
"""
Train a 2-layer MLP on pepper bid-decision counterfactual data.

Usage:
  python3 ml/train_bid.py --data bid_training.csv --out bid_model_weights.json

Streams the CSV in chunks so the full file never loads into memory.
Two passes: (1) compute target mean/std, (2) train epoch by epoch.
"""

import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Train MLP for pepper bid decisions")
    parser.add_argument("--data",    default="bid_training.csv",    help="Input CSV from cmd/collect")
    parser.add_argument("--out",     default="bid_model_weights.json", help="Output weights JSON for Go")
    parser.add_argument("--epochs",  type=int,   default=30,    help="Training epochs")
    parser.add_argument("--batch",   type=int,   default=4096,  help="Mini-batch size")
    parser.add_argument("--lr",      type=float, default=1e-3,  help="Adam learning rate")
    parser.add_argument("--val",     type=float, default=0.1,   help="Validation fraction (rows from end)")
    parser.add_argument("--h1",      type=int,   default=128,   help="Hidden layer 1 size")
    parser.add_argument("--h2",      type=int,   default=64,    help="Hidden layer 2 size")
    parser.add_argument("--wd",      type=float, default=1e-4,  help="L2 weight decay")
    parser.add_argument("--target",  default="score_delta",     help="Target column")
    parser.add_argument("--chunk",   type=int,   default=500_000, help="CSV read chunk size (rows)")
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    # --- Imports ---
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip3 install torch", file=sys.stderr)
        sys.exit(1)
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("ERROR: numpy/pandas not installed. Run: pip3 install numpy pandas", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    META = {"hand_id", "seat", "bid_level", "score_delta"}

    # --- Pass 1: count rows and compute target stats ---
    t0 = time.time()
    print(f"Pass 1: scanning {args.data} for stats ...")
    n_rows = 0
    t_sum = 0.0
    t_sum2 = 0.0
    feature_cols = None
    n_feat = 0

    for chunk in pd.read_csv(args.data, chunksize=args.chunk):
        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c not in META]
            n_feat = len(feature_cols)
        col = chunk[args.target].values.astype(np.float64)
        n_rows  += len(col)
        t_sum   += col.sum()
        t_sum2  += (col ** 2).sum()
        if n_rows % 2_000_000 < args.chunk:
            print(f"  {n_rows:>12,} rows scanned ...")

    y_mean = float(t_sum / n_rows)
    y_std  = float(np.sqrt(t_sum2 / n_rows - y_mean ** 2))
    print(f"  {n_rows:>12,} rows total  ({time.time()-t0:.1f}s)")
    print(f"  {n_feat} features  target={args.target}")
    print(f"  mean={y_mean:+.4f}  std={y_std:.4f}")

    # Split point: last val% rows go to validation.
    n_val   = int(n_rows * args.val)
    n_train = n_rows - n_val
    val_start = n_train
    print(f"  train={n_train:>10,}  val={n_val:>10,}")
    print()

    # --- Model ---
    model = nn.Sequential(
        nn.Linear(n_feat, args.h1),
        nn.ReLU(),
        nn.Linear(args.h1, args.h2),
        nn.ReLU(),
        nn.Linear(args.h2, 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:  {n_feat} → {args.h1} → {args.h2} → 1  ({n_params:,} params)")
    print(f"Config: lr={args.lr}  wd={args.wd}  batch={args.batch}  epochs={args.epochs}")
    print()
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Val RMSE':>10}  {'Time':>6}")
    print("-" * 52)
    sys.stdout.flush()

    opt      = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn  = nn.MSELoss()

    best_val   = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        train_loss = 0.0
        train_n    = 0
        val_loss   = 0.0
        val_n      = 0
        row_offset = 0

        for chunk in pd.read_csv(args.data, chunksize=args.chunk):
            chunk_len = len(chunk)
            chunk_end = row_offset + chunk_len

            X_np = chunk[feature_cols].values.astype(np.float32)
            y_np = ((chunk[args.target].values.astype(np.float32) - y_mean) / (y_std + 1e-8))

            # Rows fully in training set.
            if chunk_end <= val_start:
                # Shuffle within chunk.
                perm = np.random.permutation(chunk_len)
                X_np = X_np[perm]
                y_np = y_np[perm]
                # Mini-batch loop.
                for i in range(0, chunk_len, args.batch):
                    Xb = torch.from_numpy(X_np[i:i+args.batch])
                    yb = torch.from_numpy(y_np[i:i+args.batch])
                    opt.zero_grad()
                    pred = model(Xb).squeeze(-1)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
                    train_loss += loss.item() * len(Xb)
                    train_n    += len(Xb)

            # Rows fully in validation set.
            elif row_offset >= val_start:
                model.eval()
                with torch.no_grad():
                    for i in range(0, chunk_len, args.batch * 8):
                        Xb = torch.from_numpy(X_np[i:i+args.batch*8])
                        yb = torch.from_numpy(y_np[i:i+args.batch*8])
                        val_loss += loss_fn(model(Xb).squeeze(-1), yb).item() * len(Xb)
                        val_n    += len(Xb)
                model.train()

            # Chunk straddles the boundary.
            else:
                split = val_start - row_offset
                # Train part.
                perm = np.random.permutation(split)
                Xtr = X_np[:split][perm]
                ytr = y_np[:split][perm]
                for i in range(0, split, args.batch):
                    Xb = torch.from_numpy(Xtr[i:i+args.batch])
                    yb = torch.from_numpy(ytr[i:i+args.batch])
                    opt.zero_grad()
                    pred = model(Xb).squeeze(-1)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
                    train_loss += loss.item() * len(Xb)
                    train_n    += len(Xb)
                # Val part.
                model.eval()
                with torch.no_grad():
                    Xv = X_np[split:]
                    yv = y_np[split:]
                    for i in range(0, len(Xv), args.batch * 8):
                        Xb = torch.from_numpy(Xv[i:i+args.batch*8])
                        yb = torch.from_numpy(yv[i:i+args.batch*8])
                        val_loss += loss_fn(model(Xb).squeeze(-1), yb).item() * len(Xb)
                        val_n    += len(Xb)
                model.train()

            row_offset = chunk_end

        scheduler.step()

        if train_n > 0:
            train_loss /= train_n
        if val_n > 0:
            val_loss /= val_n
        val_rmse = (val_loss ** 0.5) * y_std

        marker = " ◀" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t_ep
        print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {val_rmse:>9.4f}  {elapsed:>5.1f}s{marker}")
        sys.stdout.flush()

    print()
    print(f"Best val MSE:  {best_val:.6f}  (RMSE ≈ {(best_val**0.5)*y_std:.4f} score points)")

    # --- Export weights ---
    model.load_state_dict(best_state)
    model.eval()

    def arr(t):
        return t.detach().numpy().tolist()

    weights = {
        "w1":         arr(model[0].weight),           # [h1][n_feat]
        "b1":         arr(model[0].bias),              # [h1]
        "w2":         arr(model[2].weight),            # [h2][h1]
        "b2":         arr(model[2].bias),              # [h2]
        "w3":         arr(model[4].weight.squeeze(0)), # [h2]
        "b3":         float(model[4].bias[0].item()),
        "y_mean":     y_mean,
        "y_std":      y_std,
        "n_features": n_feat,
        "hidden1":    args.h1,
        "hidden2":    args.h2,
        "target":     args.target,
    }

    with open(args.out, "w") as f:
        json.dump(weights, f)
    print(f"Weights saved → {args.out}")


if __name__ == "__main__":
    main()

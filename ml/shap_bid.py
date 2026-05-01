#!/usr/bin/env python3
"""
shap_bid.py — SHAP feature importance analysis for the bid model.

Reads a BidCollectRow binary file (or a sample), runs SHAP KernelExplainer
against the trained bid MLP, and prints a ranked feature importance summary.

Usage:
    python3 ml/shap_bid.py \
        --weights iters/iter_16/bid_model_weights.json \
        --data    iters/iter_16/bid_training.bin \
        --sample  2000 \
        --out     iters/iter_16/shap_summary.txt
"""

import argparse
import json
import struct
import sys
import numpy as np

# BidCollectRow binary layout (matches ml/bid_collector.go):
#   4 bytes  HandID  (uint32)
#   1 byte   Seat    (uint8)
#   1 byte   BidLevel(uint8)
#   N*4 bytes Features (float32 x BidTotalLen)
#   4 bytes  ScoreDelta (float32)
BID_TOTAL_LEN = 35  # must match BidTotalLen in bid_features.go

FEATURE_NAMES = [
    "best_trump_count", "best_has_right", "best_has_left", "best_trump_rank",
    "best_off_aces", "best_void_suits", "best_singleton_suits",
    "suit_dominance", "second_trump_count",
    "high_trump_count", "second_has_right", "second_best_rank",
    "off_kings", "sister_suit_void", "sister_suit_singleton",
    "seat_pos_norm", "is_dealer", "current_high_norm", "no_bids_yet",
    "score_us", "score_them", "score_gap", "closeout_window",
    "high_bid_is_teammate", "opponent_holding_high", "seats_left_norm",
    "passes_so_far", "partner_has_bid", "last_to_bid_flag",
    "both_bowers", "partner_bid_level", "guaranteed_tricks",
    "bid_level_norm", "bid_is_pass", "bid_is_pepper",
]

RECORD_SIZE = 6 + BID_TOTAL_LEN * 4 + 4


def load_sample(path, n_sample):
    """Read up to n_sample rows from binary BidCollectRow file."""
    import os
    file_size = os.path.getsize(path)
    total_rows = file_size // RECORD_SIZE
    print(f"  Binary file: {total_rows:,} rows  record_size={RECORD_SIZE}")

    step = max(1, total_rows // n_sample)
    rows_to_read = min(n_sample, total_rows)

    X = np.zeros((rows_to_read, BID_TOTAL_LEN), dtype=np.float32)
    y = np.zeros(rows_to_read, dtype=np.float32)

    feat_fmt = f"{BID_TOTAL_LEN}f"
    feat_size = BID_TOTAL_LEN * 4

    with open(path, "rb") as f:
        for i in range(rows_to_read):
            offset = (i * step) * RECORD_SIZE
            f.seek(offset)
            buf = f.read(RECORD_SIZE)
            if len(buf) < RECORD_SIZE:
                X = X[:i]
                y = y[:i]
                break
            feats = struct.unpack_from(feat_fmt, buf, 6)
            score = struct.unpack_from("f", buf, 6 + feat_size)[0]
            X[i] = feats
            y[i] = score

    print(f"  Loaded {len(X):,} samples (every {step}th row)")
    return X, y


def load_weights(path):
    with open(path) as f:
        d = json.load(f)
    n_feat   = int(d["n_features"])
    h1       = int(d["hidden1"])
    h2       = int(d["hidden2"])
    h3       = int(d.get("hidden3") or 0)
    y_mean   = float(d["y_mean"])
    y_std    = float(d["y_std"])

    W1 = np.array(d["w1"], dtype=np.float32).reshape(h1, n_feat)
    b1 = np.array(d["b1"], dtype=np.float32)
    W2 = np.array(d["w2"], dtype=np.float32).reshape(h2, h1)
    b2 = np.array(d["b2"], dtype=np.float32)

    if h3 > 0:
        W3 = np.array(d["w3"], dtype=np.float32).reshape(h3, h2)
        b3 = np.array(d["b3"], dtype=np.float32)
        W4 = np.array(d["w4"], dtype=np.float32).reshape(1, h3)
        b4 = float(d["b4"])
    else:
        W3 = np.array(d["w3"], dtype=np.float32).reshape(1, h2)
        b3 = np.array([float(d["b3"])], dtype=np.float32)
        W4 = None
        b4 = None

    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4,
                y_mean=y_mean, y_std=y_std, n_features=n_feat, h3=h3)


def make_predict_fn(weights):
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    W3, b3 = weights["W3"], weights["b3"]
    W4, b4 = weights["W4"], weights["b4"]
    y_mean, y_std = weights["y_mean"], weights["y_std"]
    h3 = weights["h3"]

    def predict(X):
        X = np.asarray(X, dtype=np.float32)
        h = np.maximum(0, X @ W1.T + b1)
        h = np.maximum(0, h @ W2.T + b2)
        if h3 > 0:
            h = np.maximum(0, h @ W3.T + b3)
            out = h @ W4.T + b4
        else:
            out = h @ W3.T + b3
        return (out.squeeze() * y_std + y_mean).astype(np.float64)

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data",    required=True)
    ap.add_argument("--sample",  type=int, default=2000)
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()

    print(f"Loading weights: {args.weights}")
    weights = load_weights(args.weights)
    print(f"  Model: {weights['n_features']} → {weights['W1'].shape[0]} → {weights['W2'].shape[0]}"
          + (f" → {weights['h3']}" if weights['h3'] > 0 else "") + " → 1")

    print(f"\nLoading sample from: {args.data}")
    X, y = load_sample(args.data, args.sample)

    predict = make_predict_fn(weights)

    # Sanity check
    preds = predict(X[:5])
    print(f"\nSanity check predictions: {preds}")

    print(f"\nRunning SHAP KernelExplainer on {len(X):,} samples...")
    print("  (using 200-row background — may take a minute)")

    try:
        import shap
    except ImportError:
        print("ERROR: pip install shap")
        sys.exit(1)

    # Use a small background summary to keep KernelExplainer tractable.
    bg = shap.kmeans(X, 50)
    explainer = shap.KernelExplainer(predict, bg)
    shap_values = explainer.shap_values(X, nsamples=128, silent=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_NAMES[:BID_TOTAL_LEN], mean_abs), key=lambda x: -x[1])

    lines = []
    lines.append("=" * 52)
    lines.append(" SHAP feature importance — bid model")
    lines.append(f" weights: {args.weights}")
    lines.append(f" sample:  {len(X):,} rows")
    lines.append("=" * 52)
    lines.append(f"{'rank':<5} {'feature':<28} {'mean|SHAP|':>10}")
    lines.append("-" * 52)
    for rank, (name, val) in enumerate(ranked, 1):
        lines.append(f"{rank:<5} {name:<28} {val:>10.4f}")
    lines.append("=" * 52)

    output = "\n".join(lines)
    print("\n" + output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

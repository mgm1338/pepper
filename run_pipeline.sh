#!/usr/bin/env bash
set -euo pipefail

GOGC=25 GOMEMLIMIT=3GiB ./collect-bid \
    -model model_weights.json \
    -n 100000 -rollouts 20 -workers 8 -seed 300 \
    -out iters/iter_04/bid_part1.csv \
    > iters/iter_04/collect_bid_p1.log 2>&1 && \
GOGC=25 GOMEMLIMIT=3GiB ./collect-bid \
    -model model_weights.json \
    -n 100000 -rollouts 20 -workers 8 -seed 400 \
    -out iters/iter_04/bid_part2.csv \
    > iters/iter_04/collect_bid_p2.log 2>&1 && \
GOGC=25 GOMEMLIMIT=3GiB ./collect-bid \
    -model model_weights.json \
    -n 100000 -rollouts 20 -workers 8 -seed 500 \
    -out iters/iter_04/bid_part3.csv \
    > iters/iter_04/collect_bid_p3.log 2>&1 && \
GOGC=25 GOMEMLIMIT=3GiB ./collect-bid \
    -model model_weights.json \
    -n 100000 -rollouts 20 -workers 8 -seed 600 \
    -out iters/iter_04/bid_part4.csv \
    > iters/iter_04/collect_bid_p4.log 2>&1 && \
cat iters/iter_04/bid_part1.csv > iters/iter_04/bid_training_400k.csv && \
tail -n +2 iters/iter_04/bid_part2.csv >> iters/iter_04/bid_training_400k.csv && \
tail -n +2 iters/iter_04/bid_part3.csv >> iters/iter_04/bid_training_400k.csv && \
tail -n +2 iters/iter_04/bid_part4.csv >> iters/iter_04/bid_training_400k.csv && \
python3 ml/train_bid.py \
    --data iters/iter_04/bid_training_400k.csv \
    --out iters/iter_04/bid_model_400k.json \
    --epochs 30 \
    > iters/iter_04/train_bid_400k.log 2>&1 && \
./simulate \
    -model model_weights.json \
    -bid-model iters/iter_04/bid_model_400k.json \
    -n 50000 \
    -seed 9999 \
    > iters/iter_04/eval.txt 2>&1

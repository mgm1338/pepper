#!/usr/bin/env bash
# iterate.sh — self-play training loop (card + bid co-training)
#
# Each iteration:
#   1. Collect card data using current card + bid models (MLP self-play)
#   2. Train a new card model
#   3. Collect bid data using the NEW card model + epsilon-greedy rollouts
#   4. Train a new bid model
#   5. Eval combined (new card + new bid) vs Balanced
#   6. Promote both models only if combined eval beats previous best
#
# Usage:
#   ./iterate.sh [start_iter] [max_iters]
#
# Examples:
#   ./iterate.sh          # start from iter 1, run 5 iterations
#   ./iterate.sh 4 10     # resume from iter 4, run up to iter 10

set -euo pipefail

# Keep the Go GC from letting the heap balloon into swap during long collect runs.
export GOGC=25
export GOMEMLIMIT=3GiB

ITERS_DIR="iters"
ROLLOUTS=20
WORKERS=8
EVAL_HANDS=50000
BID_EPSILON=0.9    # epsilon-greedy exploration during bid rollouts
START_ITER=${1:-1}
MAX_ITER=${2:-20}

# Scale data volume by iteration tier.
# Early iters: fast feedback. Later iters: more data to push past diminishing returns.
hands_for_iter() {
    local iter=$1
    if   [ "$iter" -le 5  ]; then echo 300000
    elif [ "$iter" -le 10 ]; then echo 500000
    else                          echo 800000
    fi
}
bid_hands_for_iter() {
    local iter=$1
    if   [ "$iter" -le 5  ]; then echo 800000
    elif [ "$iter" -le 10 ]; then echo 1200000
    else                          echo 1600000
    fi
}

BEST_SCORE_FILE="$ITERS_DIR/best_score.txt"
mkdir -p "$ITERS_DIR"

# Initialize best score from file if it exists, else 0.
if [ -f "$BEST_SCORE_FILE" ]; then
    BEST_SCORE=$(cat "$BEST_SCORE_FILE")
else
    BEST_SCORE="0.0"
fi

echo "======================================"
echo " Pepper self-play training loop"
echo " Card hands: 300k→500k→800k (by tier)"
echo " Bid hands:  800k→1.2M→1.6M (by tier)"
echo " Epsilon: $BID_EPSILON  Rollouts: $ROLLOUTS"
echo " Eval hands: $EVAL_HANDS"
echo " Scheduler: ReduceLROnPlateau (auto early-stop)"
echo " Iterations: $START_ITER → $MAX_ITER"
echo " Current best: $BEST_SCORE pts/hand"
echo "======================================"
echo ""

for iter in $(seq "$START_ITER" "$MAX_ITER"); do
    ITER_DIR="$ITERS_DIR/iter_$(printf '%02d' $iter)"
    mkdir -p "$ITER_DIR"

    HANDS=$(hands_for_iter "$iter")
    BID_HANDS=$(bid_hands_for_iter "$iter")

    echo "--------------------------------------"
    echo " Iteration $iter / $MAX_ITER  (best so far: $BEST_SCORE)"
    echo " Card hands: $HANDS  Bid hands: $BID_HANDS"
    echo "--------------------------------------"

    # --- 1. Collect card data ---
    echo "[1/5] Collecting $HANDS card-play hands (seed=$iter)..."
    DATA="$ITER_DIR/training.csv"

    BID_FLAG=""
    if [ -f bid_model_weights.json ]; then
        BID_FLAG="-bid-model bid_model_weights.json"
    fi
    bin/collect \
        -model model_weights.json \
        $BID_FLAG \
        -n "$HANDS" \
        -rollouts "$ROLLOUTS" \
        -workers "$WORKERS" \
        -seed "$iter" \
        -out "$DATA" \
        > "$ITER_DIR/collect.log" 2>&1
    ROWS=$(wc -l < "$DATA")
    echo "    Done. Rows: $ROWS"

    # --- 2. Train card model ---
    echo "[2/5] Training card model (plateau scheduler, max 100 epochs)..."
    NEW_WEIGHTS="$ITER_DIR/model_weights.json"
    python3 ml/train.py \
        --data "$DATA" \
        --out "$NEW_WEIGHTS" \
        > "$ITER_DIR/train.log" 2>&1
    rm -f "$DATA"

    VAL_RMSE=$(grep "Best val MSE" "$ITER_DIR/train.log" | grep -oE 'RMSE ≈ [0-9.]+' | grep -oE '[0-9.]+')
    CARD_EPOCHS_RAN=$(grep -c "◀\|[0-9]  0\." "$ITER_DIR/train.log" || true)
    echo "    Done. Best val RMSE: $VAL_RMSE"

    # --- 3. Collect bid data using new card model + epsilon-greedy ---
    echo "[3/5] Collecting $BID_HANDS bid hands (epsilon=$BID_EPSILON, seed=$((iter + 100)))..."
    BID_DATA="$ITER_DIR/bid_training.csv"
    bin/collect-bid \
        -model "$NEW_WEIGHTS" \
        -n "$BID_HANDS" \
        -rollouts "$ROLLOUTS" \
        -workers "$WORKERS" \
        -seed "$((iter + 100))" \
        -epsilon "$BID_EPSILON" \
        -out "$BID_DATA" \
        > "$ITER_DIR/collect_bid.log" 2>&1
    BID_ROWS=$(wc -l < "$BID_DATA")
    echo "    Done. Rows: $BID_ROWS"

    # --- 4. Train bid model ---
    echo "[4/5] Training bid model (plateau scheduler, max 100 epochs)..."
    NEW_BID_WEIGHTS="$ITER_DIR/bid_model_weights.json"
    python3 ml/train_bid.py \
        --data "$BID_DATA" \
        --out "$NEW_BID_WEIGHTS" \
        > "$ITER_DIR/train_bid.log" 2>&1
    rm -f "$BID_DATA"

    BID_RMSE=$(grep "Best val MSE" "$ITER_DIR/train_bid.log" | grep -oE 'RMSE ≈ [0-9.]+' | grep -oE '[0-9.]+')
    echo "    Done. Best val RMSE: $BID_RMSE"

    # --- 5. Eval combined vs Balanced + vs previous promoted ---
    echo "[5/5] Eval: new models vs Balanced ($EVAL_HANDS hands)..."
    EVAL_OUT="$ITER_DIR/eval.txt"
    ./simulate \
        -model "$NEW_WEIGHTS" \
        -bid-model "$NEW_BID_WEIGHTS" \
        -n "$EVAL_HANDS" \
        -seed 9999 \
        > "$EVAL_OUT" 2>&1

    ADV=$(grep "MLP avg advantage" "$EVAL_OUT" | grep -oE '[+-][0-9.]+')
    VERDICT=$(grep "^Result:" "$EVAL_OUT" | sed 's/Result: //')
    echo "    vs Balanced: $ADV pts/hand  ($VERDICT)"

    # Also eval new vs current promoted (self-improvement check).
    EVAL_VS_PREV="$ITER_DIR/eval_vs_prev.txt"
    ./simulate \
        -model "$NEW_WEIGHTS" \
        -bid-model "$NEW_BID_WEIGHTS" \
        -n "$EVAL_HANDS" \
        -seed 1234 \
        > "$EVAL_VS_PREV" 2>&1
    ADV_VS_PREV=$(grep "MLP avg advantage" "$EVAL_VS_PREV" | grep -oE '[+-][0-9.]+')
    echo "    vs prev promoted: $ADV_VS_PREV pts/hand"

    # --- 6. Promote only if beats Balanced baseline ---
    PROMOTED="no"
    if awk "BEGIN { exit !($ADV > $BEST_SCORE) }"; then
        echo "    NEW BEST ($ADV > $BEST_SCORE) — promoting both models."
        cp model_weights.json     "$ITER_DIR/model_weights_prev.json"
        cp bid_model_weights.json "$ITER_DIR/bid_model_weights_prev.json"
        cp "$NEW_WEIGHTS"         model_weights.json
        cp "$NEW_BID_WEIGHTS"     bid_model_weights.json
        BEST_SCORE="$ADV"
        echo "$BEST_SCORE" > "$BEST_SCORE_FILE"
        PROMOTED="yes"
    else
        echo "    No improvement ($ADV <= $BEST_SCORE) — keeping current models."
    fi

    # Log summary.
    echo "iter=$iter  hands=$HANDS  bid_hands=$BID_HANDS  card_rmse=$VAL_RMSE  bid_rmse=$BID_RMSE  adv_vs_balanced=$ADV  best=$BEST_SCORE  promoted=$PROMOTED" \
        >> "$ITERS_DIR/progress.log"

    echo ""
done

echo "======================================"
echo " Done. Results in $ITERS_DIR/progress.log"
echo "======================================"
cat "$ITERS_DIR/progress.log"

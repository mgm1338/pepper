#!/usr/bin/env bash
# iterate.sh — self-play training loop
#
# Each iteration:
#   1. Collect data using the current model (MLP self-play)
#   2. Train a new model on that data
#   3. Run paired eval vs Balanced to measure improvement
#   4. Save iteration artifacts and update model_weights.json if improved
#
# Usage:
#   ./iterate.sh [start_iter] [max_iters]
#
# Examples:
#   ./iterate.sh          # start from iter 1, run 5 iterations
#   ./iterate.sh 2 10     # resume from iter 2, run up to iter 10

set -euo pipefail

ITERS_DIR="iters"
HANDS=300000
ROLLOUTS=20
WORKERS=8
EVAL_HANDS=50000
EPOCHS=30
START_ITER=${1:-1}
MAX_ITER=${2:-5}

mkdir -p "$ITERS_DIR"

echo "======================================"
echo " Pepper self-play training loop"
echo " Hands/iter: $HANDS  Rollouts: $ROLLOUTS"
echo " Eval hands: $EVAL_HANDS  Epochs: $EPOCHS"
echo " Iterations: $START_ITER → $MAX_ITER"
echo "======================================"
echo ""

for iter in $(seq "$START_ITER" "$MAX_ITER"); do
    ITER_DIR="$ITERS_DIR/iter_$(printf '%02d' $iter)"
    mkdir -p "$ITER_DIR"

    echo "--------------------------------------"
    echo " Iteration $iter / $MAX_ITER"
    echo "--------------------------------------"

    # --- 1. Collect ---
    echo "[1/3] Collecting $HANDS hands (seed=$iter)..."
    DATA="$ITER_DIR/training.csv"

    if [ -f model_weights.json ]; then
        ./collect \
            -model model_weights.json \
            -n "$HANDS" \
            -rollouts "$ROLLOUTS" \
            -workers "$WORKERS" \
            -seed "$iter" \
            -out "$DATA" \
            > "$ITER_DIR/collect.log" 2>&1
        echo "    MLP self-play collection done."
    else
        ./collect \
            -n "$HANDS" \
            -rollouts "$ROLLOUTS" \
            -workers "$WORKERS" \
            -seed "$iter" \
            -out "$DATA" \
            > "$ITER_DIR/collect.log" 2>&1
        echo "    Balanced collection done (no model yet)."
    fi

    ROWS=$(wc -l < "$DATA")
    echo "    Rows: $ROWS"

    # --- 2. Train ---
    echo "[2/3] Training ($EPOCHS epochs)..."
    NEW_WEIGHTS="$ITER_DIR/model_weights.json"
    python3 ml/train.py \
        --data "$DATA" \
        --out "$NEW_WEIGHTS" \
        --epochs "$EPOCHS" \
        > "$ITER_DIR/train.log" 2>&1

    VAL_RMSE=$(grep "Best val MSE" "$ITER_DIR/train.log" | grep -oE 'RMSE ≈ [0-9.]+' | grep -oE '[0-9.]+')
    echo "    Best val RMSE: $VAL_RMSE"

    # Back up previous model before replacing.
    if [ -f model_weights.json ]; then
        cp model_weights.json "$ITER_DIR/model_weights_prev.json"
    fi
    cp "$NEW_WEIGHTS" model_weights.json

    # --- 3. Eval vs Balanced ---
    echo "[3/3] Paired eval vs Balanced ($EVAL_HANDS hands)..."
    EVAL_OUT="$ITER_DIR/eval.txt"
    ./simulate \
        -model model_weights.json \
        -n "$EVAL_HANDS" \
        -seed 9999 \
        > "$EVAL_OUT" 2>&1

    ADV=$(grep "MLP avg advantage" "$EVAL_OUT" | grep -oE '[+-][0-9.]+')
    VERDICT=$(grep "^Result:" "$EVAL_OUT" | sed 's/Result: //')
    echo "    Advantage vs Balanced: $ADV pts/hand"
    echo "    $VERDICT"

    # Summary line appended to top-level log.
    echo "iter=$iter  rmse=$VAL_RMSE  adv_vs_balanced=$ADV  rows=$ROWS" \
        >> "$ITERS_DIR/progress.log"

    echo ""
done

echo "======================================"
echo " Done. Results in $ITERS_DIR/progress.log"
echo "======================================"
cat "$ITERS_DIR/progress.log"

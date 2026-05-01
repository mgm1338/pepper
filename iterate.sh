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

# GOMEMLIMIT is the hard cap — GOGC=25 was redundant and caused excessive GC overhead (~21% CPU).
export GOGC=100
export GOMEMLIMIT=3GiB

ITERS_DIR="iters"
ROLLOUTS=5
ROLLOUT_EPSILON=0.3  # stochastic rollouts for diverse value estimates
BID_ROLLOUTS=10
WORKERS=8
EVAL_HANDS=500000
BID_EPSILON=0.9    # epsilon-greedy exploration during bid rollouts
CARD_EPSILON=0.2   # epsilon-greedy exploration during card collection
BID_ROUNDS=3       # bid collect+train rounds per card iteration
USE_GO=${USE_GO:-true} # Set to true for blazing binary format + Go trainer
START_ITER=${1:-1}
MAX_ITER=${2:-9999}

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
    EXT="csv"
    FORMAT="csv"
    if [ "$USE_GO" = "true" ]; then EXT="bin"; FORMAT="bin"; fi
    DATA="$ITER_DIR/training.$EXT"

    BID_FLAG=""
    if [ -f bid_model_weights.json ]; then
        BID_FLAG="-bid-model bid_model_weights.json"
    fi
    bin/collect \
        -model model_weights.json \
        $BID_FLAG \
        -n "$HANDS" \
        -rollouts "$ROLLOUTS" \
        -rollout-epsilon "$ROLLOUT_EPSILON" \
        -workers "$WORKERS" \
        -seed "$iter" \
        -epsilon "$CARD_EPSILON" \
        -out "$DATA" \
        -format "$FORMAT" \
        > "$ITER_DIR/collect.log" 2>&1
    
    if [ "$FORMAT" = "csv" ]; then
        ROWS=$(wc -l < "$DATA")
        echo "    Done. Rows: $ROWS"
    else
        echo "    Done. (Binary format)"
    fi

    # --- 2. Train card model ---
    echo "[2/5] Training card model (stops when LR < 1e-5)..."
    NEW_WEIGHTS="$ITER_DIR/model_weights.json"
    CARD_WARM=""
    CARD_LR="3e-3"
    if [ -f candidate_model_weights.json ]; then CARD_WARM="-warm-start candidate_model_weights.json"; CARD_LR="1e-3"
    elif [ -f model_weights.json ]; then CARD_WARM="-warm-start model_weights.json"; CARD_LR="1e-3"; fi
    if [ "$USE_GO" = "true" ]; then
        bin/train \
            -data "$DATA" \
            -format "$FORMAT" \
            -h1 256 -h2 128 \
            -epochs 1000 \
            -patience 5 \
            -lr "$CARD_LR" \
            -min-lr 1e-5 \
            -min-delta 0.001 \
            $CARD_WARM \
            -out "$NEW_WEIGHTS" \
            > "$ITER_DIR/train.log" 2>&1
    else
        python3 ml/train.py \
            --data "$DATA" \
            --out "$NEW_WEIGHTS" \
            > "$ITER_DIR/train.log" 2>&1
    fi
    rm -f "$DATA"
    cp "$NEW_WEIGHTS" candidate_model_weights.json

    VAL_RMSE=$(grep "Best val MSE" "$ITER_DIR/train.log" | grep -oE 'RMSE ≈ [0-9.]+' | grep -oE '[0-9.]+')
    CARD_EPOCHS_RAN=$(grep -c "◀\|[0-9]  0\." "$ITER_DIR/train.log" || true)
    echo "    Done. Best val RMSE: $VAL_RMSE"

    # --- 3+4. Collect + train bid model (BID_ROUNDS rounds per card iter) ---
    BID_EXT="csv"
    BID_FORMAT="csv"
    if [ "$USE_GO" = "true" ]; then BID_EXT="bin"; BID_FORMAT="bin"; fi
    NEW_BID_WEIGHTS="$ITER_DIR/bid_model_weights.json"
    BID_RMSE=""

    for bid_round in $(seq 1 "$BID_ROUNDS"); do
        echo "[3/5] Collecting $BID_HANDS bid hands (round $bid_round/$BID_ROUNDS, epsilon=$BID_EPSILON, seed=$((iter + 100 + bid_round)))..."
        BID_DATA="$ITER_DIR/bid_training_r${bid_round}.$BID_EXT"
        BID_MODEL_FLAG=""
        if [ -f candidate_bid_model_weights.json ]; then BID_MODEL_FLAG="-bid-model candidate_bid_model_weights.json"
        elif [ -f bid_model_weights.json ]; then BID_MODEL_FLAG="-bid-model bid_model_weights.json"; fi
        bin/collect-bid \
            -model "$NEW_WEIGHTS" \
            $BID_MODEL_FLAG \
            -n "$BID_HANDS" \
            -rollouts "$BID_ROLLOUTS" \
            -workers "$WORKERS" \
            -seed "$((iter + 100 + bid_round))" \
            -epsilon "$BID_EPSILON" \
            -out "$BID_DATA" \
            -format "$BID_FORMAT" \
            > "$ITER_DIR/collect_bid_r${bid_round}.log" 2>&1

        if [ "$BID_FORMAT" = "csv" ]; then
            BID_ROWS=$(wc -l < "$BID_DATA")
            echo "    Done. Rows: $BID_ROWS"
        else
            echo "    Done. (Binary format)"
        fi

        echo "[4/5] Training bid model (round $bid_round/$BID_ROUNDS, stops when LR < 1e-5)..."
        BID_WARM=""
        if [ -f candidate_bid_model_weights.json ]; then BID_WARM="-warm-start candidate_bid_model_weights.json"
        elif [ -f bid_model_weights.json ]; then BID_WARM="-warm-start bid_model_weights.json"; fi
        if [ "$USE_GO" = "true" ]; then
            bin/train \
                -data "$BID_DATA" \
                -format "$BID_FORMAT" \
                -mode bid \
                -h1 256 -h2 128 \
                -epochs 1000 \
                -patience 5 \
                -min-lr 1e-5 \
                -min-delta 0.001 \
                $BID_WARM \
                -out "$NEW_BID_WEIGHTS" \
                > "$ITER_DIR/train_bid_r${bid_round}.log" 2>&1
        else
            python3 ml/train_bid.py \
                --data "$BID_DATA" \
                --out "$NEW_BID_WEIGHTS" \
                > "$ITER_DIR/train_bid_r${bid_round}.log" 2>&1
        fi
        rm -f "$BID_DATA"
        cp "$NEW_BID_WEIGHTS" candidate_bid_model_weights.json

        BID_RMSE=$(grep "Best val MSE" "$ITER_DIR/train_bid_r${bid_round}.log" | grep -oE 'RMSE ≈ [0-9.]+' | grep -oE '[0-9.]+')
        echo "    Done. Best val RMSE: $BID_RMSE"
    done

    # --- 5. Eval combined vs Balanced + head-to-head vs current promoted ---
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

    # Head-to-head: new vs current promoted models.
    EVAL_VS_PREV="$ITER_DIR/eval_vs_prev.txt"
    ./simulate \
        -model "$NEW_WEIGHTS" \
        -bid-model "$NEW_BID_WEIGHTS" \
        -opponent-model model_weights.json \
        -opponent-bid-model bid_model_weights.json \
        -n "$EVAL_HANDS" \
        -seed 1234 \
        > "$EVAL_VS_PREV" 2>&1
    ADV_VS_PREV=$(grep "MLP avg advantage" "$EVAL_VS_PREV" | grep -oE '[+-][0-9.]+')
    echo "    vs prev promoted (head-to-head): $ADV_VS_PREV pts/hand"

    # --- 6. Promote only if beats current promoted head-to-head ---
    PROMOTED="no"
    if awk "BEGIN { exit !($ADV_VS_PREV > 0) }"; then
        echo "    NEW BEST (head-to-head: $ADV_VS_PREV > 0) — promoting both models."
        cp model_weights.json     "$ITER_DIR/model_weights_prev.json"
        cp bid_model_weights.json "$ITER_DIR/bid_model_weights_prev.json"
        cp "$NEW_WEIGHTS"         model_weights.json
        cp "$NEW_BID_WEIGHTS"     bid_model_weights.json
        BEST_SCORE="$ADV"
        echo "$BEST_SCORE" > "$BEST_SCORE_FILE"
        PROMOTED="yes"
    else
        echo "    No improvement (head-to-head: $ADV_VS_PREV <= 0) — keeping current models."
    fi

    # Log summary.
    echo "iter=$iter  hands=$HANDS  bid_hands=$BID_HANDS  card_rmse=$VAL_RMSE  bid_rmse=$BID_RMSE  adv_vs_balanced=$ADV  adv_vs_prev=$ADV_VS_PREV  best=$BEST_SCORE  promoted=$PROMOTED" \
        >> "$ITERS_DIR/progress.log"

    # Sentinel: touch iters/PAUSE to stop cleanly after this iteration.
    if [ -f "$ITERS_DIR/PAUSE" ]; then
        echo "======================================"
        echo " PAUSE sentinel detected — stopping after iter $iter."
        echo " Resume with: ./iterate.sh $((iter+1)) $MAX_ITER"
        echo "======================================"
        break
    fi

    echo ""
done

echo "======================================"
echo " Done. Results in $ITERS_DIR/progress.log"
echo "======================================"
cat "$ITERS_DIR/progress.log"

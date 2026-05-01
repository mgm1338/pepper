#!/usr/bin/env bash
set -e

HANDS=5000
EPOCHS=10
SEED=42

echo "=== 0. BUILDING BINARIES ==="
go build -o bin/collect cmd/collect/main.go
go build -o bin/train cmd/train/main.go

echo "=== 1. COLLECTION SPEED TEST ==="
echo "Collecting $HANDS hands..."

echo -n "Python Path (CSV): "
time ./bin/collect -n $HANDS -out test_collect.csv -format csv -seed $SEED > /dev/null

echo -n "Go Path (Binary): "
time ./bin/collect -n $HANDS -out test_collect.bin -format bin -seed $SEED > /dev/null

echo ""
echo "=== 2. TRAINING SPEED & CORRECTNESS TEST ==="
echo "Training for $EPOCHS epochs..."

echo "--- Python (PyTorch) ---"
time python3 ml/train.py --data test_collect.csv --out weights_py.json --epochs $EPOCHS --seed $SEED > train_py.log 2>&1
grep "Best val MSE" train_py.log

echo "--- Go (Accelerate/Adam) ---"
# Use binary for Go to see real performance
time ./bin/train -data test_collect.bin -format bin -out weights_go.json -epochs $EPOCHS -seed $SEED > train_go.log 2>&1
grep "Best val MSE" train_go.log

echo ""
echo "=== 3. WEIGHT CONSISTENCY CHECK ==="
PY_STATS=$(grep "mean=" train_py.log)
GO_STATS=$(grep "mean=" train_go.log)
echo "Python: $PY_STATS"
echo "Go:     $GO_STATS"

# Clean up
rm -f test_collect.csv test_collect.bin train_py.log train_go.log weights_py.json weights_go.json

#!/usr/bin/env bash
# Wrapper for ./collect with GC tuning to prevent heap trashing on long runs.
# Usage: ./run-collect.sh [all ./collect flags]
#   e.g. ./run-collect.sh -model model_weights.json -bid-model bid_model_weights.json -n 140000 -rollouts 10 -workers 8 -out iters/iter_02/part_remaining.csv

export GOGC=25
export GOMEMLIMIT=3GiB

exec ./collect "$@"

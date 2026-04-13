#!/usr/bin/env bash
while true; do
    echo "=== $(date) ==="
    tail -2 iters/iter_04/collect_bid_p*.log 2>/dev/null
    echo ""
    sleep 1800
done

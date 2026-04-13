# Pepper

A 6-handed bid euchre variant played with a double euchre (pinochle) deck, featuring self-play trained ML bots, a difficulty system, and an HTTP microservice for live play.

## What is Pepper?

Pepper is a 6-handed bid euchre variant played with a double euchre deck, also known as a pinochle deck (48 cards: 9 through Ace in 4 suits, 2 copies of each). Six players sit in two teams of three. Each hand has a bidding round, a trump selection, and 8 tricks. The special call "pepper" lets the winning bidder take on all four partners' best trump cards and attempt to sweep all 8 tricks for a large point swing.

Win condition: first team to reach 64 points.

## Current results

After 8 iterations of self-play training, the best model scores **+1.73 pts/hand** vs the rule-based Balanced strategy over 50K evaluation hands. That translates to roughly a 25-point advantage per game, nearly 40% of the 64-point win threshold.

## Project structure

```
internal/card/        Card types, suits, ranks, trump ranking logic
internal/game/        Game engine: bidding, tricks, scoring, history
internal/strategy/    Rule-based strategy + difficulty system (levels 1-5)
internal/mlstrategy/  MLP wrapper strategy (card play + optional bid model)
ml/                   Feature extraction, MLP inference, training data collection
  train.py            PyTorch MLP trainer (card play)
  train_bid.py        PyTorch MLP trainer (bidding)
sim/                  Batch simulation, statistics, hand profile tables
cmd/
  botserver/          HTTP microservice with difficulty levels + personalities
  collect/            Parallel counterfactual data collection (card play)
  collect-bid/        Parallel counterfactual data collection (bidding)
  simulate/           Strategy comparison runner
  evolve/             Evolutionary parameter search for the rule-based strategy
  trace/              Human-readable game trace printer
  pepper-analysis/    Pepper call profitability analyzer
iterate.sh            Automated self-play training loop
```

## Quick start

```bash
go build ./...
go test ./...
```

Run the bot server (rule-based, no model required):
```bash
go run ./cmd/botserver
```

With trained models and difficulty support:
```bash
go run ./cmd/botserver -model model_weights.json -bid-model bid_model_weights.json -level 5
```

## How training works

The bot learns card play and bidding through **counterfactual rollout training**: a form of self-play where every legal option at every decision point is evaluated, not just the option that was actually taken.

### The core idea

When a player must choose a card, there are typically 1 to 8 legal options. Instead of just playing one card and learning from the outcome, we:

1. Snapshot the current game state
2. For each legal card, simulate the rest of the hand `N` times with that card forced
3. Record the average score outcome for each card as a training target
4. Train a neural network to predict those outcomes from features of the (state, card) pair

At inference time, the network scores every legal card and the highest scorer wins. For defenders, the score is from the bidding team's perspective, so the sign is flipped.

Bidding works the same way: for each valid bid level (pass, 4-7, pepper), run counterfactual rollouts to estimate the expected score delta, then train the bid MLP to predict those values.

### Automated training loop

Training is fully automated via `iterate.sh`, which runs a self-play loop that co-trains card play and bidding models:

```bash
./iterate.sh [start_iter] [max_iters]
```

Each iteration runs 5 steps:

1. **Collect card data** using current card + bid models (MLP self-play, 8 parallel workers, 20 rollouts per decision)
2. **Train a new card model** (PyTorch, ReduceLROnPlateau scheduler, max 100 epochs)
3. **Collect bid data** using the new card model + epsilon-greedy exploration (epsilon=0.9)
4. **Train a new bid model** (same scheduler)
5. **Eval** combined (new card + new bid) vs Balanced over 50K hands

Both models are promoted only if the combined eval beats the previous best. Data volume scales by tier:

| Tier | Iterations | Card hands | Bid hands |
|------|-----------|------------|-----------|
| 1    | 1-5       | 300K       | 800K      |
| 2    | 6-10      | 500K       | 1.2M      |
| 3    | 11+       | 800K       | 1.6M      |

### Performance optimizations

Data collection is the bottleneck (billions of game state evaluations per iteration), so the simulation is heavily optimized in Go:

- **8 parallel goroutine workers** with independent RNGs and minimal coordination
- **sync.Pool** for hand buffers, trick objects, and history to eliminate per-rollout allocations
- **In-place card removal** instead of allocating new slices
- **Flattened weight matrices** (row-major `[]float32`) for cache-friendly MLP inference
- **cgo + Apple Accelerate** (CBLAS) for GEMV operations in the MLP forward pass, with a pure-Go fallback for Linux/Docker via build tags
- **Dedicated CSV writer goroutine** to avoid blocking collection workers

These optimizations moved collection times from days (pure Python) to hours (Go), enabling multiple training iterations per day.

### Feature extraction

**Card play features (43 total):**

Context features (37) shared across all legal cards at a decision point:
- Position and role: seat relative to bidder, trick order position
- Trick progress: trick number, tricks taken, tricks needed to make/set bid
- Current trick state: cards played, who is winning, winning trump rank
- Hand composition: trump count, bower holdings, off-suit aces, voids, singletons
- Trump field knowledge: trump remaining/played, bowers played
- Score context: team scores, closeout range
- Bid context: normalized bid amount

Per-card features (6) appended per candidate:
- Trump status, normalized rank, top trump status, beats winner, overtrump, suit count remaining

**Bid features (22 total):**

Context (19): best trump suit statistics, bidding position, current high bid, score context, opportunity cost (teammate holds high bid, seats remaining).

Per-action (3): normalized bid level, is-pass flag, is-pepper flag.

### Model architecture

Both models are 2-hidden-layer MLPs with ReLU activations:

```
Card play:  43 features -> 128 -> 64 -> 1  (predicts score_delta for bidding team)
Bidding:    22 features -> 128 -> 64 -> 1  (predicts score_delta for this seat's team)
```

Training uses Adam + ReduceLROnPlateau with patience=5, halving the LR on each plateau until it drops below 1e-6 (early stop). CSVs are streamed in chunks so large datasets never fully load into memory.

## Difficulty system

The bot server supports 5 difficulty levels that blend the MLP with the rule-based Balanced strategy:

| Level | Card play | Bidding | Feel |
|-------|-----------|---------|------|
| 1 | Balanced + 12% random | Balanced + overbid noise | Knows the rules, sloppy |
| 2 | Balanced + 5% random | Balanced + slight noise | Less sloppy |
| 3 | Clean Balanced | Clean Balanced | Solid, no mistakes, no edge |
| 4 | 70% MLP, 30% Balanced | Balanced | Counts cards, plays smart |
| 5 | 100% MLP | 100% MLP | Optimal AI |

Named bot personalities add character: Peter overbids, Redneck plays aggressively, Gary is cautious, Nick Palmer takes risks, and level-5 bots are named after mathematicians (Tesla, Gauss, Euler, etc.).

The `/advice` endpoint provides "Ask Grandpa" functionality: a level-5 AI suggestion for the human player, limited to 2 uses per game.

## HTTP API

The bot server exposes endpoints for bid, trump, play, and advice decisions. Each request accepts optional `level` and `personality` fields to control difficulty.

### POST /bid
```json
{ "seat": 2, "level": 4,
  "hand": ["JH","JD","AH","KH","QH","AD","KD","QD"],
  "bid_state": { "current_high": 5, "dealer_seat": 0, "scores": [10, 8] } }
-> { "bid": 6 }
```

### POST /play
```json
{ "seat": 2, "level": 5,
  "valid_plays": ["AH","KH"],
  "state": { "trump": "H", "bidder_seat": 2, "bid_amount": 5,
             "trick_number": 3, "tricks_taken": [1,1,1,0,0,0],
             "scores": [10,8], "hand": ["AH","KH","QH"],
             "trick": [{"card":"9H","seat":1}], "leader": 1,
             "history": [["JH","9D","TC","KS","AS","QH"]] } }
-> { "card": "AH" }
```

### POST /advice
```json
{ "type": "play", "seat": 2,
  "hand": ["AH","KH","QH"],
  "valid_plays": ["AH","KH"],
  "state": { ... } }
-> { "suggestion": "AH", "explanation": "Grandpa says play the AH" }
```

Card codes: rank prefix (`9`, `T`, `J`, `Q`, `K`, `A`) + suit suffix (`H`, `D`, `C`, `S`).

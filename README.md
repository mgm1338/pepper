# Pepper

A 6-player pinochle-family card game engine with a rule-based strategy, ML-trained bots, and an HTTP microservice for live play.

## What is Pepper?

Pepper is a trick-taking card game played with a double pinochle deck (48 cards: 9 through Ace in 4 suits, 2 copies of each). Six players sit in two teams of three. Each hand has a bidding round, a trump selection, and 8 tricks. The special call "pepper" lets the winning bidder take on all four partners' best trump cards and attempt to sweep all 8 tricks for a large point swing.

Win condition: first team to reach 64 points.

## Project structure

```
internal/card/        Card types, suits, ranks, trump ranking logic
internal/game/        Game engine: bidding, tricks, scoring, history
internal/strategy/    Rule-based strategy with 50+ tunable parameters
internal/mlstrategy/  MLP wrapper strategy (card play + optional bid model)
ml/                   Feature extraction, MLP inference, training data collection
  train.py            PyTorch MLP trainer (card play)
  train_bid.py        PyTorch MLP trainer (bidding)
sim/                  Batch simulation, statistics, hand profile tables
cmd/
  botserver/          HTTP microservice for live game integration
  collect/            Parallel counterfactual data collection (card play)
  collect-bid/        Parallel counterfactual data collection (bidding)
  simulate/           Strategy comparison runner
  evolve/             Evolutionary parameter search for the rule-based strategy
  trace/              Human-readable game trace printer
  pepper-analysis/    Pepper call profitability analyzer
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

With a trained model:
```bash
go run ./cmd/botserver --model model_weights.json --bid-model bid_model_weights.json
```

---

## How training works

The bot learns card play and bidding through **counterfactual rollout training**: a form of self-play where every legal option at every decision point is evaluated, not just the option that was actually taken.

### The core idea

When a player must choose a card, there are typically 1 to 8 legal options. Instead of just playing one card and learning from the outcome, we:

1. Snapshot the current game state
2. For each legal card, simulate the rest of the hand `N` times with that card forced
3. Record the average score outcome for each card as a training target
4. Train a neural network to predict those outcomes from features of the (state, card) pair

At inference time, the network scores every legal card and the highest scorer wins. For defenders, the score is from the bidding team's perspective, so the sign is flipped.

### Counterfactual rollouts in detail

The key function is `CollectHand` in `ml/collector.go`. It runs a complete hand, intercepting every play decision.

At each decision point, a `decisionPoint` snapshot is taken containing:
- All 6 players' current hands (deep copy)
- Cards already played in the current partial trick
- Full trick history
- Game score, trump suit, bid info

For each legal card `c` at this decision point, `rollout(c, ...)` is called `N` times. Each rollout:
1. Plays `c` as the forced card for this seat
2. Completes the current trick using the rollout strategy for remaining seats
3. Plays all remaining tricks using the rollout strategy for all seats
4. Scores the hand and returns `(score_delta, made_bid)`

The mean score delta and bid make rate across rollouts become the training targets for that (state, card) pair.

The "rollout strategy" defaults to the rule-based `Balanced` config. Once a trained model exists, you can pass `--model` to `cmd/collect` and the MLP itself becomes the rollout strategy, allowing iterative self-improvement.

### Feature extraction

Each training row is a (decision point, candidate card) pair with 43 features total, split into:

**Context features (37)** shared across all legal cards at a given decision point:
- Position and role: seat relative to bidder, whether on bidding team, position in trick order
- Trick progress: trick number, tricks taken by each team, tricks needed to make/set bid
- Current trick state: cards played so far, who is winning, winning trump rank
- Hand composition: trump count, right/left bower holdings, off-suit aces, voids, singletons
- Trump field knowledge: trump remaining/played, bowers played, whether best trump is top
- Score context: team scores, score gap, whether either team is in closeout range
- Bid context: normalized bid amount

**Per-card features (6)** appended once per candidate card:
- Whether the card is trump
- Normalized rank (trump rank / 13 or off-suit rank / 6)
- Whether the card is the current top unplayed trump
- Whether the card beats the current trick winner
- Whether playing this card is an overtrump (spending trump to beat trump)
- How many cards of this suit remain in hand

Context features are computed once per decision point and reused for all candidate cards, so feature extraction cost scales as `O(hand_state) + O(candidates)` rather than `O(hand_state * candidates)`.

All features are normalized to roughly [0, 1] or [-1, 1] to keep network inputs well-conditioned. The normalization denominators are defined as named constants (`card.TotalTrumpCards`, `card.TrumpRankRight`, `game.WinScore`, etc.) so they stay in sync with the game rules.

### Model architecture

Both the card-play and bid models are 2-hidden-layer MLPs with ReLU activations:

```
Card play:  43 features → 128 → 64 → 1  (predicts score_delta for bidding team)
Bidding:    20 features → 128 → 64 → 1  (predicts score_delta for this seat's team)
```

The output is a single scalar: expected score delta for the relevant team. Targets are z-score normalized during training and the model stores `y_mean` and `y_std` in the weights file for denormalization at inference time.

### Training pipeline

**Step 1: Collect data**
```bash
go run ./cmd/collect -n 100000 -rollouts 10 -workers 8 -out play_data.csv
go run ./cmd/collect-bid -n 100000 -rollouts 5 -workers 8 -out bid_data.csv
```

Each hand produces one row per (decision point, candidate card) pair. With 100k hands, rollout=10, and ~3 decisions/trick * 8 tricks * ~3 candidates each, you get roughly 7M rows for card play.

Workers run in parallel with independent RNGs seeded from the base seed. The CSV writer runs in a dedicated goroutine to avoid blocking the collection workers.

**Step 2: Train**
```bash
python3 ml/train.py --data play_data.csv --out model_weights.json
python3 ml/train_bid.py --data bid_data.csv --out bid_model_weights.json
```

The trainer does two passes over the CSV:
1. Compute target mean and standard deviation for normalization
2. Train for 30 epochs with Adam + cosine LR decay, saving the best validation checkpoint

The CSV is streamed in chunks (500k rows at a time) so large datasets never fully load into memory. The last 10% of rows are held out as a validation set.

**Step 3: Iterate**
```bash
go run ./cmd/collect --model model_weights.json -n 100000 -rollouts 10 -out play_data_v2.csv
python3 ml/train.py --data play_data_v2.csv --out model_v2_weights.json
```

Passing `--model` to the collector replaces the rule-based rollout strategy with the MLP itself. Each generation produces a model that plays against itself during data collection, allowing the bot to improve beyond the ceiling of the rule-based strategy.

### Bidding features

Bid decisions use a separate 20-feature vector:

**Context (17):** best trump suit statistics (count, right/left bower, top rank, off-suit aces, voids/singletons, suit dominance vs second suit), bidding position, current high bid, score context.

**Per-action (3):** normalized bid level, is-pass flag, is-pepper flag.

At inference time, all valid bid levels (pass, 4–7, pepper) are scored and the best one wins. The bid MLP is optional; if not loaded, bidding falls back to the rule-based strategy.

### Evolutionary strategy tuning

Before MLP training, the rule-based strategy's 30+ parameters were tuned using a 3-phase evolutionary search (`cmd/evolve`):

1. **Random search**: evaluate 200 random parameter sets against the baseline, keep top 20
2. **Tournament**: round-robin tournament among the top 20, determine rankings
3. **Evolutionary refinement**: 20 generations of mutation + crossover among top survivors, with decaying mutation strength

This produced the `Balanced` config embedded in `internal/strategy/standard.go`, which serves as the baseline and rollout strategy for MLP training.

### Simulation and evaluation

`cmd/simulate` runs head-to-head matchups between strategies over thousands of games and reports win rates, bid accuracy, and pepper success rates. Use this to verify that a new model actually beats the previous one before deploying it.

---

## HTTP API

The bot server exposes three endpoints:

### POST /bid
```json
{ "seat": 2, "hand": ["JH","JD","AH","KH","QH","AD","KD","QD"],
  "bid_state": { "current_high": 5, "dealer_seat": 0, "scores": [10, 8] } }
→ { "bid": 6 }
```

### POST /trump
```json
{ "seat": 2, "hand": ["JH","JD","AH","KH","QH","AD","KD","QD"] }
→ { "suit": "H" }
```

### POST /play
```json
{ "seat": 2, "valid_plays": ["AH","KH"],
  "state": { "trump": "H", "bidder_seat": 2, "bid_amount": 5,
             "trick_number": 3, "tricks_taken": [1,1,1,0,0,0],
             "scores": [10,8], "hand": ["AH","KH","QH"],
             "trick": [{"card":"9H","seat":1}], "leader": 1,
             "history": [["JH","9D","TC","KS","AS","QH"], ...] } }
→ { "card": "AH" }
```

Card codes: rank prefix (`9`, `T`, `J`, `Q`, `K`, `A`) + suit suffix (`H`, `D`, `C`, `S`).

The server uses the caller-supplied `valid_plays` directly so copy-index accounting (tracking which of the two identical cards has been played) stays with the calling system.

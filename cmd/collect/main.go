package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/mlstrategy"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

func main() {
	n         := flag.Int("n", 10000, "number of hands to collect")
	rollouts  := flag.Int("rollouts", 10, "counterfactual rollouts per candidate card")
	workers   := flag.Int("workers", 6, "parallel workers")
	seed         := flag.Int64("seed", 42, "random seed")
	out          := flag.String("out", "collect.csv", "output path")
	format       := flag.String("format", "csv", "output format: csv or bin")
	modelPath    := flag.String("model", "", "path to model_weights.json; use MLP for play instead of Balanced")
	bidModelPath := flag.String("bid-model", "", "path to bid_model_weights.json; use bid MLP for bidding")
	epsilon         := flag.Float64("epsilon", 0.0, "probability of playing a random card (exploration)")
	rolloutEpsilon  := flag.Float64("rollout-epsilon", 0.0, "probability of random card in rollout strategies")
	flag.Parse()

	var baseModel *ml.MLP
	if *modelPath != "" {
		var err error
		baseModel, err = ml.LoadMLP(*modelPath)
		if err != nil {
			fmt.Printf("  Model:    %s (incompatible, falling back to Balanced: %v)\n", *modelPath, err)
		} else {
			fmt.Printf("  Model:    %s (MLP self-play)\n", *modelPath)
		}
	}
	var baseBidModel *ml.BidMLP
	if *bidModelPath != "" {
		var err error
		baseBidModel, err = ml.LoadBidMLP(*bidModelPath)
		if err != nil {
			fmt.Printf("  Bid model: %s (incompatible, falling back to Balanced: %v)\n", *bidModelPath, err)
		} else {
			fmt.Printf("  Bid model: %s\n", *bidModelPath)
		}
	}

	fmt.Printf("Collecting training data:\n")
	fmt.Printf("  Hands:    %d\n", *n)
	fmt.Printf("  Rollouts: %d per card\n", *rollouts)
	fmt.Printf("  Workers:  %d\n", *workers)
	fmt.Printf("  Seed:     %d\n", *seed)
	fmt.Printf("  Epsilon:  %.2f (play)  %.2f (rollout)\n", *epsilon, *rolloutEpsilon)
	fmt.Printf("  Output:   %s (%s)\n\n", *out, *format)

	// Open output file.
	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create output: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	// Use buffered I/O for performance.
	bufW := bufio.NewWriter(f)
	defer bufW.Flush()

	var csvW *csv.Writer
	if *format == "csv" {
		csvW = csv.NewWriter(bufW)
		header := buildHeader()
		if err := csvW.Write(header); err != nil {
			fmt.Fprintf(os.Stderr, "header write failed: %v\n", err)
			os.Exit(1)
		}
	}

	// Channel for rows produced by workers.
	rowCh := make(chan []ml.CollectRow, *workers*4)

	// Writer goroutine — flushes rows from channel to output.
	var writeErr error
	var writeDone sync.WaitGroup
	writeDone.Add(1)
	go func() {
		defer writeDone.Done()
		for rows := range rowCh {
			for _, row := range rows {
				if *format == "csv" {
					if err := csvW.Write(encodeRow(row)); err != nil {
						writeErr = err
						return
					}
				} else {
					if err := row.WriteBinary(bufW); err != nil {
						writeErr = err
						return
					}
				}
			}
		}
		if csvW != nil {
			csvW.Flush()
			writeErr = csvW.Error()
		}
	}()

	// Distribute hands across workers.
	perWorker := *n / *workers
	var totalHands atomic.Int64
	var wg sync.WaitGroup

	for wid := 0; wid < *workers; wid++ {
		wg.Add(1)
		go func(wid int) {
			defer wg.Done()

			rng := rand.New(rand.NewSource(*seed + int64(wid*1_000_000)))
			gs := game.NewGame(rng.Intn(6))

			// One card+bid MLP clone per worker — all 6 seats share it since
			// CollectHand runs sequentially within a goroutine.
			var strats [6]game.Strategy
			var rolloutStrats [6]game.Strategy
			if baseModel != nil {
				sharedPlay := mlstrategy.NewMLPStrategy(baseModel.Clone(), strategy.Balanced)
				if baseBidModel != nil {
					sharedPlay = sharedPlay.WithBidModel(baseBidModel.Clone())
				}
				var playStrat game.Strategy = sharedPlay
				if *epsilon > 0 {
					playStrat = &epsilonStrategy{inner: sharedPlay, epsilon: *epsilon, rng: rng}
				}
				sharedRollout := mlstrategy.NewMLPStrategy(baseModel.Clone(), strategy.Balanced)
				if baseBidModel != nil {
					sharedRollout = sharedRollout.WithBidModel(baseBidModel.Clone())
				}
				var rolloutStrat game.Strategy = sharedRollout
				if *rolloutEpsilon > 0 {
					rolloutStrat = &epsilonStrategy{inner: sharedRollout, epsilon: *rolloutEpsilon, rng: rng}
				}
				for s := 0; s < 6; s++ {
					strats[s] = playStrat
					rolloutStrats[s] = rolloutStrat
				}
			} else {
				shared := strategy.NewStandard(strategy.Balanced)
				for s := 0; s < 6; s++ {
					strats[s] = shared
					rolloutStrats[s] = shared
				}
			}

			for i := 0; i < perWorker; i++ {
				handID := wid*perWorker + i
				rows := ml.CollectHand(handID, gs, strats, rolloutStrats, rng, *rollouts)
				if len(rows) > 0 {
					rowCh <- rows
				}

				gs.NextDealer()
				if gs.Round%50 == 0 {
					gs = game.NewGame(rng.Intn(6))
					gs.Scores[0] = rng.Intn(64)
					gs.Scores[1] = rng.Intn(64)
				}

				// Progress report from worker 0.
				if wid == 0 {
					done := totalHands.Add(1)
					if done%1000 == 0 {
						fmt.Printf("  %d / %d hands\n", done, int64(*n))
					}
				} else {
					totalHands.Add(1)
				}
			}
		}(wid)
	}

	wg.Wait()
	close(rowCh)
	writeDone.Wait()

	if writeErr != nil {
		fmt.Fprintf(os.Stderr, "write error: %v\n", writeErr)
		os.Exit(1)
	}

	fmt.Printf("\nDone. Data written to %s\n", *out)
}

type epsilonStrategy struct {
	inner   game.Strategy
	epsilon float64
	rng     *rand.Rand
}

func (s *epsilonStrategy) Bid(seat int, state *game.BidState) int { return s.inner.Bid(seat, state) }
func (s *epsilonStrategy) ChooseTrump(seat int, hand []card.Card) card.Suit { return s.inner.ChooseTrump(seat, hand) }
func (s *epsilonStrategy) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card { return s.inner.GivePepper(seat, hand, trump) }
func (s *epsilonStrategy) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card { return s.inner.PepperDiscard(seat, hand, trump, received) }
func (s *epsilonStrategy) Play(seat int, validPlays []card.Card, state *game.TrickState) card.Card {
	if len(validPlays) > 1 && s.rng.Float64() < s.epsilon {
		return validPlays[s.rng.Intn(len(validPlays))]
	}
	return s.inner.Play(seat, validPlays, state)
}

// buildHeader returns the CSV column names.
func buildHeader() []string {
	cols := []string{"hand_id", "trick_num", "seat", "is_bidding_team"}
	for _, name := range ml.FeatureNames {
		cols = append(cols, name)
	}
	cols = append(cols, "score_delta", "made_bid_rate")
	return cols
}

// encodeRow converts a CollectRow to a CSV string slice.
func encodeRow(r ml.CollectRow) []string {
	row := make([]string, 0, 4+ml.TotalFeatureLen+2)
	row = append(row,
		strconv.Itoa(r.HandID),
		strconv.Itoa(r.TrickNumber),
		strconv.Itoa(r.Seat),
		boolStr(r.IsBiddingTeam),
	)
	for _, v := range r.Features {
		row = append(row, strconv.FormatFloat(float64(v), 'f', 6, 32))
	}
	row = append(row,
		strconv.FormatFloat(float64(r.ScoreDelta), 'f', 4, 32),
		strconv.FormatFloat(float64(r.MadeBidRate), 'f', 4, 32),
	)
	return row
}

func boolStr(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

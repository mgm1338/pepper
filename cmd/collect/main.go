package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/mlstrategy"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

func main() {
	n         := flag.Int("n", 10000, "number of hands to collect")
	rollouts  := flag.Int("rollouts", 10, "counterfactual rollouts per candidate card")
	workers   := flag.Int("workers", 6, "parallel workers")
	seed      := flag.Int64("seed", 42, "random seed")
	out       := flag.String("out", "collect.csv", "output CSV path")
	modelPath    := flag.String("model", "", "path to model_weights.json; use MLP for play instead of Balanced")
	bidModelPath := flag.String("bid-model", "", "path to bid_model_weights.json; use bid MLP for bidding")
	flag.Parse()

	var baseModel *ml.MLP
	if *modelPath != "" {
		var err error
		baseModel, err = ml.LoadMLP(*modelPath)
		if err != nil {
			log.Fatalf("load MLP: %v", err)
		}
		fmt.Printf("  Model:    %s (MLP self-play)\n", *modelPath)
	}
	var baseBidModel *ml.BidMLP
	if *bidModelPath != "" {
		var err error
		baseBidModel, err = ml.LoadBidMLP(*bidModelPath)
		if err != nil {
			log.Fatalf("load bid MLP: %v", err)
		}
		fmt.Printf("  Bid model: %s\n", *bidModelPath)
	}

	fmt.Printf("Collecting training data:\n")
	fmt.Printf("  Hands:    %d\n", *n)
	fmt.Printf("  Rollouts: %d per card\n", *rollouts)
	fmt.Printf("  Workers:  %d\n", *workers)
	fmt.Printf("  Seed:     %d\n", *seed)
	fmt.Printf("  Output:   %s\n\n", *out)

	// Open output file and write CSV header.
	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create output: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	header := buildHeader()
	if err := w.Write(header); err != nil {
		fmt.Fprintf(os.Stderr, "header write failed: %v\n", err)
		os.Exit(1)
	}

	// Channel for rows produced by workers.
	rowCh := make(chan []ml.CollectRow, *workers*4)

	// Writer goroutine — flushes rows from channel to CSV.
	var writeErr error
	var writeDone sync.WaitGroup
	writeDone.Add(1)
	go func() {
		defer writeDone.Done()
		for rows := range rowCh {
			for _, row := range rows {
				if err := w.Write(encodeRow(row)); err != nil {
					writeErr = err
					return
				}
			}
		}
		w.Flush()
		writeErr = w.Error()
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

			// Each worker gets its own MLP clone (independent scratch buffers).
			// Bidding/trump/pepper always use Balanced fallback.
			var strats [6]game.Strategy
			var rolloutStrats [6]game.Strategy
			for s := 0; s < 6; s++ {
				if baseModel != nil {
					play := mlstrategy.NewMLPStrategy(baseModel.Clone(), strategy.Balanced)
					if baseBidModel != nil {
						play = play.WithBidModel(baseBidModel)
					}
					strats[s] = play
					rollout := mlstrategy.NewMLPStrategy(baseModel.Clone(), strategy.Balanced)
					if baseBidModel != nil {
						rollout = rollout.WithBidModel(baseBidModel)
					}
					rolloutStrats[s] = rollout
				} else {
					strats[s] = strategy.NewStandard(strategy.Balanced)
					rolloutStrats[s] = strategy.NewStandard(strategy.Balanced)
				}
			}

			for i := 0; i < perWorker; i++ {
				handID := wid*perWorker + i
				rows := ml.CollectHand(handID, gs, strats, rolloutStrats, rng, *rollouts)
				if len(rows) > 0 {
					rowCh <- rows
				}

				// Advance game state.
				// We don't track actual scores here since CollectHand uses neutral
				// scores for feature extraction. Reset game periodically.
				gs.NextDealer()
				if gs.Round%50 == 0 {
					gs = game.NewGame(rng.Intn(6))
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

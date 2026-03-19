// Package main collects bid decision training data via counterfactual rollouts.
//
// For each hand, it intercepts every bid decision and evaluates each valid bid
// level (pass, 4-7, pepper) by simulating the rest of the auction and playing
// the full hand N times. The expected score delta for the bidding seat's team
// is recorded as the training target.
//
// Usage:
//
//	collect-bid -n 50000 -rollouts 20 -workers 8 -out bid_training.csv
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

func main() {
	n        := flag.Int("n", 50000, "number of hands to collect")
	rollouts := flag.Int("rollouts", 20, "counterfactual rollouts per bid level")
	workers  := flag.Int("workers", 8, "parallel workers")
	seed     := flag.Int64("seed", 42, "random seed")
	out      := flag.String("out", "bid_training.csv", "output CSV path")
	flag.Parse()

	fmt.Printf("Collecting bid training data:\n")
	fmt.Printf("  Hands:    %d\n", *n)
	fmt.Printf("  Rollouts: %d per bid level\n", *rollouts)
	fmt.Printf("  Workers:  %d\n", *workers)
	fmt.Printf("  Seed:     %d\n", *seed)
	fmt.Printf("  Output:   %s\n\n", *out)

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create output: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	if err := w.Write(buildHeader()); err != nil {
		fmt.Fprintf(os.Stderr, "header write failed: %v\n", err)
		os.Exit(1)
	}

	rowCh := make(chan []ml.BidCollectRow, *workers*4)

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

	perWorker := *n / *workers
	var totalHands atomic.Int64
	var wg sync.WaitGroup

	for wid := 0; wid < *workers; wid++ {
		wg.Add(1)
		go func(wid int) {
			defer wg.Done()

			rng := rand.New(rand.NewSource(*seed + int64(wid*1_000_000)))
			gs := game.NewGame(rng.Intn(6))

			// Vary scores slightly across workers so the model sees different score contexts.
			if wid > 0 {
				gs.Scores[0] = rng.Intn(64)
				gs.Scores[1] = rng.Intn(64)
			}

			var strats [6]game.Strategy
			var rolloutStrats [6]game.Strategy
			for s := 0; s < 6; s++ {
				strats[s] = strategy.NewStandard(strategy.Balanced)
				rolloutStrats[s] = strategy.NewStandard(strategy.Balanced)
			}

			for i := 0; i < perWorker; i++ {
				handID := wid*perWorker + i
				rows := ml.CollectBidHand(handID, gs, strats, rolloutStrats, rng, *rollouts)
				if len(rows) > 0 {
					rowCh <- rows
				}

				gs.NextDealer()
				if gs.Round%50 == 0 {
					gs = game.NewGame(rng.Intn(6))
					// Randomize scores for variety.
					gs.Scores[0] = rng.Intn(64)
					gs.Scores[1] = rng.Intn(64)
				}

				if wid == 0 {
					done := totalHands.Add(1)
					if done%1000 == 0 {
						fmt.Printf("  %d / %d hands\n", done*int64(*workers), int64(*n))
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

func buildHeader() []string {
	cols := []string{"hand_id", "seat", "bid_level"}
	for _, name := range ml.BidFeatureNames {
		cols = append(cols, name)
	}
	cols = append(cols, "score_delta")
	return cols
}

func encodeRow(r ml.BidCollectRow) []string {
	row := make([]string, 0, 3+ml.BidTotalLen+1)
	row = append(row,
		strconv.Itoa(r.HandID),
		strconv.Itoa(r.Seat),
		strconv.Itoa(r.BidLevel),
	)
	for _, v := range r.Features {
		row = append(row, strconv.FormatFloat(float64(v), 'f', 6, 32))
	}
	row = append(row, strconv.FormatFloat(float64(r.ScoreDelta), 'f', 4, 32))
	return row
}

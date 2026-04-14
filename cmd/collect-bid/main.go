package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
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

type epsilonStrategy struct {
	inner   game.Strategy
	epsilon float64
	rng     *rand.Rand
}

func (s *epsilonStrategy) Bid(seat int, state game.BidState) int { return s.inner.Bid(seat, state) }
func (s *epsilonStrategy) ChooseTrump(seat int, hand []card.Card) card.Suit { return s.inner.ChooseTrump(seat, hand) }
func (s *epsilonStrategy) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card { return s.inner.GivePepper(seat, hand, trump) }
func (s *epsilonStrategy) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card { return s.inner.PepperDiscard(seat, hand, trump, received) }
func (s *epsilonStrategy) Play(seat int, validPlays []card.Card, state game.TrickState) card.Card {
	if len(validPlays) > 1 && s.rng.Float64() < s.epsilon {
		return validPlays[s.rng.Intn(len(validPlays))]
	}
	return s.inner.Play(seat, validPlays, state)
}

func main() {
	n         := flag.Int("n", 50000, "number of hands to collect")
	rollouts  := flag.Int("rollouts", 20, "counterfactual rollouts per bid level")
	workers   := flag.Int("workers", 8, "parallel workers")
	seed      := flag.Int64("seed", 42, "random seed")
	out       := flag.String("out", "bid_training.csv", "output path")
	format    := flag.String("format", "csv", "output format: csv or bin")
	modelPath    := flag.String("model", "", "path to card-play model_weights.json for rollouts")
	bidModelPath := flag.String("bid-model", "", "path to bid_model_weights.json for rollouts")
	epsilon      := flag.Float64("epsilon", 0.0, "probability of playing a random card during rollouts")
	flag.Parse()

	var cardModel *ml.MLP
	if *modelPath != "" {
		var err error
		cardModel, err = ml.LoadMLP(*modelPath)
		if err != nil { log.Fatalf("load card model: %v", err) }
	}
	var bidModel *ml.BidMLP
	if *bidModelPath != "" {
		var err error
		bidModel, err = ml.LoadBidMLP(*bidModelPath)
		if err != nil { log.Fatalf("load bid model: %v", err) }
	}

	fmt.Printf("Collecting bid training data:\n")
	fmt.Printf("  Hands:    %d\n", *n)
	fmt.Printf("  Rollouts: %d per bid level\n", *rollouts)
	fmt.Printf("  Workers:  %d\n", *workers)
	fmt.Printf("  Seed:     %d\n", *seed)
	fmt.Printf("  Output:   %s (%s)\n", *out, *format)

	f, err := os.Create(*out)
	if err != nil { log.Fatalf("cannot create output: %v", err) }
	defer f.Close()

	bufW := bufio.NewWriter(f)
	defer bufW.Flush()

	var csvW *csv.Writer
	if *format == "csv" {
		csvW = csv.NewWriter(bufW)
		csvW.Write(buildHeader())
	}

	rowCh := make(chan []ml.BidCollectRow, *workers*4)
	var writeErr error
	var writeDone sync.WaitGroup
	writeDone.Add(1)
	go func() {
		defer writeDone.Done()
		for rows := range rowCh {
			for _, row := range rows {
				if *format == "csv" {
					if err := csvW.Write(encodeRow(row)); err != nil { writeErr = err; return }
				} else {
					if err := row.WriteBinary(bufW); err != nil { writeErr = err; return }
				}
			}
		}
		if csvW != nil { csvW.Flush(); writeErr = csvW.Error() }
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
			if wid > 0 { gs.Scores[0] = rng.Intn(64); gs.Scores[1] = rng.Intn(64) }

			var strats [6]game.Strategy
			var rolloutStrats [6]game.Strategy
			for s := 0; s < 6; s++ {
				strats[s] = strategy.NewStandard(strategy.Balanced)
				var base game.Strategy
				if cardModel != nil {
					rollout := mlstrategy.NewMLPStrategy(cardModel.Clone(), strategy.Balanced)
					if bidModel != nil { rollout = rollout.WithBidModel(bidModel) }
					base = rollout
				} else {
					base = strategy.NewStandard(strategy.Balanced)
				}
				if *epsilon > 0 { rolloutStrats[s] = &epsilonStrategy{inner: base, epsilon: *epsilon, rng: rng} } else { rolloutStrats[s] = base }
			}

			for i := 0; i < perWorker; i++ {
				handID := wid*perWorker + i
				rows := ml.CollectBidHand(handID, gs, strats, rolloutStrats, rng, *rollouts)
				if len(rows) > 0 { rowCh <- rows }
				gs.NextDealer()
				if gs.Round%50 == 0 {
					gs = game.NewGame(rng.Intn(6))
					gs.Scores[0] = rng.Intn(64); gs.Scores[1] = rng.Intn(64)
				}
				if wid == 0 {
					done := totalHands.Add(1)
					if done%1000 == 0 { fmt.Printf("  %d / %d hands\n", done, int64(*n)) }
				} else { totalHands.Add(1) }
			}
		}(wid)
	}

	wg.Wait()
	close(rowCh)
	writeDone.Wait()
	if writeErr != nil { log.Fatalf("write error: %v", writeErr) }
	fmt.Printf("\nDone. Data written to %s\n", *out)
}

func buildHeader() []string {
	cols := []string{"hand_id", "seat", "bid_level"}
	for _, name := range ml.BidFeatureNames { cols = append(cols, name) }
	cols = append(cols, "score_delta")
	return cols
}

func encodeRow(r ml.BidCollectRow) []string {
	row := make([]string, 0, 3+ml.BidTotalLen+1)
	row = append(row, strconv.Itoa(r.HandID), strconv.Itoa(r.Seat), strconv.Itoa(r.BidLevel))
	for _, v := range r.Features { row = append(row, strconv.FormatFloat(float64(v), 'f', 6, 32)) }
	row = append(row, strconv.FormatFloat(float64(r.ScoreDelta), 'f', 4, 32))
	return row
}

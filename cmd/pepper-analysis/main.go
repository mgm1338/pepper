package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
)

// pepperRecord tracks one pepper attempt and its outcome.
type pepperRecord struct {
	// Caller's hand before exchange.
	RightBowers int
	LeftBowers  int
	OtherTrump  int
	TotalTrump  int // RR + LL + OtherTrump
	Voids       int
	Singletons  int

	// What the caller received from partners.
	ReceivedTrump int // how many of the 2 received cards were trump

	// Outcome.
	TricksTaken int
	Made        bool // took all 8
}

// profileKey is a grouping key for aggregating results.
type profileKey struct {
	RR, LL, Other, Voids int
}

type profileStats struct {
	Key      profileKey
	Attempts int
	Made     int
	TotalTricks int
}

func (s *profileStats) WinRate() float64 {
	if s.Attempts == 0 {
		return 0
	}
	return float64(s.Made) / float64(s.Attempts)
}

func (s *profileStats) AvgTricks() float64 {
	if s.Attempts == 0 {
		return 0
	}
	return float64(s.TotalTricks) / float64(s.Attempts)
}

// alwaysPepperStrategy wraps a standard strategy but always calls pepper.
type alwaysPepperStrategy struct {
	inner *strategy.StandardStrategy
}

func (a *alwaysPepperStrategy) Bid(seat int, state game.BidState) int {
	// Always call pepper when it's our turn to act.
	return game.PepperBid
}
func (a *alwaysPepperStrategy) Play(seat int, hand []card.Card, state game.TrickState) card.Card {
	return a.inner.Play(seat, hand, state)
}
func (a *alwaysPepperStrategy) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card {
	return a.inner.GivePepper(seat, hand, trump)
}
func (a *alwaysPepperStrategy) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
	return a.inner.PepperDiscard(seat, hand, trump, received)
}
func (a *alwaysPepperStrategy) ChooseTrump(seat int, hand []card.Card) card.Suit {
	return a.inner.ChooseTrump(seat, hand)
}

func main() {
	n       := flag.Int("n", 200000, "number of pepper hands to simulate")
	workers := flag.Int("workers", 6, "parallel workers")
	seed    := flag.Int64("seed", 42, "random seed")
	out     := flag.String("out", "pepper_analysis.csv", "output CSV")
	flag.Parse()

	fmt.Printf("Pepper analysis: %d hands, %d workers\n\n", *n, *workers)

	records := runPepperSim(*n, *seed, *workers)

	// Aggregate by profile.
	grouped := map[profileKey]*profileStats{}
	for _, r := range records {
		k := profileKey{r.RightBowers, r.LeftBowers, r.OtherTrump, r.Voids}
		if _, ok := grouped[k]; !ok {
			grouped[k] = &profileStats{Key: k}
		}
		s := grouped[k]
		s.Attempts++
		s.TotalTricks += r.TricksTaken
		if r.Made {
			s.Made++
		}
	}

	// Sort by total trump desc, then RR desc, then LL desc.
	var stats []*profileStats
	for _, s := range grouped {
		stats = append(stats, s)
	}
	sort.Slice(stats, func(i, j int) bool {
		ti := stats[i].Key.RR + stats[i].Key.LL + stats[i].Key.Other
		tj := stats[j].Key.RR + stats[j].Key.LL + stats[j].Key.Other
		if ti != tj { return ti > tj }
		if stats[i].Key.RR != stats[j].Key.RR { return stats[i].Key.RR > stats[j].Key.RR }
		if stats[i].Key.LL != stats[j].Key.LL { return stats[i].Key.LL > stats[j].Key.LL }
		return stats[i].Key.Voids > stats[j].Key.Voids
	})

	// Print summary.
	fmt.Println("=== Pepper Success Rate by Hand Profile ===")
	fmt.Println("  RR = right bowers held   LL = left bowers held")
	fmt.Println("  Other = other trump cards (non-bower)")
	fmt.Println("  Voids = void non-trump suits (ruffing power)")
	fmt.Println("  Win% = took all 8 tricks   AvgTricks = average tricks taken")
	fmt.Println("  ExpVal = expected score per pepper call (+16 if made, -16 caller + 2×opp tricks if missed)")
	fmt.Println()
	fmt.Printf("  %-5s %-5s %-6s %-6s %-7s %-10s %-10s %-10s %s\n",
		"RR", "LL", "Other", "Voids", "Trump", "Win%", "AvgTricks", "ExpVal", "N")
	fmt.Println("  " + strings.Repeat("-", 68))
	for _, s := range stats {
		if s.Attempts < 50 {
			continue
		}
		trump := s.Key.RR + s.Key.LL + s.Key.Other
		ev := expectedValue(records, s.Key)
		fmt.Printf("  %-5d %-5d %-6d %-6d %-7d %-9.1f%% %-10.2f %-10.2f %d\n",
			s.Key.RR, s.Key.LL, s.Key.Other, s.Key.Voids,
			trump, s.WinRate()*100, s.AvgTricks(), ev, s.Attempts)
	}

	// Write CSV.
	writeCSV(*out, stats, records)
	fmt.Printf("\nFull results written to %s\n", *out)
}

func expectedValue(records []pepperRecord, k profileKey) float64 {
	total := 0.0
	n := 0
	for _, r := range records {
		if r.RightBowers != k.RR || r.LeftBowers != k.LL || r.OtherTrump != k.Other || r.Voids != k.Voids {
			continue
		}
		n++
		if r.Made {
			total += 16
		} else {
			// Caller loses 16. Opponents get 2 per trick.
			oppTricks := 8 - r.TricksTaken
			total += -16 // caller's score
			_ = oppTricks // opponents' gain is separate; show caller EV only
		}
	}
	if n == 0 {
		return 0
	}
	return total / float64(n)
}

func runPepperSim(n int, seed int64, workers int) []pepperRecord {
	perWorker := n / workers
	results := make([][]pepperRecord, workers)
	done := make(chan int, workers)

	for w := 0; w < workers; w++ {
		go func(wid int) {
			rng := rand.New(rand.NewSource(seed + int64(wid*1000000)))
			var recs []pepperRecord

			pepperCaller := &alwaysPepperStrategy{inner: strategy.NewStandard(strategy.Balanced)}
			normal := strategy.NewStandard(strategy.Balanced)

			for i := 0; i < perWorker; i++ {
				rec, ok := runOnePepperHand(rng, pepperCaller, normal)
				if ok {
					recs = append(recs, rec)
				}
			}
			results[wid] = recs
			done <- wid
		}(w)
	}

	for range make([]struct{}, workers) {
		<-done
	}

	var all []pepperRecord
	for _, r := range results {
		all = append(all, r...)
	}
	return all
}

// runOnePepperHand deals a hand and forces seat 0 to call pepper.
// Returns the record and true if pepper was actually called (seat 0 won the bid).
func runOnePepperHand(rng *rand.Rand, pepperCaller game.Strategy, normal game.Strategy) (pepperRecord, bool) {
	// Force dealer = seat 5 so seat 0 bids first and can call pepper immediately.
	gs := game.NewGame(5)

	var strategies [6]game.Strategy
	strategies[0] = pepperCaller
	for i := 1; i < 6; i++ {
		strategies[i] = normal
	}

	// We need to intercept the hand to capture the profile.
	// Run a hand and check if seat 0 was the pepper caller.
	hands := card.Deal(rng)

	// Force seat 0 to pepper via the bidding round.
	bidResult := game.RunBidding(
		hands,
		gs.Dealer,
		gs.Scores,
		func(seat int, state game.BidState) int {
			return strategies[seat].Bid(seat, state)
		},
	)

	if !bidResult.IsPepper || bidResult.Winner != 0 {
		return pepperRecord{}, false
	}

	// Capture seat 0's hand profile before exchange.
	trump := strategies[0].ChooseTrump(0, hands[0])
	profile := buildProfile(hands[0], trump)

	// Play out THIS hand (same cards, same bid result) — not a new deal.
	result := game.PlayHandFrom(gs, strategies, rng, game.NoopLogger{}, hands, bidResult)

	return pepperRecord{
		RightBowers:   profile.RR,
		LeftBowers:    profile.LL,
		OtherTrump:    profile.Other,
		TotalTrump:    profile.RR + profile.LL + profile.Other,
		Voids:         profile.Voids,
		TricksTaken:   result.TricksTaken[int(game.TeamOf(0))],
		Made:          result.MadeBid,
	}, true
}

type hProfile struct{ RR, LL, Other, Voids, Singletons int }

func buildProfile(hand []card.Card, trump card.Suit) hProfile {
	var p hProfile
	suitCounts := [4]int{}
	for _, c := range hand {
		r := card.TrumpRank(c, trump)
		if r >= 0 {
			switch {
			case card.IsRightBower(c, trump):
				p.RR++
			case card.IsLeftBower(c, trump):
				p.LL++
			default:
				p.Other++
			}
		} else {
			suitCounts[c.Suit]++
		}
	}
	suits := []card.Suit{card.Spades, card.Clubs, card.Hearts, card.Diamonds}
	for _, s := range suits {
		if s == trump {
			continue
		}
		switch suitCounts[s] {
		case 0:
			p.Voids++
		case 1:
			p.Singletons++
		}
	}
	// Also check partner suit (left bower removed from it).
	partner := card.PartnerSuit(trump)
	switch suitCounts[partner] {
	case 0:
		p.Voids++
	case 1:
		p.Singletons++
	}
	return p
}

func writeCSV(path string, stats []*profileStats, records []pepperRecord) {
	f, _ := os.Create(path)
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"rr", "ll", "other_trump", "voids", "total_trump", "attempts", "made", "win_rate", "avg_tricks", "exp_val"})
	for _, s := range stats {
		if s.Attempts < 10 {
			continue
		}
		trump := s.Key.RR + s.Key.LL + s.Key.Other
		ev := expectedValue(records, s.Key)
		w.Write([]string{
			strconv.Itoa(s.Key.RR),
			strconv.Itoa(s.Key.LL),
			strconv.Itoa(s.Key.Other),
			strconv.Itoa(s.Key.Voids),
			strconv.Itoa(trump),
			strconv.Itoa(s.Attempts),
			strconv.Itoa(s.Made),
			fmt.Sprintf("%.4f", s.WinRate()),
			fmt.Sprintf("%.3f", s.AvgTricks()),
			fmt.Sprintf("%.3f", ev),
		})
	}
	w.Flush()
}

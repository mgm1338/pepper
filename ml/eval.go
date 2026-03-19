package ml

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/sim"
)

// Result holds a genome and its fitness score against a baseline.
type Result struct {
	Genome       Genome
	WinRate      float64 // kept for tournament display; for paired eval = AvgAdvantage normalized
	AvgAdvantage float64 // average per-hand score advantage over baseline (paired eval)
	Games        int
	Rank         int
}

func (r Result) String() string {
	return fmt.Sprintf("adv=%.3f win=%.1f%% hands=%d | %s", r.AvgAdvantage, r.WinRate*100, r.Games, r.Genome)
}

// Evaluate runs n paired hands: same deal played by genome (team0) and opponent (team0).
// Fitness = average per-hand score advantage of genome over opponent.
// This removes card-luck variance and measures pure decision quality.
func Evaluate(genome, opponent Genome, n int, seed int64) Result {
	cfg0 := genome.ToConfig("Candidate")
	cfg1 := opponent.ToConfig("Opponent")

	factoryA := func(rng *rand.Rand) [6]game.Strategy {
		var strats [6]game.Strategy
		for seat := 0; seat < 6; seat++ {
			if seat%2 == 0 {
				strats[seat] = strategy.NewStandard(cfg0)
			} else {
				strats[seat] = strategy.NewStandard(cfg1)
			}
		}
		return strats
	}
	factoryB := func(rng *rand.Rand) [6]game.Strategy {
		var strats [6]game.Strategy
		for seat := 0; seat < 6; seat++ {
			// Baseline plays team0 on same cards to measure marginal value.
			strats[seat] = strategy.NewStandard(cfg1)
		}
		return strats
	}

	pr := sim.RunPairedHands(n, factoryA, factoryB, seed)

	// Normalize advantage to a [0,1]-ish win-rate equivalent for display.
	// +8 advantage per hand (rough max) maps to ~1.0, 0 advantage = 0.5.
	normalized := 0.5 + pr.AvgAdvantage/16.0

	return Result{
		Genome:       genome,
		WinRate:      normalized,
		AvgAdvantage: pr.AvgAdvantage,
		Games:        n,
	}
}

// EvaluateParallel runs evaluations for many genomes concurrently.
func EvaluateParallel(genomes []Genome, opponent Genome, n int, baseSeed int64, workers int) []Result {
	results := make([]Result, len(genomes))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for i, g := range genomes {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, genome Genome) {
			defer wg.Done()
			defer func() { <-sem }()
			results[idx] = Evaluate(genome, opponent, n, baseSeed+int64(idx))
		}(i, g)
	}
	wg.Wait()
	return results
}

// SortByWinRate sorts results descending by win rate.
func SortByWinRate(results []Result) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].WinRate > results[j].WinRate
	})
}

// Tournament runs a round-robin among the given genomes using paired evaluation.
// Each pair plays n paired hands with A as team0, then n paired hands with B as team0.
// Fitness = total score advantage across all matchups.
func Tournament(genomes []Genome, n int, baseSeed int64, workers int) []Result {
	type matchup struct {
		i, j int
	}
	var pairs []matchup
	for i := 0; i < len(genomes); i++ {
		for j := i + 1; j < len(genomes); j++ {
			pairs = append(pairs, matchup{i, j})
		}
	}

	totalAdv := make([]float64, len(genomes))
	totalHands := make([]int, len(genomes))
	var mu sync.Mutex
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for idx, p := range pairs {
		wg.Add(1)
		sem <- struct{}{}
		go func(pairIdx int, a, b int) {
			defer wg.Done()
			defer func() { <-sem }()

			seed := baseSeed + int64(pairIdx*100)

			// A vs B: A advantage over B.
			rAB := Evaluate(genomes[a], genomes[b], n, seed)
			// B vs A: B advantage over A.
			rBA := Evaluate(genomes[b], genomes[a], n, seed+1)

			mu.Lock()
			totalAdv[a] += rAB.AvgAdvantage
			totalAdv[b] += rBA.AvgAdvantage
			totalHands[a] += n
			totalHands[b] += n
			mu.Unlock()
		}(idx, p.i, p.j)
	}
	wg.Wait()

	results := make([]Result, len(genomes))
	for i, g := range genomes {
		avg := 0.0
		if totalHands[i] > 0 {
			avg = totalAdv[i] / float64(len(genomes)-1) // normalize by matchups
		}
		results[i] = Result{
			Genome:       g,
			WinRate:      0.5 + avg/16.0,
			AvgAdvantage: avg,
			Games:        totalHands[i],
		}
	}
	SortByWinRate(results)
	for i := range results {
		results[i].Rank = i + 1
	}
	return results
}

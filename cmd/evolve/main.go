package main

import (
	"flag"
	"fmt"

	"github.com/max/pepper/ml"
)

func main() {
	p1cand  := flag.Int("p1-candidates", 100, "phase 1: number of random candidates")
	p1games := flag.Int("p1-games", 10000, "phase 1: games per candidate")
	p1keep  := flag.Int("p1-keep", 20, "phase 1: top N to keep")

	p2games := flag.Int("p2-games", 10000, "phase 2: games per pair in tournament")

	p3gens  := flag.Int("p3-gens", 10, "phase 3: number of generations")
	p3muts  := flag.Int("p3-muts", 8, "phase 3: mutations per elite per generation")
	p3games := flag.Int("p3-games", 10000, "phase 3: games per candidate per generation")
	p3keep  := flag.Int("p3-keep", 5, "phase 3: survivors per generation")

	workers    := flag.Int("workers", 6, "parallel workers for evaluation")
	seed       := flag.Int64("seed", 0, "random seed (0 = time-based)")
	out        := flag.String("out", "evolve_results.csv", "output CSV path")
	phase1only := flag.Bool("phase1-only", false, "stop after phase 1")
	phase2only := flag.Bool("phase2-only", false, "stop after phase 2 (default if -p3-gens not set)")

	flag.Parse()

	cfg := ml.DefaultEvolveConfig()
	cfg.Phase1Candidates  = *p1cand
	cfg.Phase1Games       = *p1games
	cfg.Phase1Keep        = *p1keep
	cfg.Phase2Games       = *p2games
	cfg.Phase2Workers     = *workers
	cfg.Phase3Generations = *p3gens
	cfg.Phase3MutationsEach = *p3muts
	cfg.Phase3Games       = *p3games
	cfg.Phase3Keep        = *p3keep
	cfg.Phase3Workers     = *workers
	cfg.OutFile           = *out

	if *seed != 0 {
		cfg.Seed = *seed
	}
	if *phase1only {
		cfg.Phase3Generations = 0
		cfg.Phase1Keep = 0 // signal to skip phase 2
	} else if *phase2only {
		cfg.Phase3Generations = 0
	}
	_ = phase2only

	fmt.Printf("Evolve config:\n")
	fmt.Printf("  Phase 1: %d candidates × %d games, keep top %d\n",
		cfg.Phase1Candidates, cfg.Phase1Games, cfg.Phase1Keep)
	if cfg.Phase1Keep > 0 {
		fmt.Printf("  Phase 2: %d-way tournament × %d games/pair\n",
			cfg.Phase1Keep, cfg.Phase2Games)
	}
	if cfg.Phase3Generations > 0 {
		fmt.Printf("  Phase 3: %d generations × %d mutations × %d games\n",
			cfg.Phase3Generations, cfg.Phase3MutationsEach, cfg.Phase3Games)
	}
	fmt.Printf("  Workers: %d  Seed: %d\n\n", cfg.Phase2Workers, cfg.Seed)

	ml.Run(cfg)
}

package main

import (
	"flag"
	"fmt"
	"math/rand"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
)

func main() {
	n := flag.Int("n", 1, "number of full games to trace")
	hands := flag.Int("hands", 0, "stop after this many hands total (0 = full game)")
	seed := flag.Int64("seed", 1, "random seed")
	strat := flag.String("strat", "balanced", "strategy for all players: conservative, balanced, aggressive")
	tricks := flag.Bool("tricks", false, "show every card played in each trick")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))
	log := game.NewPrintLogger(*tricks)

	var cfg strategy.Config
	switch *strat {
	case "conservative":
		cfg = strategy.Conservative
	case "aggressive":
		cfg = strategy.Aggressive
	default:
		cfg = strategy.Balanced
	}

	for g := 0; g < *n; g++ {
		fmt.Printf("\n%s\nGAME %d  (strategy: %s)\n%s\n",
			"╔══════════════════════════════════════════════════════════╗",
			g+1, cfg.Name,
			"╚══════════════════════════════════════════════════════════╝")

		gs := game.NewGame(rng.Intn(6))
		var strategies [6]game.Strategy
		for i := range strategies {
			strategies[i] = strategy.NewStandard(cfg)
		}

		handCount := 0
		for {
			fmt.Printf("\n--- Hand %d  (dealer: seat %d, score: Team0=%d Team1=%d) ---\n",
				gs.Round+1, gs.Dealer, gs.Scores[0], gs.Scores[1])
			result := game.PlayHand(gs, strategies, rng, log)

			gs.ApplyScore(game.Team0, result.ScoreDelta[0])
			gs.ApplyScore(game.Team1, result.ScoreDelta[1])
			gs.NextDealer()
			handCount++

			if *hands > 0 && handCount >= *hands {
				fmt.Printf("\n(stopped after %d hands — score: Team0=%d Team1=%d)\n",
					handCount, gs.Scores[0], gs.Scores[1])
				break
			}

			if over, winner := gs.IsOver(); over {
				log.OnGameOver(winner, gs.Scores, gs.Round)
				break
			}
		}
	}
}

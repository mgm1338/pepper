package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/mlstrategy"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
	"github.com/max/pepper/sim"
)

func main() {
	n          := flag.Int("n", 100000, "number of games per matchup")
	seed       := flag.Int64("seed", 42, "random seed")
	table      := flag.Bool("table", false, "run large balanced self-play and print full bid table")
	minRows    := flag.Int("min", 500, "minimum hand samples to show in table")
	workers    := flag.Int("workers", 6, "parallel workers for table run")
	checkpoint := flag.Int("checkpoint", 100000, "write CSV checkpoint every N games")
	modelPath  := flag.String("model", "", "path to model_weights.json; runs MLP vs Balanced paired eval")
	flag.Parse()

	if *modelPath != "" {
		model, err := ml.LoadMLP(*modelPath)
		if err != nil {
			log.Fatalf("load MLP: %v", err)
		}
		mlpFactory := func(rng *rand.Rand) [6]game.Strategy {
			var strats [6]game.Strategy
			for i := range strats {
				strats[i] = mlstrategy.NewMLPStrategy(model, strategy.Balanced)
			}
			return strats
		}
		balancedFactory := func(rng *rand.Rand) [6]game.Strategy {
			var strats [6]game.Strategy
			for i := range strats {
				strats[i] = strategy.NewStandard(strategy.Balanced)
			}
			return strats
		}
		fmt.Printf("Paired eval: MLP vs Balanced — %d hands, seed=%d\n\n", *n, *seed)
		result := sim.RunPairedHands(*n, mlpFactory, balancedFactory, *seed)
		fmt.Printf("Hands played:       %d\n", result.Hands)
		fmt.Printf("MLP avg advantage:  %+.4f pts/hand vs Balanced\n", result.AvgAdvantage)
		if result.AvgAdvantage > 0 {
			fmt.Printf("Result: MLP is BETTER than Balanced\n")
		} else {
			fmt.Printf("Result: MLP is WORSE than Balanced\n")
		}
		return
	}

	if *table {
		fmt.Printf("Building hand table: %d games, %d workers, seed=%d\n\n", *n, *workers, *seed)
		factory := func(rng *rand.Rand) [6]game.Strategy {
			var strats [6]game.Strategy
			for i := range strats {
				strats[i] = strategy.NewStandard(strategy.Balanced)
			}
			return strats
		}
		ht := sim.RunHandTable(*n, factory, *seed, *workers, *checkpoint, "hand_table.csv")
		ht.PrintSummary(*minRows)
		ht.WriteCSV("hand_table.csv")
		fmt.Printf("\nFull table written to hand_table.csv\n")
		return
	}

	fmt.Printf("Running sweep: %d games per matchup, seed=%d\n\n", *n, *seed)

	configs := []sim.NamedFactory{
		{
			Name: "Conservative",
			Factory: func(rng *rand.Rand) game.Strategy {
				return strategy.NewStandard(strategy.Conservative)
			},
		},
		{
			Name: "BalancedV1",
			Factory: func(rng *rand.Rand) game.Strategy {
				return strategy.NewStandard(strategy.BalancedV1)
			},
		},
		{
			Name: "Balanced",
			Factory: func(rng *rand.Rand) game.Strategy {
				return strategy.NewStandard(strategy.Balanced)
			},
		},
		{
			Name: "Aggressive",
			Factory: func(rng *rand.Rand) game.Strategy {
				return strategy.NewStandard(strategy.Aggressive)
			},
		},
	}

	sim.RunSweep(*n, configs, *seed)
}

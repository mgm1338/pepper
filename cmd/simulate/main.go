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
	modelPath        := flag.String("model", "", "path to model_weights.json; runs MLP vs Balanced paired eval")
	bidModelPath     := flag.String("bid-model", "", "path to bid_model_weights.json; combined with -model for full MLP eval")
	oppModelPath     := flag.String("opponent-model", "", "path to opponent model_weights.json; runs MLP vs MLP instead of vs Balanced")
	oppBidModelPath  := flag.String("opponent-bid-model", "", "path to opponent bid_model_weights.json")
	flag.Parse()

	if *modelPath != "" {
		model, err := ml.LoadMLP(*modelPath)
		if err != nil {
			log.Fatalf("load MLP: %v", err)
		}
		var bidModel *ml.BidMLP
		if *bidModelPath != "" {
			bidModel, err = ml.LoadBidMLP(*bidModelPath)
			if err != nil {
				log.Fatalf("load bid MLP: %v", err)
			}
		}
		mlpFactory := func(rng *rand.Rand) [6]game.Strategy {
			var strats [6]game.Strategy
			for i := range strats {
				s := mlstrategy.NewMLPStrategy(model, strategy.Balanced)
				if bidModel != nil {
					s = s.WithBidModel(bidModel)
				}
				strats[i] = s
			}
			return strats
		}

		var oppFactory func(*rand.Rand) [6]game.Strategy
		var oppLabel string
		if *oppModelPath != "" {
			oppModel, err := ml.LoadMLP(*oppModelPath)
			if err != nil {
				log.Fatalf("load opponent MLP: %v", err)
			}
			var oppBidModel *ml.BidMLP
			if *oppBidModelPath != "" {
				oppBidModel, err = ml.LoadBidMLP(*oppBidModelPath)
				if err != nil {
					log.Printf("opponent bid model incompatible, falling back to Balanced: %v", err)
					oppBidModel = nil
				}
			}
			oppFactory = func(rng *rand.Rand) [6]game.Strategy {
				var strats [6]game.Strategy
				for i := range strats {
					s := mlstrategy.NewMLPStrategy(oppModel, strategy.Balanced)
					if oppBidModel != nil {
						s = s.WithBidModel(oppBidModel)
					}
					strats[i] = s
				}
				return strats
			}
			oppLabel = "prev MLP"
		} else {
			oppFactory = func(rng *rand.Rand) [6]game.Strategy {
				var strats [6]game.Strategy
				for i := range strats {
					strats[i] = strategy.NewStandard(strategy.Balanced)
				}
				return strats
			}
			oppLabel = "Balanced"
		}

		var result sim.PairedResult
		if *oppModelPath != "" {
			fmt.Printf("Head-to-head: new MLP (seats 0,2,4) vs %s (seats 1,3,5) — %d hands, seed=%d\n\n", oppLabel, *n, *seed)
			result = sim.RunHeadToHead(*n, mlpFactory, oppFactory, *seed)
		} else {
			fmt.Printf("Paired eval: MLP vs %s — %d hands, seed=%d\n\n", oppLabel, *n, *seed)
			result = sim.RunPairedHands(*n, mlpFactory, oppFactory, *seed)
		}
		fmt.Printf("Hands played:       %d\n", result.Hands)
		fmt.Printf("MLP avg advantage:  %+.4f pts/hand vs %s\n", result.AvgAdvantage, oppLabel)
		if result.AvgAdvantage > 0 {
			fmt.Printf("Result: MLP is BETTER than %s\n", oppLabel)
		} else {
			fmt.Printf("Result: MLP is WORSE than %s\n", oppLabel)
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
			Name: "Balanced",
			Factory: func(rng *rand.Rand) game.Strategy {
				return strategy.NewStandard(strategy.Balanced)
			},
		},
	}

	sim.RunSweep(*n, configs, *seed)
}

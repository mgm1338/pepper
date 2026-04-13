package ml_test

import (
	"math/rand"
	"testing"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/mlstrategy"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

// BenchmarkCollectHandMLP measures collect with MLP self-play (realistic production load).
func BenchmarkCollectHandMLP(b *testing.B) {
	m, err := ml.LoadMLP("../model_weights.json")
	if err != nil {
		b.Skip("model_weights.json not found:", err)
	}
	bm, err := ml.LoadBidMLP("../bid_model_weights.json")
	if err != nil {
		b.Skip("bid_model_weights.json not found:", err)
	}
	rng := rand.New(rand.NewSource(42))
	gs := game.NewGame(0)

	makeStrats := func() [6]game.Strategy {
		var s [6]game.Strategy
		for i := 0; i < 6; i++ {
			play := mlstrategy.NewMLPStrategy(m.Clone(), strategy.Balanced)
			play = play.WithBidModel(bm)
			s[i] = play
		}
		return s
	}
	strats := makeStrats()
	rolloutStrats := makeStrats()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ml.CollectHand(i, gs, strats, rolloutStrats, rng, 20)
		gs.NextDealer()
	}
}

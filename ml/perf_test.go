package ml

import (
	"math/rand"
	"testing"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
)

// BenchmarkCollectHand measures the optimized end-to-end cost for card-play collection (Balanced strategy).
func BenchmarkCollectHand(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	gs := game.NewGame(0)
	var strats [6]game.Strategy
	var rollouts [6]game.Strategy
	for s := 0; s < 6; s++ {
		strats[s] = strategy.NewStandard(strategy.Balanced)
		rollouts[s] = strategy.NewStandard(strategy.Balanced)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CollectHand(i, gs, strats, rollouts, rng, 20)
		gs.NextDealer()
	}
}


// BenchmarkCollectBidHand measures the optimized end-to-end cost.
func BenchmarkCollectBidHand(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	gs := game.NewGame(0)
	var strats [6]game.Strategy
	var rollouts [6]game.Strategy
	for s := 0; s < 6; s++ {
		strats[s] = strategy.NewStandard(strategy.Balanced)
		rollouts[s] = strategy.NewStandard(strategy.Balanced)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CollectBidHand(i, gs, strats, rollouts, rng, 20)
		gs.NextDealer()
	}
}

// BenchmarkDropCardInPlace measures the in-place version (no alloc).
func BenchmarkDropCardInPlace(b *testing.B) {
	hand := []card.Card{
		{Suit: card.Spades, Rank: card.Ace},
		{Suit: card.Spades, Rank: card.King},
		{Suit: card.Hearts, Rank: card.Jack},
		{Suit: card.Clubs, Rank: card.Ten},
		{Suit: card.Diamonds, Rank: card.Nine},
	}
	target := hand[2]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h := hand[:5] // reset length each time using the same backing array
		_ = dropCardInPlace(h, target)
	}
}

// BenchmarkDropCardAlloc measures the old allocating version for comparison.
func BenchmarkDropCardAlloc(b *testing.B) {
	hand := []card.Card{
		{Suit: card.Spades, Rank: card.Ace},
		{Suit: card.Spades, Rank: card.King},
		{Suit: card.Hearts, Rank: card.Jack},
		{Suit: card.Clubs, Rank: card.Ten},
		{Suit: card.Diamonds, Rank: card.Nine},
	}
	target := hand[2]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := make([]card.Card, 0, len(hand))
		removed := false
		for _, c := range hand {
			if !removed && c.Equal(target) {
				removed = true
				continue
			}
			result = append(result, c)
		}
		_ = result
	}
}

// BenchmarkCopyHandsPool measures pooled hand copy vs allocating.
func BenchmarkCopyHandsPool(b *testing.B) {
	src := [6][]card.Card{}
	for i := range src {
		src[i] = make([]card.Card, 8)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf := handBufPool.Get().(*[6][]card.Card)
		var hands [6][]card.Card
		for j, h := range src {
			hands[j] = (*buf)[j][:len(h)]
			copy(hands[j], h)
		}
		_ = hands
		handBufPool.Put(buf)
	}
}

func BenchmarkCopyHandsAlloc(b *testing.B) {
	src := [6][]card.Card{}
	for i := range src {
		src[i] = make([]card.Card, 8)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = copyHands(src)
	}
}

// BenchmarkMLPScoreGo measures the pure-Go 8-wide unrolled path (no CGo).
func BenchmarkMLPScoreGo(b *testing.B) {
	m, err := LoadMLP("../model_weights.json")
	if err != nil {
		b.Skip("model_weights.json not found:", err)
	}
	var feat [TotalFeatureLen]float32
	for i := range feat {
		feat[i] = float32(i) * 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.scoreGoFallback(feat)
	}
}

// BenchmarkMLPScore measures a single MLP forward pass (flat weights).
func BenchmarkMLPScore(b *testing.B) {
	m, err := LoadMLP("../model_weights.json")
	if err != nil {
		b.Skip("model_weights.json not found:", err)
	}
	var feat [TotalFeatureLen]float32
	for i := range feat {
		feat[i] = float32(i) * 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Score(feat)
	}
}

// BenchmarkMLPScoreJagged is the old jagged-[][]float32 path for comparison.
func BenchmarkMLPScoreJagged(b *testing.B) {
	m, err := LoadMLP("../model_weights.json")
	if err != nil {
		b.Skip("model_weights.json not found:", err)
	}
	var feat [TotalFeatureLen]float32
	for i := range feat {
		feat[i] = float32(i) * 0.01
	}
	w := m.w
	h1 := make([]float32, len(w.B1))
	h2 := make([]float32, len(w.B2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for i2, bi := range w.B1 {
			v := bi
			for j := range feat {
				v += w.W1[i2][j] * feat[j]
			}
			if v < 0 { v = 0 }
			h1[i2] = v
		}
		for i2, bi := range w.B2 {
			v := bi
			for j, hj := range h1 {
				v += w.W2[i2][j] * hj
			}
			if v < 0 { v = 0 }
			h2[i2] = v
		}
		out := w.B3
		for j, hj := range h2 {
			out += w.W3[j] * hj
		}
		_ = out*w.YStd + w.YMean
	}
}

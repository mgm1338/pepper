package ml

import (
	"math"
	"math/rand"
	"testing"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
)

func balancedStrats() [6]game.Strategy {
	var s [6]game.Strategy
	for i := range s {
		s[i] = strategy.NewStandard(strategy.Balanced)
	}
	return s
}

// --- CollectBidHand ---

func TestCollectBidHand_rowCount(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	gs := game.NewGame(0)
	strats := balancedStrats()
	rows := CollectBidHand(0, gs, strats, strats, rng, BidCollectOpts{Rollouts: 5})

	// Each hand has up to 6 bid decisions. Stuck hands produce 0 rows (dealer forced).
	// We should get at least 1 row (some seat sees a valid bid decision).
	if len(rows) == 0 {
		t.Fatal("CollectBidHand returned 0 rows")
	}
	if len(rows) > 6*7 { // 6 seats × up to 7 bid levels
		t.Errorf("suspiciously many rows: %d", len(rows))
	}
}

func TestCollectBidHand_fieldRanges(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	gs := game.NewGame(0)
	strats := balancedStrats()

	for hand := 0; hand < 50; hand++ {
		rows := CollectBidHand(hand, gs, strats, strats, rng, BidCollectOpts{Rollouts: 3})
		gs.NextDealer()

		for i, r := range rows {
			if r.Seat < 0 || r.Seat > 5 {
				t.Errorf("hand %d row %d: Seat=%d out of [0,5]", hand, i, r.Seat)
			}
			if r.BidLevel < game.PassBid || r.BidLevel > game.PepperBid {
				t.Errorf("hand %d row %d: BidLevel=%d out of [%d,%d]",
					hand, i, r.BidLevel, game.PassBid, game.PepperBid)
			}
			if math.IsNaN(float64(r.ScoreDelta)) || math.IsInf(float64(r.ScoreDelta), 0) {
				t.Errorf("hand %d row %d: ScoreDelta=%f is NaN/Inf", hand, i, r.ScoreDelta)
			}
			for j, f := range r.Features {
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					t.Errorf("hand %d row %d feat %d: %f is NaN/Inf", hand, i, j, f)
				}
			}
		}
	}
}

func TestCollectBidHand_featureLen(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	gs := game.NewGame(0)
	strats := balancedStrats()
	rows := CollectBidHand(0, gs, strats, strats, rng, BidCollectOpts{Rollouts: 3})
	for i, r := range rows {
		if len(r.Features) != BidTotalLen {
			t.Errorf("row %d: Features len=%d, want %d", i, len(r.Features), BidTotalLen)
		}
	}
}

func TestCollectBidHand_deterministic(t *testing.T) {
	run := func() []BidCollectRow {
		rng := rand.New(rand.NewSource(123))
		gs := game.NewGame(2)
		strats := balancedStrats()
		return CollectBidHand(0, gs, strats, strats, rng, BidCollectOpts{Rollouts: 5})
	}

	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("determinism: len(a)=%d len(b)=%d", len(a), len(b))
	}
	for i := range a {
		if a[i].ScoreDelta != b[i].ScoreDelta {
			t.Errorf("row %d: ScoreDelta %f vs %f", i, a[i].ScoreDelta, b[i].ScoreDelta)
		}
		if a[i].Seat != b[i].Seat {
			t.Errorf("row %d: Seat %d vs %d", i, a[i].Seat, b[i].Seat)
		}
		if a[i].BidLevel != b[i].BidLevel {
			t.Errorf("row %d: BidLevel %d vs %d", i, a[i].BidLevel, b[i].BidLevel)
		}
	}
}

// --- CollectHand ---

func TestCollectHand_rowCount(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	gs := game.NewGame(0)
	strats := balancedStrats()
	rows := CollectHand(0, gs, strats, strats, rng, CollectOpts{Rollouts: 5})

	// Each hand produces rows per (trick × seat) decision. At least 1 row expected.
	if len(rows) == 0 {
		t.Fatal("CollectHand returned 0 rows")
	}
}

func TestCollectHand_fieldRanges(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	gs := game.NewGame(0)
	strats := balancedStrats()

	for hand := 0; hand < 30; hand++ {
		rows := CollectHand(hand, gs, strats, strats, rng, CollectOpts{Rollouts: 3})
		gs.NextDealer()

		for i, r := range rows {
			if r.Seat < 0 || r.Seat > 5 {
				t.Errorf("hand %d row %d: Seat=%d out of [0,5]", hand, i, r.Seat)
			}
			if r.TrickNumber < 0 || r.TrickNumber >= game.TotalTricks {
				t.Errorf("hand %d row %d: TrickNumber=%d out of [0,%d)",
					hand, i, r.TrickNumber, game.TotalTricks)
			}
			if math.IsNaN(float64(r.ScoreDelta)) || math.IsInf(float64(r.ScoreDelta), 0) {
				t.Errorf("hand %d row %d: ScoreDelta NaN/Inf", hand, i)
			}
			for j, f := range r.Features {
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					t.Errorf("hand %d row %d feat %d: NaN/Inf", hand, i, j)
				}
			}
		}
	}
}

func TestCollectHand_deterministic(t *testing.T) {
	run := func() []CollectRow {
		rng := rand.New(rand.NewSource(456))
		gs := game.NewGame(1)
		strats := balancedStrats()
		return CollectHand(0, gs, strats, strats, rng, CollectOpts{Rollouts: 3})
	}

	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("determinism: len(a)=%d len(b)=%d", len(a), len(b))
	}
	for i := range a {
		if a[i].ScoreDelta != b[i].ScoreDelta {
			t.Errorf("row %d: ScoreDelta %f vs %f", i, a[i].ScoreDelta, b[i].ScoreDelta)
		}
	}
}

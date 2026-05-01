package game_test

import (
	"math/rand"
	"testing"

	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
)

func balancedStrategies() [6]game.Strategy {
	var s [6]game.Strategy
	for i := range s {
		s[i] = strategy.NewStandard(strategy.Balanced)
	}
	return s
}

// TestPlayHand_completes verifies a seeded hand runs to completion without panic.
func TestPlayHand_completes(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	gs := game.NewGame(0)
	strats := balancedStrategies()
	result := game.PlayHand(gs, strats, rng, game.NoopLogger{})

	// TricksTaken is [2]int by team; sum must equal TotalTricks.
	total := result.TricksTaken[0] + result.TricksTaken[1]
	if total != game.TotalTricks {
		t.Errorf("TricksTaken sum = %d, want %d", total, game.TotalTricks)
	}
}

// TestPlayHand_deterministic verifies same seed produces same result.
func TestPlayHand_deterministic(t *testing.T) {
	runOne := func() game.HandResult {
		rng := rand.New(rand.NewSource(99))
		gs := game.NewGame(3)
		return game.PlayHand(gs, balancedStrategies(), rng, game.NoopLogger{})
	}

	a, b := runOne(), runOne()
	if a.BidderSeat != b.BidderSeat {
		t.Errorf("BidderSeat: %d vs %d", a.BidderSeat, b.BidderSeat)
	}
	if a.BidAmount != b.BidAmount {
		t.Errorf("BidAmount: %d vs %d", a.BidAmount, b.BidAmount)
	}
	if a.Trump != b.Trump {
		t.Errorf("Trump: %v vs %v", a.Trump, b.Trump)
	}
	if a.MadeBid != b.MadeBid {
		t.Errorf("MadeBid: %v vs %v", a.MadeBid, b.MadeBid)
	}
	for team := 0; team < 2; team++ {
		if a.ScoreDelta[team] != b.ScoreDelta[team] {
			t.Errorf("ScoreDelta[%d]: %d vs %d", team, a.ScoreDelta[team], b.ScoreDelta[team])
		}
	}
}

// TestPlayHand_scoreConsistency verifies ScoreDelta is consistent with MadeBid.
func TestPlayHand_scoreConsistency(t *testing.T) {
	rng := rand.New(rand.NewSource(777))
	gs := game.NewGame(0)
	strats := balancedStrategies()

	for i := 0; i < 20; i++ {
		result := game.PlayHand(gs, strats, rng, game.NoopLogger{})
		gs.NextDealer()

		// Stuck bids at 3 are valid but count as normal bids for scoring.
		if result.BidAmount < game.StuckBid || result.BidAmount > game.PepperBid {
			t.Errorf("hand %d: BidAmount %d out of range [%d,%d]",
				i, result.BidAmount, game.StuckBid, game.PepperBid)
		}

		bidTeam := game.TeamOf(result.BidderSeat)
		oppTeam := 1 - bidTeam
		if result.MadeBid {
			if result.ScoreDelta[bidTeam] <= 0 {
				t.Errorf("hand %d: MadeBid=true but bidTeam ScoreDelta=%d", i, result.ScoreDelta[bidTeam])
			}
		} else {
			if result.ScoreDelta[bidTeam] >= 0 && !result.IsStuck {
				t.Errorf("hand %d: MadeBid=false but bidTeam ScoreDelta=%d", i, result.ScoreDelta[bidTeam])
			}
		}
		// Opponent always scores their tricks in a normal hand.
		if result.MadeBid && !result.IsPepper {
			if result.ScoreDelta[oppTeam] < 0 {
				t.Errorf("hand %d: made non-pepper bid but oppTeam ScoreDelta=%d (should be >=0)", i, result.ScoreDelta[oppTeam])
			}
		}
	}
}

// TestPlayHand_tricksTakenBounds verifies each team takes 0..TotalTricks tricks.
func TestPlayHand_tricksTakenBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(555))
	gs := game.NewGame(2)
	strats := balancedStrategies()

	for i := 0; i < 30; i++ {
		result := game.PlayHand(gs, strats, rng, game.NoopLogger{})
		gs.NextDealer()

		for team, n := range result.TricksTaken {
			if n < 0 || n > game.TotalTricks {
				t.Errorf("hand %d team %d: TricksTaken=%d out of [0,%d]",
					i, team, n, game.TotalTricks)
			}
		}
	}
}

package game

import (
	"testing"

	"github.com/max/pepper/internal/card"
)

func TestPepperBidConstant(t *testing.T) {
	// PepperBid must match pepper2's value (8) so the botserver and PHP agree.
	if PepperBid != 8 {
		t.Errorf("PepperBid = %d, want 8 (must match pepper2)", PepperBid)
	}
}

func TestPepperBidAboveMaxRegular(t *testing.T) {
	// PepperBid should be strictly above the highest regular bid (7).
	maxRegular := 7
	if PepperBid <= maxRegular {
		t.Errorf("PepperBid (%d) must be > max regular bid (%d)", PepperBid, maxRegular)
	}
}

func TestRunBidding_PepperShortCircuits(t *testing.T) {
	hand := []card.Card{{Rank: card.Jack, Suit: card.Hearts}}
	hands := [6][]card.Card{hand, hand, hand, hand, hand, hand}

	// Seat 0 immediately calls pepper.
	result := RunBidding(hands, 5, [2]int{}, func(seat int, state BidState) int {
		if seat == 0 {
			return PepperBid
		}
		return PassBid
	})

	if !result.IsPepper {
		t.Error("expected IsPepper=true")
	}
	if result.Winner != 0 {
		t.Errorf("Winner = %d, want 0", result.Winner)
	}
	if result.Amount != PepperBid {
		t.Errorf("Amount = %d, want %d", result.Amount, PepperBid)
	}
}

func TestRunBidding_DealerStuck(t *testing.T) {
	hand := []card.Card{{Rank: card.Nine, Suit: card.Clubs}}
	hands := [6][]card.Card{hand, hand, hand, hand, hand, hand}

	// Everyone passes — dealer (seat 2) gets stuck.
	result := RunBidding(hands, 2, [2]int{}, func(_ int, _ BidState) int {
		return PassBid
	})

	if !result.IsStuck {
		t.Error("expected IsStuck=true")
	}
	if result.Winner != 2 {
		t.Errorf("Winner = %d, want 2 (dealer)", result.Winner)
	}
	if result.Amount != StuckBid {
		t.Errorf("Amount = %d, want %d (StuckBid)", result.Amount, StuckBid)
	}
}

func TestRunBidding_HighestBidWins(t *testing.T) {
	hand := []card.Card{{Rank: card.Ace, Suit: card.Spades}}
	hands := [6][]card.Card{hand, hand, hand, hand, hand, hand}

	bids := map[int]int{1: 4, 2: 5, 3: PassBid, 4: 6, 5: PassBid}
	result := RunBidding(hands, 0, [2]int{}, func(seat int, state BidState) int {
		return bids[seat]
	})

	if result.Winner != 4 {
		t.Errorf("Winner = %d, want 4 (bid 6)", result.Winner)
	}
	if result.Amount != 6 {
		t.Errorf("Amount = %d, want 6", result.Amount)
	}
}

func TestRunBidding_BidStatePropagatesCurrentHigh(t *testing.T) {
	hand := []card.Card{{Rank: card.King, Suit: card.Diamonds}}
	hands := [6][]card.Card{hand, hand, hand, hand, hand, hand}

	var highsSeen []int
	RunBidding(hands, 0, [2]int{}, func(seat int, state BidState) int {
		highsSeen = append(highsSeen, state.CurrentHigh)
		if seat == 1 {
			return 5
		}
		return PassBid
	})

	// After seat 1 bids 5, subsequent seats should see CurrentHigh=5.
	for i, high := range highsSeen {
		seat := (0 + 1 + i) % 6
		if seat > 1 && high != 5 {
			t.Errorf("seat %d: CurrentHigh = %d, want 5", seat, high)
		}
	}
}

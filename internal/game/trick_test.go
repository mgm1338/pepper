package game

import (
	"testing"

	"github.com/max/pepper/internal/card"
)

// helpers — seat constants for readability
const (
	seat0 = 0
	seat1 = 1
	seat2 = 2
	seat3 = 3
)

var (
	sp = card.Spades
	cl = card.Clubs // partner of Spades

	rightBower  = card.Card{Suit: card.Spades, Rank: card.Jack, CopyIndex: 0}
	rightBower2 = card.Card{Suit: card.Spades, Rank: card.Jack, CopyIndex: 1}
	leftBower   = card.Card{Suit: card.Clubs, Rank: card.Jack, CopyIndex: 0}
	trumpAce    = card.Card{Suit: card.Spades, Rank: card.Ace, CopyIndex: 0}
	trumpKing   = card.Card{Suit: card.Spades, Rank: card.King, CopyIndex: 0}
	trumpNine   = card.Card{Suit: card.Spades, Rank: card.Nine, CopyIndex: 0}
	heartAce    = card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	heartKing   = card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}
	heartNine   = card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}
	diamondAce  = card.Card{Suit: card.Diamonds, Rank: card.Ace, CopyIndex: 0}
)

func makeTrickCards(trump card.Suit, plays ...struct {
	c    card.Card
	seat int
}) *Trick {
	t := NewTrick(plays[0].seat, trump)
	for _, p := range plays {
		t.Add(p.c, p.seat)
	}
	return t
}

// shorthand builder
type play struct {
	c    card.Card
	seat int
}

func buildTrick(leader int, trump card.Suit, plays ...play) *Trick {
	t := NewTrick(leader, trump)
	for _, p := range plays {
		t.Add(p.c, p.seat)
	}
	return t
}

// --- Winner ---

func TestTrickWinner_singleCard(t *testing.T) {
	tr := buildTrick(seat0, sp, play{heartAce, seat0})
	if got := tr.Winner(); got != seat0 {
		t.Errorf("Winner = %d, want %d", got, seat0)
	}
}

func TestTrickWinner_highLedSuitWins(t *testing.T) {
	// seat0 leads heartNine, seat1 plays heartAce — ace wins
	tr := buildTrick(seat0, sp, play{heartNine, seat0}, play{heartAce, seat1})
	if got := tr.Winner(); got != seat1 {
		t.Errorf("Winner = %d, want seat1 (higher led suit)", got)
	}
}

func TestTrickWinner_offSuitCannotWin(t *testing.T) {
	// seat0 leads heartNine, seat1 plays diamondAce (off-suit, no trump) — seat0 keeps winning
	tr := buildTrick(seat0, sp, play{heartNine, seat0}, play{diamondAce, seat1})
	if got := tr.Winner(); got != seat0 {
		t.Errorf("Winner = %d, want seat0 (off-suit cannot win)", got)
	}
}

func TestTrickWinner_trumpBeatsLedSuit(t *testing.T) {
	// seat0 leads heartAce, seat1 plays trump nine — trump wins
	tr := buildTrick(seat0, sp, play{heartAce, seat0}, play{trumpNine, seat1})
	if got := tr.Winner(); got != seat1 {
		t.Errorf("Winner = %d, want seat1 (trump beats non-trump)", got)
	}
}

func TestTrickWinner_higherTrumpWins(t *testing.T) {
	// seat0 leads trumpNine, seat1 plays trumpAce, seat2 plays trumpKing — ace wins
	tr := buildTrick(seat0, sp,
		play{trumpNine, seat0},
		play{trumpAce, seat1},
		play{trumpKing, seat2},
	)
	if got := tr.Winner(); got != seat1 {
		t.Errorf("Winner = %d, want seat1 (trump ace)", got)
	}
}

func TestTrickWinner_rightBowerBeatsAll(t *testing.T) {
	tr := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{leftBower, seat1},
		play{rightBower, seat2},
	)
	if got := tr.Winner(); got != seat2 {
		t.Errorf("Winner = %d, want seat2 (right bower)", got)
	}
}

func TestTrickWinner_leftBowerBeatsTrumpAce(t *testing.T) {
	tr := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{leftBower, seat1},
	)
	if got := tr.Winner(); got != seat1 {
		t.Errorf("Winner = %d, want seat1 (left bower)", got)
	}
}

func TestTrickWinner_identicalTrump_firstPlayed(t *testing.T) {
	// Both copies of right bower: earlier play order wins (seat0)
	tr := buildTrick(seat0, sp,
		play{rightBower, seat0},
		play{rightBower2, seat1},
	)
	if got := tr.Winner(); got != seat0 {
		t.Errorf("Winner = %d, want seat0 (first played wins tie)", got)
	}
}

func TestTrickWinner_leftBowerCountsAsTrump_notClubs(t *testing.T) {
	// Left bower led (clubs suit but counts as spades trump).
	// heartAce follows — off-suit — left bower keeps winning.
	tr := buildTrick(seat0, sp,
		play{leftBower, seat0},
		play{heartAce, seat1},
	)
	if got := tr.Winner(); got != seat0 {
		t.Errorf("Winner = %d, want seat0 (left bower is trump)", got)
	}
}

func TestTrickWinner_multipleCards_correctFinal(t *testing.T) {
	// Full 4-seat trick: seat0 leads heartKing, seat1 plays trumpNine,
	// seat2 plays trumpAce, seat3 plays heartAce (off-suit after trump played).
	tr := buildTrick(seat0, sp,
		play{heartKing, seat0},
		play{trumpNine, seat1},
		play{trumpAce, seat2},
		play{heartAce, seat3},
	)
	if got := tr.Winner(); got != seat2 {
		t.Errorf("Winner = %d, want seat2 (trump ace)", got)
	}
}

// --- WinnerCard ---

func TestTrickWinnerCard_matchesWinnerSeat(t *testing.T) {
	tr := buildTrick(seat0, sp,
		play{heartAce, seat0},
		play{trumpNine, seat1},
		play{trumpKing, seat2},
	)
	wantCard := trumpKing
	wantSeat := seat2
	if got := tr.Winner(); got != wantSeat {
		t.Errorf("Winner seat = %d, want %d", got, wantSeat)
	}
	if got := tr.WinnerCard(); !got.Equal(wantCard) {
		t.Errorf("WinnerCard = %v, want %v", got, wantCard)
	}
}

func TestTrickWinnerCard_singleCard(t *testing.T) {
	tr := buildTrick(seat0, sp, play{heartAce, seat0})
	if got := tr.WinnerCard(); !got.Equal(heartAce) {
		t.Errorf("WinnerCard = %v, want heartAce", got)
	}
}

// --- Reset clears winner state ---

func TestTrickReset_clearsWinner(t *testing.T) {
	tr := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{rightBower, seat1},
	)
	// Before reset: right bower wins
	if got := tr.Winner(); got != seat1 {
		t.Fatalf("pre-reset Winner = %d, want seat1", got)
	}

	tr.Reset(seat2, card.Hearts)
	tr.Add(heartKing, seat2)
	tr.Add(heartNine, seat3)

	if got := tr.Winner(); got != seat2 {
		t.Errorf("post-reset Winner = %d, want seat2 (heartKing)", got)
	}
	if got := tr.WinnerCard(); !got.Equal(heartKing) {
		t.Errorf("post-reset WinnerCard = %v, want heartKing", got)
	}
}

// --- LedSuit ---

func TestTrickLedSuit_nonTrump(t *testing.T) {
	tr := buildTrick(seat0, sp, play{heartAce, seat0})
	if got := tr.LedSuit(); got != card.Hearts {
		t.Errorf("LedSuit = %v, want Hearts", got)
	}
}

func TestTrickLedSuit_leftBowerCountsAsTrump(t *testing.T) {
	tr := buildTrick(seat0, sp, play{leftBower, seat0})
	if got := tr.LedSuit(); got != sp {
		t.Errorf("LedSuit = %v, want Spades (left bower = trump)", got)
	}
}

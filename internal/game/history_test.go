package game

import (
	"testing"

	"github.com/max/pepper/internal/card"
)

// helpers
var (
	trump = card.Spades

	rightBower0 = card.Card{Suit: card.Spades, Rank: card.Jack, CopyIndex: 0}
	rightBower1 = card.Card{Suit: card.Spades, Rank: card.Jack, CopyIndex: 1}
	leftBower0  = card.Card{Suit: card.Clubs, Rank: card.Jack, CopyIndex: 0}  // partner suit for Spades
	leftBower1  = card.Card{Suit: card.Clubs, Rank: card.Jack, CopyIndex: 1}
	trumpAce0   = card.Card{Suit: card.Spades, Rank: card.Ace, CopyIndex: 0}
	trumpKing0  = card.Card{Suit: card.Spades, Rank: card.King, CopyIndex: 0}
	trumpNine0  = card.Card{Suit: card.Spades, Rank: card.Nine, CopyIndex: 0}
	heartAce0   = card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	heartKing0  = card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}
)

func TestHandHistory_RecordAndIsSeen(t *testing.T) {
	var h HandHistory
	if h.IsSeen(rightBower0) {
		t.Fatal("empty history: IsSeen should return false")
	}

	h.Record([]card.Card{rightBower0, heartAce0})
	if !h.IsSeen(rightBower0) {
		t.Fatal("rightBower0 should be seen after Record")
	}
	if !h.IsSeen(heartAce0) {
		t.Fatal("heartAce0 should be seen after Record")
	}
	// copy index distinguishes cards
	if h.IsSeen(rightBower1) {
		t.Fatal("rightBower1 not recorded, should not be seen")
	}
}

func TestHandHistory_PlayedSlice(t *testing.T) {
	var h HandHistory
	h.Record([]card.Card{trumpAce0, heartKing0})
	s := h.PlayedSlice()
	if len(s) != 2 {
		t.Fatalf("PlayedSlice len = %d, want 2", len(s))
	}
	if !s[0].Equal(trumpAce0) || !s[1].Equal(heartKing0) {
		t.Fatal("PlayedSlice order mismatch")
	}
}

func TestHandHistory_Played(t *testing.T) {
	var h HandHistory
	h.Record([]card.Card{trumpAce0})
	p := h.Played()
	if len(p) != 1 || !p[0].Equal(trumpAce0) {
		t.Fatal("Played returned wrong cards")
	}
	// Played must be a copy — mutating it must not affect history.
	p[0] = heartAce0
	if h.IsSeen(heartAce0) {
		t.Fatal("Played must return a copy, not the underlying slice")
	}
}

func TestHandHistory_TrumpPlayed(t *testing.T) {
	var h HandHistory
	if h.TrumpPlayed(trump) != 0 {
		t.Fatal("empty history: TrumpPlayed should be 0")
	}
	h.Record([]card.Card{rightBower0, heartAce0, trumpKing0})
	if got := h.TrumpPlayed(trump); got != 2 {
		t.Fatalf("TrumpPlayed = %d, want 2", got)
	}
}

func TestHandHistory_TrumpRemaining(t *testing.T) {
	var h HandHistory
	// 14 total trump cards at start
	if got := h.TrumpRemaining(trump); got != card.TotalTrumpCards {
		t.Fatalf("TrumpRemaining empty = %d, want %d", got, card.TotalTrumpCards)
	}
	h.Record([]card.Card{rightBower0, leftBower0, trumpAce0})
	if got := h.TrumpRemaining(trump); got != card.TotalTrumpCards-3 {
		t.Fatalf("TrumpRemaining after 3 trump = %d, want %d", got, card.TotalTrumpCards-3)
	}
}

func TestHandHistory_BowersPlayed(t *testing.T) {
	var h HandHistory
	if h.RightBowersPlayed(trump) != 0 || h.LeftBowersPlayed(trump) != 0 {
		t.Fatal("empty history: bowers should be 0")
	}
	h.Record([]card.Card{rightBower0, rightBower1, leftBower0})
	if got := h.RightBowersPlayed(trump); got != 2 {
		t.Fatalf("RightBowersPlayed = %d, want 2", got)
	}
	if got := h.LeftBowersPlayed(trump); got != 1 {
		t.Fatalf("LeftBowersPlayed = %d, want 1", got)
	}
}

func TestHandHistory_IsTopTrump(t *testing.T) {
	var h HandHistory
	// Right bower is top trump when nothing played.
	if !h.IsTopTrump(rightBower0, trump) {
		t.Fatal("rightBower0 should be top trump on empty history")
	}
	// Left bower is NOT top trump when both right bowers unplayed.
	if h.IsTopTrump(leftBower0, trump) {
		t.Fatal("leftBower0 should not be top trump when right bowers unplayed")
	}
	// After both right bowers played, left bower becomes top trump.
	h.Record([]card.Card{rightBower0, rightBower1})
	if !h.IsTopTrump(leftBower0, trump) {
		t.Fatal("leftBower0 should be top trump after both right bowers played")
	}
	// Non-trump card is never top trump.
	if h.IsTopTrump(heartAce0, trump) {
		t.Fatal("non-trump card should never be top trump")
	}
}

func TestHandHistory_CardsPlayedInSuit(t *testing.T) {
	var h HandHistory
	h.Record([]card.Card{trumpAce0, trumpKing0, heartAce0, heartKing0})
	if got := h.CardsPlayedInSuit(card.Spades, trump); got != 2 {
		t.Fatalf("CardsPlayedInSuit(Spades) = %d, want 2", got)
	}
	if got := h.CardsPlayedInSuit(card.Hearts, trump); got != 2 {
		t.Fatalf("CardsPlayedInSuit(Hearts) = %d, want 2", got)
	}
	// Left bower counts as trump suit, not clubs.
	h.Record([]card.Card{leftBower0})
	if got := h.CardsPlayedInSuit(card.Spades, trump); got != 3 {
		t.Fatalf("CardsPlayedInSuit(Spades) after left bower = %d, want 3", got)
	}
	if got := h.CardsPlayedInSuit(card.Clubs, trump); got != 0 {
		t.Fatalf("left bower should not count as clubs, got %d", got)
	}
}

func TestHandHistory_Reset(t *testing.T) {
	var h HandHistory
	h.Record([]card.Card{trumpAce0, heartAce0, trumpKing0})
	h.Reset()

	if h.n != 0 {
		t.Fatalf("Reset: n = %d, want 0", h.n)
	}
	if len(h.PlayedSlice()) != 0 {
		t.Fatal("Reset: PlayedSlice should be empty")
	}
	if h.IsSeen(trumpAce0) {
		t.Fatal("Reset: IsSeen should return false after reset")
	}
	if h.TrumpPlayed(trump) != 0 {
		t.Fatalf("Reset: TrumpPlayed = %d, want 0", h.TrumpPlayed(trump))
	}

	// Verify history can be used normally after reset.
	h.Record([]card.Card{heartKing0})
	if !h.IsSeen(heartKing0) {
		t.Fatal("Reset: new cards should be recorded after reset")
	}
	if h.IsSeen(trumpAce0) {
		t.Fatal("Reset: pre-reset cards must not reappear")
	}
}

func TestHandHistory_MultipleRecords(t *testing.T) {
	var h HandHistory
	h.Record([]card.Card{trumpAce0, heartAce0})
	h.Record([]card.Card{trumpKing0, heartKing0})
	if got := h.TrumpPlayed(trump); got != 2 {
		t.Fatalf("TrumpPlayed after two records = %d, want 2", got)
	}
	s := h.PlayedSlice()
	if len(s) != 4 {
		t.Fatalf("PlayedSlice len = %d, want 4", len(s))
	}
}

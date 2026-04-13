package game

import (
	"testing"

	"github.com/max/pepper/internal/card"
)

func TestRemoveCard_removesFirstMatch(t *testing.T) {
	c0 := card.Card{Suit: card.Spades, Rank: card.Ace, CopyIndex: 0}
	c1 := card.Card{Suit: card.Spades, Rank: card.Ace, CopyIndex: 1}
	c2 := card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}

	hand := []card.Card{c0, c1, c2}
	result := removeCard(hand, c0)

	if len(result) != 2 {
		t.Fatalf("len = %d, want 2", len(result))
	}
	if !result[0].Equal(c1) || !result[1].Equal(c2) {
		t.Fatal("wrong cards remaining after remove")
	}
}

func TestRemoveCard_onlyRemovesOne(t *testing.T) {
	c0 := card.Card{Suit: card.Spades, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{c0, c0} // two identical cards
	result := removeCard(hand, c0)
	if len(result) != 1 {
		t.Fatalf("should remove only first match, len = %d", len(result))
	}
}

func TestRemoveCard_targetNotPresent(t *testing.T) {
	c0 := card.Card{Suit: card.Spades, Rank: card.Ace, CopyIndex: 0}
	c1 := card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}
	hand := []card.Card{c0}
	result := removeCard(hand, c1)
	if len(result) != 1 || !result[0].Equal(c0) {
		t.Fatal("hand should be unchanged when target not present")
	}
}

func TestRemoveCard_singleCard(t *testing.T) {
	c0 := card.Card{Suit: card.Spades, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{c0}
	result := removeCard(hand, c0)
	if len(result) != 0 {
		t.Fatalf("removing only card should give empty slice, len = %d", len(result))
	}
}

package card

import (
	"math/rand"
	"testing"
)

func TestNewPinochleDeck_length(t *testing.T) {
	d := NewPinochleDeck()
	if len(d) != 48 {
		t.Fatalf("deck len = %d, want 48", len(d))
	}
}

func TestNewPinochleDeck_composition(t *testing.T) {
	d := NewPinochleDeck()
	counts := map[Card]int{}
	for _, c := range d {
		counts[c]++
	}
	for _, s := range []Suit{Spades, Clubs, Hearts, Diamonds} {
		for _, r := range []Rank{Nine, Ten, Jack, Queen, King, Ace} {
			for _, cp := range []int{0, 1} {
				k := Card{Suit: s, Rank: r, CopyIndex: cp}
				if counts[k] != 1 {
					t.Errorf("%v appears %d times, want 1", k, counts[k])
				}
			}
		}
	}
}

func TestShuffle_deterministicWithSeed(t *testing.T) {
	d1 := NewPinochleDeck()
	d2 := NewPinochleDeck()
	Shuffle(d1, rand.New(rand.NewSource(42)))
	Shuffle(d2, rand.New(rand.NewSource(42)))
	if len(d1) != 48 {
		t.Fatalf("len after shuffle = %d", len(d1))
	}
	for i := range d1 {
		if !d1[i].Equal(d2[i]) {
			t.Fatalf("seeded shuffle diverged at %d", i)
		}
	}
}

func TestShuffle_preservesCards(t *testing.T) {
	orig := NewPinochleDeck()
	shuf := NewPinochleDeck()
	Shuffle(shuf, rand.New(rand.NewSource(7)))
	origCount := map[Card]int{}
	shufCount := map[Card]int{}
	for i := range orig {
		origCount[orig[i]]++
		shufCount[shuf[i]]++
	}
	for k, v := range origCount {
		if shufCount[k] != v {
			t.Errorf("count mismatch for %v: %d vs %d", k, v, shufCount[k])
		}
	}
}

func TestDeal_shape(t *testing.T) {
	hands := Deal(rand.New(rand.NewSource(1)))
	total := 0
	for _, h := range hands {
		if len(h) != 8 {
			t.Errorf("hand len = %d, want 8", len(h))
		}
		total += len(h)
	}
	if total != 48 {
		t.Errorf("total cards dealt = %d, want 48", total)
	}
	seen := map[Card]int{}
	for _, h := range hands {
		for _, c := range h {
			seen[c]++
			if seen[c] > 1 {
				t.Errorf("card %v dealt twice", c)
			}
		}
	}
}

func TestDealAround_fixedSeatKept(t *testing.T) {
	fixed := []Card{
		{Suit: Spades, Rank: Ace, CopyIndex: 0},
		{Suit: Spades, Rank: King, CopyIndex: 0},
		{Suit: Hearts, Rank: Ace, CopyIndex: 0},
		{Suit: Hearts, Rank: King, CopyIndex: 0},
		{Suit: Diamonds, Rank: Ace, CopyIndex: 0},
		{Suit: Diamonds, Rank: King, CopyIndex: 0},
		{Suit: Clubs, Rank: Ace, CopyIndex: 0},
		{Suit: Clubs, Rank: King, CopyIndex: 0},
	}
	hands := DealAround(2, fixed, rand.New(rand.NewSource(3)))

	if len(hands[2]) != 8 {
		t.Fatalf("fixed hand len = %d", len(hands[2]))
	}
	for i, c := range fixed {
		if !hands[2][i].Equal(c) {
			t.Errorf("fixed card %d = %v, want %v", i, hands[2][i], c)
		}
	}
	// All other hands get 8 cards, none overlap fixed.
	fixedSet := map[Card]bool{}
	for _, c := range fixed {
		fixedSet[c] = true
	}
	for i, h := range hands {
		if i == 2 {
			continue
		}
		if len(h) != 8 {
			t.Errorf("hand %d len = %d", i, len(h))
		}
		for _, c := range h {
			if fixedSet[c] {
				t.Errorf("hand %d contains fixed card %v", i, c)
			}
		}
	}
}

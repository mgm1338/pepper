package game

import (
	"testing"

	"github.com/max/pepper/internal/card"
)

// --- Cards field: count and indexing ---

func TestTrickCards_countAfterAdds(t *testing.T) {
	trick := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{trumpKing, seat1},
		play{heartAce, seat2},
	)
	if trick.NCards != 3 {
		t.Fatalf("len(Cards) = %d, want 3", trick.NCards)
	}
}

func TestTrickCards_emptyAfterNew(t *testing.T) {
	trick := NewTrick(seat0, sp)
	if trick.NCards != 0 {
		t.Fatalf("new trick has %d cards, want 0", trick.NCards)
	}
}

func TestTrickCards_indexingPreservesCard(t *testing.T) {
	trick := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{heartKing, seat1},
	)
	if !trick.Cards[0].Card.Equal(trumpAce) {
		t.Errorf("Cards[0].Card = %v, want trumpAce", trick.Cards[0].Card)
	}
	if !trick.Cards[1].Card.Equal(heartKing) {
		t.Errorf("Cards[1].Card = %v, want heartKing", trick.Cards[1].Card)
	}
}

func TestTrickCards_indexingPreservesSeat(t *testing.T) {
	trick := buildTrick(seat1, sp,
		play{heartAce, seat1},
		play{heartKing, seat2},
		play{heartNine, seat3},
	)
	if trick.Cards[0].Seat != seat1 {
		t.Errorf("Cards[0].Seat = %d, want %d", trick.Cards[0].Seat, seat1)
	}
	if trick.Cards[1].Seat != seat2 {
		t.Errorf("Cards[1].Seat = %d, want %d", trick.Cards[1].Seat, seat2)
	}
	if trick.Cards[2].Seat != seat3 {
		t.Errorf("Cards[2].Seat = %d, want %d", trick.Cards[2].Seat, seat3)
	}
}

// --- PlayOrder field ---

func TestTrickCards_playOrderMonotone(t *testing.T) {
	plays := []play{
		{trumpAce, seat0},
		{heartAce, seat1},
		{diamondAce, seat2},
		{heartNine, seat3},
	}
	trick := buildTrick(seat0, sp, plays...)
	for i, pc := range trick.Cards[:trick.NCards] {
		if pc.PlayOrder != i {
			t.Errorf("Cards[%d].PlayOrder = %d, want %d", i, pc.PlayOrder, i)
		}
	}
}

func TestTrickCards_playOrderStartsAtZero(t *testing.T) {
	trick := NewTrick(seat2, sp)
	trick.Add(heartAce, seat2)
	if trick.Cards[0].PlayOrder != 0 {
		t.Errorf("first card PlayOrder = %d, want 0", trick.Cards[0].PlayOrder)
	}
}

// --- Range iteration ---

func TestTrickCards_rangeIteratesAll(t *testing.T) {
	cards := []card.Card{trumpAce, trumpKing, heartAce, heartKing, heartNine, diamondAce}
	seats := []int{seat0, seat1, seat2, seat3, 4, 5}
	trick := NewTrick(seat0, sp)
	for i, c := range cards {
		trick.Add(c, seats[i])
	}
	seen := 0
	for _, pc := range trick.Cards[:trick.NCards] {
		if !pc.Card.Equal(cards[seen]) {
			t.Errorf("Cards[%d] = %v, want %v", seen, pc.Card, cards[seen])
		}
		seen++
	}
	if seen != 6 {
		t.Errorf("range saw %d cards, want 6", seen)
	}
}

// --- Reset clears Cards ---

func TestTrickCards_resetClearsCount(t *testing.T) {
	trick := buildTrick(seat0, sp,
		play{trumpAce, seat0},
		play{heartAce, seat1},
	)
	trick.Reset(seat2, card.Hearts)
	if trick.NCards != 0 {
		t.Errorf("after Reset, len(Cards) = %d, want 0", trick.NCards)
	}
}

func TestTrickCards_resetThenAdd(t *testing.T) {
	trick := buildTrick(seat0, sp, play{trumpAce, seat0})
	trick.Reset(seat1, card.Hearts)
	trick.Add(heartAce, seat1)
	if trick.NCards != 1 {
		t.Fatalf("after Reset+Add, len(Cards) = %d, want 1", trick.NCards)
	}
	if trick.Cards[0].PlayOrder != 0 {
		t.Errorf("first card after reset has PlayOrder = %d, want 0", trick.Cards[0].PlayOrder)
	}
	if !trick.Cards[0].Card.Equal(heartAce) {
		t.Errorf("card after reset = %v, want heartAce", trick.Cards[0].Card)
	}
}

// --- Full 6-player trick ---

func TestTrickCards_sixPlayers(t *testing.T) {
	trump := card.Spades
	trick := NewTrick(0, trump)
	for seat := 0; seat < 6; seat++ {
		trick.Add(card.Card{Suit: card.Hearts, Rank: card.Rank(seat), CopyIndex: 0}, seat)
	}
	if trick.NCards != 6 {
		t.Fatalf("6-player trick has %d cards, want 6", trick.NCards)
	}
	for i, pc := range trick.Cards[:trick.NCards] {
		if pc.Seat != i {
			t.Errorf("Cards[%d].Seat = %d, want %d", i, pc.Seat, i)
		}
		if pc.PlayOrder != i {
			t.Errorf("Cards[%d].PlayOrder = %d, want %d", i, pc.PlayOrder, i)
		}
	}
}

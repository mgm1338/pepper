package game

import (
	"testing"

	"github.com/max/pepper/internal/card"
)

func TestValidPlays_leadReturnsAll(t *testing.T) {
	hand := []card.Card{heartAce, trumpAce, diamondAce}
	tr := NewTrick(0, sp)
	got := ValidPlays(hand, tr, sp)
	if len(got) != 3 {
		t.Errorf("leading should return full hand, got %d", len(got))
	}
}

func TestValidPlays_mustFollowSuit(t *testing.T) {
	hand := []card.Card{heartKing, heartNine, trumpAce, diamondAce}
	tr := NewTrick(0, sp)
	tr.Add(heartAce, 0)
	got := ValidPlays(hand, tr, sp)
	if len(got) != 2 {
		t.Fatalf("must follow hearts, got %d plays", len(got))
	}
	for _, c := range got {
		if c.Suit != card.Hearts {
			t.Errorf("got non-heart play %v", c)
		}
	}
}

func TestValidPlays_voidReturnsAll(t *testing.T) {
	hand := []card.Card{trumpAce, diamondAce}
	tr := NewTrick(0, sp)
	tr.Add(heartAce, 0)
	got := ValidPlays(hand, tr, sp)
	if len(got) != 2 {
		t.Errorf("void in suit should return full hand, got %d", len(got))
	}
}

func TestValidPlays_leftBowerFollowsTrump(t *testing.T) {
	// Lead trump. Left bower (club jack) must follow as trump.
	hand := []card.Card{leftBower, heartAce}
	tr := NewTrick(0, sp)
	tr.Add(trumpAce, 0)
	got := ValidPlays(hand, tr, sp)
	if len(got) != 1 || !got[0].Equal(leftBower) {
		t.Errorf("left bower should be the only valid follow, got %v", got)
	}
}

func TestValidPlaysInto_reusesBuffer(t *testing.T) {
	hand := []card.Card{heartKing, trumpAce}
	tr := NewTrick(0, sp)
	tr.Add(heartAce, 0)
	var buf []card.Card
	got := ValidPlaysInto(&buf, hand, tr, sp)
	if len(got) != 1 || !got[0].Equal(heartKing) {
		t.Errorf("expected only heartKing, got %v", got)
	}
}

func TestValidPlaysInto_leadPassthrough(t *testing.T) {
	hand := []card.Card{heartKing, trumpAce}
	tr := NewTrick(0, sp)
	var buf []card.Card
	got := ValidPlaysInto(&buf, hand, tr, sp)
	if len(got) != 2 {
		t.Errorf("leading should pass through hand, got %d", len(got))
	}
}

func TestBestTrump_found(t *testing.T) {
	hand := []card.Card{heartAce, trumpNine, rightBower, trumpAce}
	c, ok := BestTrump(hand, sp)
	if !ok {
		t.Fatal("BestTrump should find trump")
	}
	if !c.Equal(rightBower) {
		t.Errorf("BestTrump = %v, want right bower", c)
	}
}

func TestBestTrump_noTrump(t *testing.T) {
	hand := []card.Card{heartAce, diamondAce}
	_, ok := BestTrump(hand, sp)
	if ok {
		t.Error("BestTrump should not find trump when none present")
	}
}

func TestPepperExchange(t *testing.T) {
	// Caller seat 0 (team 0), partners are 2 and 4.
	caller := []card.Card{
		{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.Ten, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.Queen, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.King, CopyIndex: 0},
		{Suit: card.Diamonds, Rank: card.Nine, CopyIndex: 0},
		{Suit: card.Diamonds, Rank: card.Ten, CopyIndex: 0},
		{Suit: card.Diamonds, Rank: card.Queen, CopyIndex: 0},
		{Suit: card.Diamonds, Rank: card.King, CopyIndex: 0},
	}
	p2 := []card.Card{rightBower, heartAce}
	p4 := []card.Card{trumpAce, heartKing}
	other1 := []card.Card{diamondAce}
	other3 := []card.Card{diamondAce}
	other5 := []card.Card{diamondAce}
	hands := [6][]card.Card{caller, other1, p2, other3, p4, other5}

	give := func(seat int, hand []card.Card, trump card.Suit) card.Card {
		c, _ := BestTrump(hand, trump)
		return c
	}
	discard := func(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
		// discard first two non-trump (there are plenty).
		out := [2]card.Card{}
		idx := 0
		for _, c := range hand {
			if card.TrumpRank(c, trump) < 0 {
				out[idx] = c
				idx++
				if idx == 2 {
					break
				}
			}
		}
		return out
	}

	partners := PepperExchange(&hands, 0, sp, give, discard)
	if partners != [2]int{2, 4} {
		t.Errorf("partners = %v, want [2 4]", partners)
	}
	if len(hands[0]) != 8 {
		t.Errorf("caller hand len = %d, want 8", len(hands[0]))
	}
	if len(hands[2]) != 1 || len(hands[4]) != 1 {
		t.Errorf("partner hands should shrink by 1")
	}
	// Caller should now hold both trump cards given.
	gotRight, gotTA := false, false
	for _, c := range hands[0] {
		if c.Equal(rightBower) {
			gotRight = true
		}
		if c.Equal(trumpAce) {
			gotTA = true
		}
	}
	if !gotRight || !gotTA {
		t.Errorf("caller missing received trump: gotRight=%v gotTA=%v", gotRight, gotTA)
	}
}

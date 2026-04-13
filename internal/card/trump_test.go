package card

import "testing"

// --- TrumpRank ---

func TestTrumpRank_rightBower(t *testing.T) {
	// Jack of trump suit = right bower (rank 13).
	for _, trump := range []Suit{Spades, Clubs, Hearts, Diamonds} {
		c := Card{Suit: trump, Rank: Jack}
		if got := TrumpRank(c, trump); got != TrumpRankRight {
			t.Errorf("TrumpRank(J%v, %v) = %d, want %d (right bower)", trump, trump, got, TrumpRankRight)
		}
	}
}

func TestTrumpRank_leftBower(t *testing.T) {
	// Jack of partner suit = left bower (rank 11).
	cases := []struct{ trump, suit Suit }{
		{Spades, Clubs}, {Clubs, Spades}, {Hearts, Diamonds}, {Diamonds, Hearts},
	}
	for _, tc := range cases {
		c := Card{Suit: tc.suit, Rank: Jack}
		if got := TrumpRank(c, tc.trump); got != TrumpRankLeft {
			t.Errorf("TrumpRank(J%v, trump=%v) = %d, want %d (left bower)", tc.suit, tc.trump, got, TrumpRankLeft)
		}
	}
}

func TestTrumpRank_jackOfOtherSuit(t *testing.T) {
	// Jack of a non-partner suit: not trump.
	// With Spades trump, Hearts is not a partner suit (partner is Clubs).
	c := Card{Suit: Hearts, Rank: Jack}
	if got := TrumpRank(c, Spades); got != TrumpRankNone {
		t.Errorf("TrumpRank(J♥, trump=Spades) = %d, want %d", got, TrumpRankNone)
	}
}

func TestTrumpRank_trumpNonBower(t *testing.T) {
	trump := Spades
	cases := []struct {
		rank Rank
		want int
	}{
		{Ace, TrumpRankAce},
		{King, TrumpRankKing},
		{Queen, TrumpRankQueen},
		{Ten, TrumpRankTen},
		{Nine, TrumpRankNine},
	}
	for _, tc := range cases {
		c := Card{Suit: trump, Rank: tc.rank}
		if got := TrumpRank(c, trump); got != tc.want {
			t.Errorf("TrumpRank(%v%v, %v) = %d, want %d", tc.rank, trump, trump, got, tc.want)
		}
	}
}

func TestTrumpRank_nonTrump(t *testing.T) {
	// Non-trump, non-bower cards return TrumpRankNone (-1).
	// When trump=Hearts, partner suit is Diamonds (same color = red).
	// Non-trump: Spades/Clubs (any rank), Diamonds non-Jack.
	cases := []struct{ suit Suit; rank Rank }{
		{Spades, Ace}, {Spades, King}, {Clubs, Ace}, {Clubs, Queen},
		{Diamonds, Nine}, // Diamonds non-Jack is non-trump when trump=Hearts
		{Spades, Jack},   // J♠ is not a bower when trump=Hearts (partner=Diamonds)
		{Clubs, Jack},    // J♣ is not a bower when trump=Hearts
	}
	for _, tc := range cases {
		c := Card{Suit: tc.suit, Rank: tc.rank}
		if got := TrumpRank(c, Hearts); got != TrumpRankNone {
			t.Errorf("TrumpRank(%v%v, Hearts) = %d, want %d (non-trump)", tc.rank, tc.suit, got, TrumpRankNone)
		}
	}
}

func TestTrumpRank_copyIndexDoesNotMatter(t *testing.T) {
	// Both copies of a card have the same trump rank.
	c0 := Card{Suit: Spades, Rank: Ace, CopyIndex: 0}
	c1 := Card{Suit: Spades, Rank: Ace, CopyIndex: 1}
	if TrumpRank(c0, Spades) != TrumpRank(c1, Spades) {
		t.Error("CopyIndex should not affect TrumpRank")
	}
}

// --- EffectiveSuit ---

func TestEffectiveSuit_leftBowerCountsAsTrump(t *testing.T) {
	// Left bower (Jack of partner suit) counts as trump for suit-following purposes.
	cases := []struct{ trump, suit Suit }{
		{Spades, Clubs}, {Clubs, Spades}, {Hearts, Diamonds}, {Diamonds, Hearts},
	}
	for _, tc := range cases {
		c := Card{Suit: tc.suit, Rank: Jack}
		if got := EffectiveSuit(c, tc.trump); got != tc.trump {
			t.Errorf("EffectiveSuit(J%v, trump=%v) = %v, want %v (left bower = trump)", tc.suit, tc.trump, got, tc.trump)
		}
	}
}

func TestEffectiveSuit_normalCardKeepsSuit(t *testing.T) {
	cases := []struct{ suit Suit; rank Rank }{
		{Hearts, Ace}, {Diamonds, King}, {Clubs, Queen}, {Spades, Ten},
	}
	for _, tc := range cases {
		c := Card{Suit: tc.suit, Rank: tc.rank}
		if got := EffectiveSuit(c, Hearts); got != tc.suit {
			t.Errorf("EffectiveSuit(%v%v, Hearts) = %v, want %v", tc.rank, tc.suit, got, tc.suit)
		}
	}
}

func TestEffectiveSuit_rightBowerKeepsTrumpSuit(t *testing.T) {
	c := Card{Suit: Spades, Rank: Jack}
	if got := EffectiveSuit(c, Spades); got != Spades {
		t.Errorf("EffectiveSuit(right bower) = %v, want Spades", got)
	}
}

func TestEffectiveSuit_jackOfNonPartnerSuitKeepsSuit(t *testing.T) {
	// J♥ with trump=Spades: Hearts is not partner of Spades (Clubs is). Keeps Hearts.
	c := Card{Suit: Hearts, Rank: Jack}
	if got := EffectiveSuit(c, Spades); got != Hearts {
		t.Errorf("EffectiveSuit(J♥, Spades) = %v, want Hearts", got)
	}
}

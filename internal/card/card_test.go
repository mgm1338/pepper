package card

import "testing"

func TestCardEqual(t *testing.T) {
	a := Card{Suit: Spades, Rank: Ace, CopyIndex: 0}
	b := Card{Suit: Spades, Rank: Ace, CopyIndex: 0}
	c := Card{Suit: Spades, Rank: Ace, CopyIndex: 1}
	d := Card{Suit: Hearts, Rank: Ace, CopyIndex: 0}
	if !a.Equal(b) {
		t.Error("identical cards should be Equal")
	}
	if a.Equal(c) {
		t.Error("different CopyIndex should not be Equal")
	}
	if a.Equal(d) {
		t.Error("different suit should not be Equal")
	}
}

func TestCardSameAs(t *testing.T) {
	a := Card{Suit: Spades, Rank: Ace, CopyIndex: 0}
	b := Card{Suit: Spades, Rank: Ace, CopyIndex: 1}
	c := Card{Suit: Hearts, Rank: Ace, CopyIndex: 0}
	if !a.SameAs(b) {
		t.Error("same suit/rank different copy should be SameAs")
	}
	if a.SameAs(c) {
		t.Error("different suit should not be SameAs")
	}
}

func TestCardString(t *testing.T) {
	c := Card{Suit: Hearts, Rank: Ace}
	if got := c.String(); got == "" {
		t.Error("String should not be empty")
	}
}

func TestRankString(t *testing.T) {
	cases := map[Rank]string{
		Nine: "9", Ten: "10", Jack: "J", Queen: "Q", King: "K", Ace: "A",
	}
	for r, want := range cases {
		if got := r.String(); got != want {
			t.Errorf("Rank(%d).String() = %q, want %q", r, got, want)
		}
	}
	if Rank(99).String() != "?" {
		t.Error("unknown rank should return ?")
	}
}

func TestSuitString(t *testing.T) {
	if Spades.String() == "" || Clubs.String() == "" || Hearts.String() == "" || Diamonds.String() == "" {
		t.Error("suit strings should not be empty")
	}
	if Suit(99).String() != "?" {
		t.Error("unknown suit should return ?")
	}
}

func TestSameColor(t *testing.T) {
	if !SameColor(Spades, Clubs) {
		t.Error("Spades and Clubs are both black")
	}
	if !SameColor(Hearts, Diamonds) {
		t.Error("Hearts and Diamonds are both red")
	}
	if SameColor(Spades, Hearts) {
		t.Error("Spades and Hearts differ in color")
	}
	if SameColor(Clubs, Diamonds) {
		t.Error("Clubs and Diamonds differ in color")
	}
}

func TestIsRightBower(t *testing.T) {
	c := Card{Suit: Spades, Rank: Jack}
	if !IsRightBower(c, Spades) {
		t.Error("J of trump should be right bower")
	}
	if IsRightBower(c, Hearts) {
		t.Error("J of non-trump should not be right bower")
	}
	if IsRightBower(Card{Suit: Spades, Rank: Ace}, Spades) {
		t.Error("Ace of trump is not right bower")
	}
}

func TestIsLeftBower(t *testing.T) {
	if !IsLeftBower(Card{Suit: Clubs, Rank: Jack}, Spades) {
		t.Error("J of partner suit should be left bower")
	}
	if IsLeftBower(Card{Suit: Spades, Rank: Jack}, Spades) {
		t.Error("J of trump is right not left")
	}
	if IsLeftBower(Card{Suit: Hearts, Rank: Jack}, Spades) {
		t.Error("J of non-partner should not be left bower")
	}
}

func TestNonTrumpRank(t *testing.T) {
	cases := []struct {
		c    Card
		want int
	}{
		{Card{Rank: Nine}, NonTrumpRankNine},
		{Card{Rank: Ten}, NonTrumpRankTen},
		{Card{Rank: Jack}, NonTrumpRankJack},
		{Card{Rank: Queen}, NonTrumpRankQueen},
		{Card{Rank: King}, NonTrumpRankKing},
		{Card{Rank: Ace}, NonTrumpRankAce},
	}
	for _, tc := range cases {
		if got := NonTrumpRank(tc.c); got != tc.want {
			t.Errorf("NonTrumpRank(%v) = %d, want %d", tc.c, got, tc.want)
		}
	}
}

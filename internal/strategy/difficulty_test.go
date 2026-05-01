package strategy

import (
	"testing"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

func sampleHand() []card.Card {
	return []card.Card{
		{Suit: card.Spades, Rank: card.Jack, CopyIndex: 0},
		{Suit: card.Spades, Rank: card.Ace, CopyIndex: 0},
		{Suit: card.Spades, Rank: card.King, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0},
		{Suit: card.Diamonds, Rank: card.Ten, CopyIndex: 0},
		{Suit: card.Clubs, Rank: card.Queen, CopyIndex: 0},
		{Suit: card.Clubs, Rank: card.Nine, CopyIndex: 0},
	}
}

func TestNewBot_zeroConfig(t *testing.T) {
	b := NewBot(BotConfig{}, nil)
	if b == nil {
		t.Fatal("NewBot returned nil")
	}
}

func TestBot_ChooseTrump(t *testing.T) {
	b := NewBot(BotConfig{}, nil)
	trump := b.ChooseTrump(0, sampleHand())
	if trump < 0 || trump > 3 {
		t.Errorf("ChooseTrump returned invalid suit %v", trump)
	}
}

func TestBot_GivePepper(t *testing.T) {
	b := NewBot(BotConfig{}, nil)
	hand := sampleHand()
	give := b.GivePepper(0, hand, card.Spades)
	found := false
	for _, c := range hand {
		if c.Equal(give) {
			found = true
			break
		}
	}
	if !found {
		t.Error("GivePepper returned card not in hand")
	}
}

func TestBot_PepperDiscard(t *testing.T) {
	b := NewBot(BotConfig{}, nil)
	hand := sampleHand()
	received := [2]card.Card{
		{Suit: card.Spades, Rank: card.Queen, CopyIndex: 0},
		{Suit: card.Spades, Rank: card.Ten, CopyIndex: 0},
	}
	fullHand := append(append([]card.Card{}, hand...), received[0], received[1])
	discards := b.PepperDiscard(0, fullHand, card.Spades, received)
	if discards[0].Equal(discards[1]) {
		t.Error("PepperDiscard returned duplicate cards")
	}
}

func TestBot_Bid(t *testing.T) {
	b := NewBot(BotConfig{}, nil)
	state := game.BidState{
		Hand:        sampleHand(),
		Seat:        0,
		DealerSeat:  5,
		CurrentHigh: 0,
		HighSeat:    -1,
		SeatsLeft:   5,
	}
	bid := b.Bid(0, &state)
	if bid < 0 || (bid > 0 && bid < game.MinBid && bid != game.PepperBid) {
		t.Errorf("Bid returned invalid value %d", bid)
	}
}

func TestBot_Play_validCard(t *testing.T) {
	b := NewBot(BotConfig{Slop: 1.0}, nil) // max slop to exercise random branch
	hand := sampleHand()
	trick := game.NewTrick(1, card.Spades)
	trick.Add(card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}, 1)
	valid := game.ValidPlays(hand, trick, card.Spades)
	state := game.TrickState{
		Trick: trick, Trump: card.Spades, Seat: 0,
		BidderSeat: 0, BidAmount: 4, TrickNumber: 0,
		Hand: hand,
	}
	for i := 0; i < 20; i++ {
		c := b.Play(0, valid, &state)
		found := false
		for _, v := range valid {
			if v.Equal(c) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("Play returned %v not in valid plays", c)
		}
	}
}

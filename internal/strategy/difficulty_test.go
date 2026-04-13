package strategy

import (
	"math/rand"
	"testing"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

func newRNG() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

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

func TestNewDifficulty_clampsLevel(t *testing.T) {
	d := NewDifficulty(0, nil, newRNG())
	if d.cfg != DifficultyLevels[0] {
		t.Error("level 0 should clamp to level 1 cfg")
	}
	d = NewDifficulty(99, nil, newRNG())
	if d.cfg != DifficultyLevels[4] {
		t.Error("level 99 should clamp to level 5 cfg")
	}
	d = NewDifficulty(3, nil, newRNG())
	if d.cfg != DifficultyLevels[2] {
		t.Error("level 3 cfg mismatch")
	}
}

func TestNewLevel45_46_Peter(t *testing.T) {
	d45 := NewLevel45(nil, newRNG())
	if d45.cfg != Level45Config {
		t.Error("Level45 cfg mismatch")
	}
	d46 := NewLevel46(nil, newRNG())
	if d46.cfg != Level46Config {
		t.Error("Level46 cfg mismatch")
	}
	peter := NewPeter(nil, newRNG())
	if peter.cfg != PeterConfig {
		t.Error("Peter cfg mismatch")
	}
}

func TestNewPersonality(t *testing.T) {
	cases := map[string]DifficultyConfig{
		"aggressive": AggressiveConfig,
		"cautious":   CautiousConfig,
		"risky":      RiskyConfig,
		"scientist":  DifficultyLevels[4],
		"unknown":    DifficultyLevels[2],
	}
	for name, want := range cases {
		d := NewPersonality(name, nil, newRNG())
		if d.cfg != want {
			t.Errorf("NewPersonality(%q) cfg mismatch", name)
		}
	}
}

func TestDifficulty_DelegatesMethods(t *testing.T) {
	d := NewDifficulty(3, nil, newRNG())
	hand := sampleHand()

	trump := d.ChooseTrump(0, hand)
	// Any valid suit is fine; ensure it returns a legal Suit value.
	if trump < 0 || trump > 3 {
		t.Errorf("ChooseTrump returned invalid suit %v", trump)
	}

	give := d.GivePepper(0, hand, card.Spades)
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

	received := [2]card.Card{
		{Suit: card.Spades, Rank: card.Queen, CopyIndex: 0},
		{Suit: card.Spades, Rank: card.Ten, CopyIndex: 0},
	}
	fullHand := append(append([]card.Card{}, hand...), received[0], received[1])
	discards := d.PepperDiscard(0, fullHand, card.Spades, received)
	if discards[0].Equal(discards[1]) {
		t.Error("PepperDiscard returned duplicate cards")
	}
}

func TestDifficulty_Bid(t *testing.T) {
	d := NewDifficulty(3, nil, newRNG())
	state := game.BidState{
		Hand:        sampleHand(),
		Seat:        0,
		DealerSeat:  5,
		CurrentHigh: 0,
		HighSeat:    -1,
		SeatsLeft:   5,
	}
	bid := d.Bid(0, state)
	if bid < 0 || (bid > 0 && bid < game.MinBid && bid != game.PepperBid) {
		t.Errorf("Bid returned invalid value %d", bid)
	}
}

func TestDifficulty_Play(t *testing.T) {
	d := NewDifficulty(1, nil, newRNG()) // Level 1 has PlayEps > 0
	hand := sampleHand()
	trick := game.NewTrick(1, card.Spades)
	trick.Add(card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}, 1)
	valid := game.ValidPlays(hand, trick, card.Spades)
	state := game.TrickState{
		Trick: trick, Trump: card.Spades, Seat: 0,
		BidderSeat: 0, BidAmount: 4, TrickNumber: 0,
		Hand: hand,
	}
	// Run multiple times to exercise the epsilon-random branch.
	for i := 0; i < 20; i++ {
		c := d.Play(0, valid, state)
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

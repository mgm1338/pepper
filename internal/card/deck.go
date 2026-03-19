package card

import "math/rand"

// NewPinochleDeck returns all 48 cards in a pinochle deck:
// 9, 10, J, Q, K, A in each of 4 suits, two copies each.
func NewPinochleDeck() []Card {
	cards := make([]Card, 0, 48)
	for _, suit := range []Suit{Spades, Clubs, Hearts, Diamonds} {
		for _, rank := range []Rank{Nine, Ten, Jack, Queen, King, Ace} {
			cards = append(cards, Card{Suit: suit, Rank: rank, CopyIndex: 0})
			cards = append(cards, Card{Suit: suit, Rank: rank, CopyIndex: 1})
		}
	}
	return cards
}

// Shuffle randomly shuffles a slice of cards in place using the provided rng.
func Shuffle(cards []Card, rng *rand.Rand) {
	rng.Shuffle(len(cards), func(i, j int) {
		cards[i], cards[j] = cards[j], cards[i]
	})
}

// Deal shuffles the deck and returns 6 hands of 8 cards each.
func Deal(rng *rand.Rand) [6][]Card {
	deck := NewPinochleDeck()
	Shuffle(deck, rng)
	var hands [6][]Card
	for i := 0; i < 6; i++ {
		hands[i] = make([]Card, 8)
		copy(hands[i], deck[i*8:(i+1)*8])
	}
	return hands
}

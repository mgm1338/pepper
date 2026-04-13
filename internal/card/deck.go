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

// DealAround deals 8 cards to each of the 5 seats other than fixedSeat,
// randomly sampling from the 40 cards not held by fixedSeat.
// The returned hands[fixedSeat] is a copy of fixedHand; all other hands are freshly sampled.
func DealAround(fixedSeat int, fixedHand []Card, rng *rand.Rand) [6][]Card {
	// Build the remaining 40 cards by removing fixedHand from a full deck.
	used := make(map[Card]int, len(fixedHand))
	for _, c := range fixedHand {
		used[c]++
	}
	remaining := make([]Card, 0, 40)
	for _, c := range NewPinochleDeck() {
		if used[c] > 0 {
			used[c]--
		} else {
			remaining = append(remaining, c)
		}
	}
	Shuffle(remaining, rng)

	var hands [6][]Card
	j := 0
	for i := 0; i < 6; i++ {
		if i == fixedSeat {
			hands[i] = make([]Card, len(fixedHand))
			copy(hands[i], fixedHand)
		} else {
			hands[i] = make([]Card, 8)
			copy(hands[i], remaining[j:j+8])
			j += 8
		}
	}
	return hands
}

package game

import "github.com/max/pepper/internal/card"

// PepperExchange performs the card exchange when pepper is called.
// Each partner gives their best trump card to the caller.
// The caller receives 2 cards and discards 2, keeping their hand at 8.
// Returns the updated hands and the seats that will sit out (the two partners).
func PepperExchange(
	hands [6][]card.Card,
	callerSeat int,
	trump card.Suit,
	giveFn func(seat int, hand []card.Card, trump card.Suit) card.Card,
	discardFn func(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card,
) ([6][]card.Card, [2]int) {
	partners := Partners(callerSeat)
	var received [2]card.Card

	// Each partner gives their best trump card.
	for i, partnerSeat := range partners {
		given := giveFn(partnerSeat, hands[partnerSeat], trump)
		received[i] = given
		hands[partnerSeat] = removeCard(hands[partnerSeat], given)
	}

	// Caller receives the 2 cards, then discards 2.
	callerHand := append(hands[callerSeat], received[0], received[1])
	discards := discardFn(callerSeat, callerHand, trump, received)
	for _, d := range discards {
		callerHand = removeCard(callerHand, d)
	}
	hands[callerSeat] = callerHand

	return hands, partners
}

// removeCard removes the first occurrence of target from hand and returns the result.
func removeCard(hand []card.Card, target card.Card) []card.Card {
	result := make([]card.Card, 0, len(hand))
	removed := false
	for _, c := range hand {
		if !removed && c.Equal(target) {
			removed = true
			continue
		}
		result = append(result, c)
	}
	return result
}

// BestTrump returns the highest-ranked trump card in the hand.
// Returns the card and true if found, zero Card and false if no trump in hand.
func BestTrump(hand []card.Card, trump card.Suit) (card.Card, bool) {
	best := card.Card{}
	bestRank := -1
	found := false
	for _, c := range hand {
		r := card.TrumpRank(c, trump)
		if r > bestRank {
			bestRank = r
			best = c
			found = true
		}
	}
	return best, found
}

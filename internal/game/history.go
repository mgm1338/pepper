package game

import "github.com/max/pepper/internal/card"

// HandHistory tracks all cards played in completed tricks this hand.
// Passed to every strategy call so bots can reason about what is still live.
type HandHistory struct {
	played []card.Card // cards from completed tricks, in play order
}

// Record adds a card to the history. Called after each trick completes.
func (h *HandHistory) Record(cards []card.Card) {
	h.played = append(h.played, cards...)
}

// Played returns a copy of all cards seen so far.
func (h *HandHistory) Played() []card.Card {
	cp := make([]card.Card, len(h.played))
	copy(cp, h.played)
	return cp
}

// IsSeen returns true if this specific card (suit+rank+copy) has been played.
func (h *HandHistory) IsSeen(c card.Card) bool {
	for _, p := range h.played {
		if p.Equal(c) {
			return true
		}
	}
	return false
}

// RightBowersPlayed returns how many right bowers have been played (0, 1, or 2).
func (h *HandHistory) RightBowersPlayed(trump card.Suit) int {
	n := 0
	for _, p := range h.played {
		if card.IsRightBower(p, trump) {
			n++
		}
	}
	return n
}

// LeftBowersPlayed returns how many left bowers have been played (0, 1, or 2).
func (h *HandHistory) LeftBowersPlayed(trump card.Suit) int {
	n := 0
	for _, p := range h.played {
		if card.IsLeftBower(p, trump) {
			n++
		}
	}
	return n
}

// TrumpPlayed returns how many trump cards have been played total.
func (h *HandHistory) TrumpPlayed(trump card.Suit) int {
	n := 0
	for _, p := range h.played {
		if card.TrumpRank(p, trump) >= 0 {
			n++
		}
	}
	return n
}

// TrumpRemaining returns how many trump cards have NOT been played yet.
// Total trump in a pinochle deck = 14 (2 right bowers + 2 left bowers + 2 each of A,K,Q,10,9).
func (h *HandHistory) TrumpRemaining(trump card.Suit) int {
	const totalTrump = 14
	return totalTrump - h.TrumpPlayed(trump)
}

// IsTopTrump returns true if c is the highest unplayed trump card.
// Useful for knowing when the left bower is now the best card.
func (h *HandHistory) IsTopTrump(c card.Card, trump card.Suit) bool {
	myRank := card.TrumpRank(c, trump)
	if myRank < 0 {
		return false
	}
	// Check if any higher-ranked trump card is still unplayed.
	allTrump := allTrumpCards(trump)
	for _, t := range allTrump {
		if card.TrumpRank(t, trump) > myRank && !h.IsSeen(t) {
			return false
		}
	}
	return true
}

// SuitVoid returns true if no cards of the given effective suit remain unplayed
// other than what is in the provided hand. Useful for knowing if leading a suit
// is safe (opponents can only ruff, not follow).
func (h *HandHistory) CardsPlayedInSuit(suit card.Suit, trump card.Suit) int {
	n := 0
	for _, p := range h.played {
		if card.EffectiveSuit(p, trump) == suit {
			n++
		}
	}
	return n
}

// allTrumpCards returns all 14 trump cards for a given trump suit.
func allTrumpCards(trump card.Suit) []card.Card {
	partner := card.PartnerSuit(trump)
	var cards []card.Card
	// Right bowers (J of trump, copies 0 and 1).
	cards = append(cards, card.Card{Suit: trump, Rank: card.Jack, CopyIndex: 0})
	cards = append(cards, card.Card{Suit: trump, Rank: card.Jack, CopyIndex: 1})
	// Left bowers (J of partner suit).
	cards = append(cards, card.Card{Suit: partner, Rank: card.Jack, CopyIndex: 0})
	cards = append(cards, card.Card{Suit: partner, Rank: card.Jack, CopyIndex: 1})
	// Remaining trump ranks: A, K, Q, 10, 9 (copies 0 and 1 each).
	for _, rank := range []card.Rank{card.Ace, card.King, card.Queen, card.Ten, card.Nine} {
		cards = append(cards, card.Card{Suit: trump, Rank: rank, CopyIndex: 0})
		cards = append(cards, card.Card{Suit: trump, Rank: rank, CopyIndex: 1})
	}
	return cards
}

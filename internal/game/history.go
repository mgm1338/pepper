package game

import "github.com/max/pepper/internal/card"

// cachedTrumpCards holds pre-computed trump card lists for each suit, indexed by Suit value.
// Built once at init to avoid repeated allocations in IsTopTrump.
var cachedTrumpCards [4][]card.Card

func init() {
	for s := card.Suit(0); s < 4; s++ {
		cachedTrumpCards[s] = buildTrumpCards(s)
	}
}

// maxHandCards is the maximum cards played in a hand: 8 tricks × 6 players.
const maxHandCards = 48

// HandHistory tracks all cards played in completed tricks this hand.
// Passed to every strategy call so bots can reason about what is still live.
// The fixed-size array avoids heap allocation as the history grows.
type HandHistory struct {
	played    [maxHandCards]card.Card
	n         int
	trump     card.Suit
	seatVoids [6]uint8 // bitmask per seat: bit i set if seat is known void in suit i
}

// Reset clears per-hand state for reuse (e.g. from a sync.Pool).
// Trump must be re-set via SetTrump after Reset.
func (h *HandHistory) Reset() {
	h.n = 0
	h.seatVoids = [6]uint8{}
}

// SetTrump records the trump suit for this hand. Must be called before RecordTrick.
func (h *HandHistory) SetTrump(trump card.Suit) { h.trump = trump }

// Voids returns a snapshot of the per-seat void bitmasks.
func (h *HandHistory) Voids() [6]uint8 { return h.seatVoids }

// SetVoids restores a previously-captured void snapshot (used when resuming a rollout).
func (h *HandHistory) SetVoids(v [6]uint8) { h.seatVoids = v }

// IsVoidInSuit reports whether seat has been shown void in the given suit.
func (h *HandHistory) IsVoidInSuit(seat int, suit card.Suit) bool {
	return h.seatVoids[seat]&(1<<uint(suit)) != 0
}

// Record adds cards to the history. Called after each trick completes.
// Does not update void tracking (no seat info); use RecordTrick for that.
func (h *HandHistory) Record(cards []card.Card) {
	h.n += copy(h.played[h.n:], cards)
}

// RecordTrick adds the cards from a completed trick and updates per-seat void tracking.
// cards[0] is the lead card; any follower that plays a different effective suit is void in the led suit.
func (h *HandHistory) RecordTrick(cards []PlayedCard) {
	if len(cards) == 0 {
		return
	}
	ledSuit := card.EffectiveSuit(cards[0].Card, h.trump)
	for _, pc := range cards {
		h.played[h.n] = pc.Card
		h.n++
		if card.EffectiveSuit(pc.Card, h.trump) != ledSuit {
			h.seatVoids[pc.Seat] |= 1 << uint(ledSuit)
		}
	}
}

// Played returns a copy of all cards seen so far.
func (h *HandHistory) Played() []card.Card {
	cp := make([]card.Card, h.n)
	copy(cp, h.played[:h.n])
	return cp
}

// PlayedSlice returns the underlying slice directly (no copy).
// The caller must not modify the returned slice.
// Use in rollout hot paths where the history is read-only.
func (h *HandHistory) PlayedSlice() []card.Card {
	return h.played[:h.n]
}

// IsSeen returns true if this specific card (suit+rank+copy) has been played.
func (h *HandHistory) IsSeen(c card.Card) bool {
	for _, p := range h.played[:h.n] {
		if p.Equal(c) {
			return true
		}
	}
	return false
}

// RightBowersPlayed returns how many right bowers have been played (0, 1, or 2).
func (h *HandHistory) RightBowersPlayed(trump card.Suit) int {
	n := 0
	for _, p := range h.played[:h.n] {
		if card.IsRightBower(p, trump) {
			n++
		}
	}
	return n
}

// LeftBowersPlayed returns how many left bowers have been played (0, 1, or 2).
func (h *HandHistory) LeftBowersPlayed(trump card.Suit) int {
	n := 0
	for _, p := range h.played[:h.n] {
		if card.IsLeftBower(p, trump) {
			n++
		}
	}
	return n
}

// TrumpPlayed returns how many trump cards have been played total.
func (h *HandHistory) TrumpPlayed(trump card.Suit) int {
	n := 0
	for _, p := range h.played[:h.n] {
		if card.TrumpRank(p, trump) >= 0 {
			n++
		}
	}
	return n
}

// TrumpRemaining returns how many trump cards have NOT been played yet.
func (h *HandHistory) TrumpRemaining(trump card.Suit) int {
	return card.TotalTrumpCards - h.TrumpPlayed(trump)
}

// IsTopTrump returns true if c is the highest unplayed trump card.
func (h *HandHistory) IsTopTrump(c card.Card, trump card.Suit) bool {
	myRank := card.TrumpRank(c, trump)
	if myRank < 0 {
		return false
	}
	for _, t := range cachedTrumpCards[trump] {
		if card.TrumpRank(t, trump) > myRank && !h.IsSeen(t) {
			return false
		}
	}
	return true
}

// CardsPlayedInSuit returns how many cards of the given effective suit have been played.
func (h *HandHistory) CardsPlayedInSuit(suit card.Suit, trump card.Suit) int {
	n := 0
	for _, p := range h.played[:h.n] {
		if card.EffectiveSuit(p, trump) == suit {
			n++
		}
	}
	return n
}

// buildTrumpCards constructs the full list of 14 trump cards for a given trump suit.
// Called once per suit at init time; use cachedTrumpCards for lookups.
func buildTrumpCards(trump card.Suit) []card.Card {
	partner := card.PartnerSuit(trump)
	cards := make([]card.Card, 0, card.TotalTrumpCards)
	cards = append(cards, card.Card{Suit: trump, Rank: card.Jack, CopyIndex: 0})
	cards = append(cards, card.Card{Suit: trump, Rank: card.Jack, CopyIndex: 1})
	cards = append(cards, card.Card{Suit: partner, Rank: card.Jack, CopyIndex: 0})
	cards = append(cards, card.Card{Suit: partner, Rank: card.Jack, CopyIndex: 1})
	for _, rank := range []card.Rank{card.Ace, card.King, card.Queen, card.Ten, card.Nine} {
		cards = append(cards, card.Card{Suit: trump, Rank: rank, CopyIndex: 0})
		cards = append(cards, card.Card{Suit: trump, Rank: rank, CopyIndex: 1})
	}
	return cards
}

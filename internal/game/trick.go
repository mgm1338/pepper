package game

import "github.com/max/pepper/internal/card"

// PlayedCard pairs a card with the seat that played it and its play order.
type PlayedCard struct {
	Card      card.Card
	Seat      int
	PlayOrder int // 0 = led first, increasing
}

// Trick holds the cards played in a single trick.
type Trick struct {
	Cards  []PlayedCard
	Trump  card.Suit
	Leader int // seat that led
}

// NewTrick creates an empty trick with the given leader and trump.
func NewTrick(leader int, trump card.Suit) *Trick {
	return &Trick{
		Leader: leader,
		Trump:  trump,
		Cards:  make([]PlayedCard, 0, 6),
	}
}

// Add records a card played by a seat.
func (t *Trick) Add(c card.Card, seat int) {
	t.Cards = append(t.Cards, PlayedCard{
		Card:      c,
		Seat:      seat,
		PlayOrder: len(t.Cards),
	})
}

// LedSuit returns the effective suit of the first card played.
func (t *Trick) LedSuit() card.Suit {
	if len(t.Cards) == 0 {
		panic("no cards played yet")
	}
	return card.EffectiveSuit(t.Cards[0].Card, t.Trump)
}

// Winner returns the seat that wins this trick.
// Rules:
//  1. Highest trump wins if any trump was played.
//  2. Otherwise highest card of the led suit wins.
//  3. Tie (identical cards): earlier play order wins.
func (t *Trick) Winner() int {
	if len(t.Cards) == 0 {
		panic("no cards played")
	}

	best := t.Cards[0]
	bestTrumpRank := card.TrumpRank(best.Card, t.Trump)
	bestNonRank := card.NonTrumpRank(best.Card)
	ledSuit := card.EffectiveSuit(best.Card, t.Trump)

	for _, pc := range t.Cards[1:] {
		pcTrumpRank := card.TrumpRank(pc.Card, t.Trump)
		pcEffSuit := card.EffectiveSuit(pc.Card, t.Trump)

		if bestTrumpRank >= 0 {
			// Current best is trump. Only a higher trump can beat it.
			if pcTrumpRank > bestTrumpRank {
				best = pc
				bestTrumpRank = pcTrumpRank
				bestNonRank = card.NonTrumpRank(pc.Card)
			} else if pcTrumpRank == bestTrumpRank {
				// Identical trump cards: earlier play order wins (best already has lower order).
				// No change needed.
				_ = bestNonRank
			}
		} else {
			// Current best is not trump.
			if pcTrumpRank >= 0 {
				// First trump played beats any non-trump.
				best = pc
				bestTrumpRank = pcTrumpRank
				bestNonRank = card.NonTrumpRank(pc.Card)
				ledSuit = card.EffectiveSuit(best.Card, t.Trump) // update for clarity
			} else if pcEffSuit == ledSuit {
				// Same led suit, no trump involved.
				pcNonRank := card.NonTrumpRank(pc.Card)
				if pcNonRank > bestNonRank {
					best = pc
					bestNonRank = pcNonRank
				}
				// Equal non-trump ranks: earlier play order wins (no change).
			}
			// Off-suit, non-trump: cannot win.
		}
	}

	return best.Seat
}

// TrickState is the view passed to a strategy's Play method.
type TrickState struct {
	Trick       *Trick
	Trump       card.Suit
	Seat        int
	BidderSeat  int        // seat that won the bid this hand
	BidAmount   int        // what the bidder bid
	TrickNumber int        // 0-indexed
	TricksTaken [6]int     // tricks taken so far this hand by each seat
	Scores      [2]int
	History     *HandHistory // all cards played in completed tricks
	Hand        []card.Card  // full hand at time of decision (superset of valid plays)
}

// ValidPlays returns the subset of hand that are legal to play.
// A player must follow the led suit (using effective suit) if possible.
func ValidPlays(hand []card.Card, trick *Trick, trump card.Suit) []card.Card {
	if len(trick.Cards) == 0 {
		// Leading: any card is valid.
		return hand
	}

	ledSuit := trick.LedSuit()
	var followers []card.Card
	for _, c := range hand {
		if card.EffectiveSuit(c, trump) == ledSuit {
			followers = append(followers, c)
		}
	}
	if len(followers) > 0 {
		return followers
	}
	// No cards of led suit: any card is valid.
	return hand
}

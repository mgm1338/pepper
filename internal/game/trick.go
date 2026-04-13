package game

import "github.com/max/pepper/internal/card"

// PlayedCard pairs a card with the seat that played it and its play order.
type PlayedCard struct {
	Card      card.Card
	Seat      int
	PlayOrder int // 0 = led first, increasing
}

// Trick holds the cards played in a single trick.
// winnerIdx, winnerTR, winnerNR, and ledSuit are maintained incrementally in Add
// so that Winner() and WinnerCard() are O(1) field reads.
type Trick struct {
	Cards     []PlayedCard
	Trump     card.Suit
	Leader    int
	winnerIdx int       // index into Cards of current winner
	winnerTR  int       // TrumpRank of current winner (-1 if non-trump)
	winnerNR  int       // NonTrumpRank of current winner
	ledSuit   card.Suit // effective suit of the led card
}

// NewTrick creates an empty trick with the given leader and trump.
func NewTrick(leader int, trump card.Suit) *Trick {
	return &Trick{
		Leader: leader,
		Trump:  trump,
		Cards:  make([]PlayedCard, 0, 6),
	}
}

// Reset resets the trick in place for reuse — avoids allocation in rollout loops.
func (t *Trick) Reset(leader int, trump card.Suit) {
	t.Leader = leader
	t.Trump = trump
	t.Cards = t.Cards[:0]
	t.winnerIdx = 0
	t.winnerTR = -1
}

// Add records a card played by a seat and updates the winner incrementally.
func (t *Trick) Add(c card.Card, seat int) {
	idx := len(t.Cards)
	t.Cards = append(t.Cards, PlayedCard{Card: c, Seat: seat, PlayOrder: idx})

	newTR := card.TrumpRank(c, t.Trump)

	if idx == 0 {
		// First card: becomes leader and initial winner.
		t.winnerIdx = 0
		t.winnerTR = newTR
		t.winnerNR = card.NonTrumpRank(c)
		t.ledSuit = card.EffectiveSuit(c, t.Trump)
		return
	}

	if t.winnerTR >= 0 {
		// Current winner is trump — only a strictly higher trump can beat it.
		if newTR > t.winnerTR {
			t.winnerIdx = idx
			t.winnerTR = newTR
			t.winnerNR = card.NonTrumpRank(c)
		}
		// Equal trump rank: earlier play order wins — no change.
	} else {
		// Current winner is not trump.
		if newTR >= 0 {
			// First trump played beats any non-trump.
			t.winnerIdx = idx
			t.winnerTR = newTR
			t.winnerNR = card.NonTrumpRank(c)
		} else if card.EffectiveSuit(c, t.Trump) == t.ledSuit {
			// Same led suit, no trump — higher rank wins.
			if nr := card.NonTrumpRank(c); nr > t.winnerNR {
				t.winnerIdx = idx
				t.winnerNR = nr
			}
			// Equal non-trump ranks: earlier play order wins — no change.
		}
		// Off-suit non-trump: cannot win.
	}
}

// LedSuit returns the effective suit of the first card played.
func (t *Trick) LedSuit() card.Suit {
	if len(t.Cards) == 0 {
		panic("no cards played yet")
	}
	return t.ledSuit
}

// Winner returns the seat that currently wins this trick (O(1)).
func (t *Trick) Winner() int {
	if len(t.Cards) == 0 {
		panic("no cards played")
	}
	return t.Cards[t.winnerIdx].Seat
}

// WinnerCard returns the card currently winning this trick (O(1)).
func (t *Trick) WinnerCard() card.Card {
	if len(t.Cards) == 0 {
		panic("no cards played")
	}
	return t.Cards[t.winnerIdx].Card
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
	return hand
}

// ValidPlaysInto writes legal plays into *dst (reused across calls) and returns the result.
// When leading or void in suit, returns hand directly without touching *dst.
// The pointer lets the caller's buffer grow once and stay allocated — zero allocs after warmup.
func ValidPlaysInto(dst *[]card.Card, hand []card.Card, trick *Trick, trump card.Suit) []card.Card {
	if len(trick.Cards) == 0 {
		return hand
	}
	ledSuit := trick.LedSuit()
	*dst = (*dst)[:0]
	for _, c := range hand {
		if card.EffectiveSuit(c, trump) == ledSuit {
			*dst = append(*dst, c)
		}
	}
	if len(*dst) > 0 {
		return *dst
	}
	return hand
}

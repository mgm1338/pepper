package strategy

import (
	"github.com/max/pepper/internal/card"
)

// HandEval holds a breakdown of hand strength for bidding decisions.
type HandEval struct {
	Trump          card.Suit
	TrumpCount     int     // total trump cards in hand
	RightBowers    int     // 0, 1, or 2
	LeftBowers     int     // 0, 1, or 2
	HighTrump      int     // A or K trump cards
	VoidSuits      int     // suits with zero cards (excluding trump)
	SingletonSuits int     // suits with exactly one card (excluding trump)
	TrickEstimate  float64 // estimated tricks caller can take alone
}

// EvaluateHand scores a hand for a given (potential) trump suit.
func EvaluateHand(hand []card.Card, trump card.Suit) HandEval {
	e := HandEval{Trump: trump}

	// Count trump components.
	for _, c := range hand {
		r := card.TrumpRank(c, trump)
		if r < 0 {
			continue
		}
		e.TrumpCount++
		switch {
		case card.IsRightBower(c, trump):
			e.RightBowers++
		case card.IsLeftBower(c, trump):
			e.LeftBowers++
		case c.Rank == card.Ace || c.Rank == card.King:
			e.HighTrump++
		}
	}

	// Count non-trump suit lengths.
	suitCounts := [4]int{}
	suits := []card.Suit{card.Spades, card.Clubs, card.Hearts, card.Diamonds}
	for _, c := range hand {
		if card.EffectiveSuit(c, trump) != trump {
			suitCounts[c.Suit]++
		}
	}
	for _, s := range suits {
		if s == trump {
			continue
		}
		switch suitCounts[s] {
		case 0:
			e.VoidSuits++
		case 1:
			e.SingletonSuits++
		}
	}

	// Estimate tricks the caller can take alone.
	//
	// Calibration targets (6-handed, 8 tricks):
	//   Right bower: nearly certain winner (~95%)
	//   Left bower: very strong but loses to right bower (~85%)
	//   Ace/King of trump: solid but others hold high trump too (~55%)
	//   Lower trump (Q/10/9): unlikely to win a trick outright (~30%)
	//   Void suit: can ruff an ace lead, but partner may lead trump first (~50%)
	//   Singleton: some ruffing value, less reliable (~25%)
	//
	// This produces bids of 4-5 on typical hands, 6 on very strong hands.
	otherTrump := max(0, e.TrumpCount-e.RightBowers-e.LeftBowers-e.HighTrump)
	e.TrickEstimate = float64(e.RightBowers)*0.95 +
		float64(e.LeftBowers)*0.85 +
		float64(e.HighTrump)*0.55 +
		float64(otherTrump)*0.30 +
		float64(e.VoidSuits)*0.50 +
		float64(e.SingletonSuits)*0.25

	return e
}

// BestTrumpSuit returns the suit that produces the strongest HandEval.
func BestTrumpSuit(hand []card.Card) (card.Suit, HandEval) {
	suits := []card.Suit{card.Spades, card.Clubs, card.Hearts, card.Diamonds}
	bestSuit := card.Spades
	bestEval := EvaluateHand(hand, card.Spades)
	for _, s := range suits[1:] {
		e := EvaluateHand(hand, s)
		if e.TrickEstimate > bestEval.TrickEstimate {
			bestEval = e
			bestSuit = s
		}
	}
	return bestSuit, bestEval
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

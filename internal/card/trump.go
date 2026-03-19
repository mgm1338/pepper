package card

// EffectiveSuit returns the suit a card belongs to for trick-following purposes.
// A left bower (Jack of the partner suit) counts as trump, not its printed suit.
func EffectiveSuit(c Card, trump Suit) Suit {
	if IsLeftBower(c, trump) {
		return trump
	}
	return c.Suit
}

// IsRightBower returns true if the card is the Jack of the trump suit.
func IsRightBower(c Card, trump Suit) bool {
	return c.Rank == Jack && c.Suit == trump
}

// IsLeftBower returns true if the card is the Jack of the same-color non-trump suit.
func IsLeftBower(c Card, trump Suit) bool {
	return c.Rank == Jack && c.Suit == PartnerSuit(trump) && c.Suit != trump
}

// TrumpRank returns a numeric rank for a trump card (higher = stronger).
// Non-trump cards return -1.
// Ranking: right bower (13), left bower (11), A (9), K (7), Q (5), 10 (3), 9 (1).
// The two copies of each share the same base value; the trick engine uses play
// order to break ties between identical cards.
func TrumpRank(c Card, trump Suit) int {
	if IsRightBower(c, trump) {
		return 13
	}
	if IsLeftBower(c, trump) {
		return 11
	}
	if EffectiveSuit(c, trump) != trump {
		return -1 // not a trump card
	}
	switch c.Rank {
	case Ace:
		return 9
	case King:
		return 7
	case Queen:
		return 5
	case Ten:
		return 3
	case Nine:
		return 1
	}
	return -1
}

// NonTrumpRank returns a numeric rank for a non-trump card within its suit.
// Used to compare cards of the same led suit when no trump is played.
func NonTrumpRank(c Card) int {
	switch c.Rank {
	case Ace:
		return 6
	case King:
		return 5
	case Queen:
		return 4
	case Jack:
		return 3
	case Ten:
		return 2
	case Nine:
		return 1
	}
	return 0
}

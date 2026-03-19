package card

// TotalTrumpCards is the number of trump cards in a pinochle deck for any suit.
// Includes 2 right bowers, 2 left bowers, and 2 each of A, K, Q, 10, 9.
const TotalTrumpCards = 14

// Trump rank values (higher = stronger). Non-trump returns TrumpRankNone.
const (
	TrumpRankNone  = -1
	TrumpRankNine  = 1
	TrumpRankTen   = 3
	TrumpRankQueen = 5
	TrumpRankKing  = 7
	TrumpRankAce   = 9
	TrumpRankLeft  = 11
	TrumpRankRight = 13
)

// Non-trump rank values within a suit (higher = stronger).
const (
	NonTrumpRankNine  = 1
	NonTrumpRankTen   = 2
	NonTrumpRankJack  = 3
	NonTrumpRankQueen = 4
	NonTrumpRankKing  = 5
	NonTrumpRankAce   = 6
)

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
// Non-trump cards return TrumpRankNone (-1).
// Ranking: right bower (13), left bower (11), A (9), K (7), Q (5), 10 (3), 9 (1).
// The two copies of each share the same base value; the trick engine uses play
// order to break ties between identical cards.
func TrumpRank(c Card, trump Suit) int {
	if IsRightBower(c, trump) {
		return TrumpRankRight
	}
	if IsLeftBower(c, trump) {
		return TrumpRankLeft
	}
	if EffectiveSuit(c, trump) != trump {
		return TrumpRankNone
	}
	switch c.Rank {
	case Ace:
		return TrumpRankAce
	case King:
		return TrumpRankKing
	case Queen:
		return TrumpRankQueen
	case Ten:
		return TrumpRankTen
	case Nine:
		return TrumpRankNine
	}
	return TrumpRankNone
}

// NonTrumpRank returns a numeric rank for a non-trump card within its suit.
// Used to compare cards of the same led suit when no trump is played.
func NonTrumpRank(c Card) int {
	switch c.Rank {
	case Ace:
		return NonTrumpRankAce
	case King:
		return NonTrumpRankKing
	case Queen:
		return NonTrumpRankQueen
	case Jack:
		return NonTrumpRankJack
	case Ten:
		return NonTrumpRankTen
	case Nine:
		return NonTrumpRankNine
	}
	return 0
}

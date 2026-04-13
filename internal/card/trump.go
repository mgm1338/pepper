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

// trumpRankTable[suit][rank][trump] gives TrumpRank for any (suit, rank, trump) combination.
// Indexed as [c.Suit][c.Rank][trump]. CopyIndex is irrelevant to rank.
// 4×6×4 = 96 int8 entries (96 bytes, fits in L1 cache permanently).
var trumpRankTable [4][6][4]int8

// effectiveSuitTable[suit][rank][trump] gives EffectiveSuit for any combination.
var effectiveSuitTable [4][6][4]int8

func init() {
	for trump := Suit(0); trump < 4; trump++ {
		partner := PartnerSuit(trump)
		for suit := Suit(0); suit < 4; suit++ {
			for rank := Rank(0); rank < 6; rank++ {
				// Compute effective suit.
				effSuit := suit
				if rank == Jack && suit == partner {
					effSuit = trump
				}
				effectiveSuitTable[suit][rank][trump] = int8(effSuit)

				// Compute trump rank.
				var tr int8
				if rank == Jack && suit == trump {
					tr = TrumpRankRight
				} else if rank == Jack && suit == partner {
					tr = TrumpRankLeft
				} else if effSuit == trump {
					switch rank {
					case Ace:
						tr = TrumpRankAce
					case King:
						tr = TrumpRankKing
					case Queen:
						tr = TrumpRankQueen
					case Ten:
						tr = TrumpRankTen
					case Nine:
						tr = TrumpRankNine
					default:
						tr = TrumpRankNone
					}
				} else {
					tr = TrumpRankNone
				}
				trumpRankTable[suit][rank][trump] = tr
			}
		}
	}
}

// EffectiveSuit returns the suit a card belongs to for trick-following purposes.
// A left bower (Jack of the partner suit) counts as trump, not its printed suit.
func EffectiveSuit(c Card, trump Suit) Suit {
	return Suit(effectiveSuitTable[c.Suit][c.Rank][trump])
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
	return int(trumpRankTable[c.Suit][c.Rank][trump])
}

// nonTrumpRankTable[rank] gives NonTrumpRank for any rank.
// Indexed by card.Rank (0-5). Jack (rank 4) maps to NonTrumpRankJack.
var nonTrumpRankTable [6]int8

func init() {
	nonTrumpRankTable[Nine]  = NonTrumpRankNine
	nonTrumpRankTable[Ten]   = NonTrumpRankTen
	nonTrumpRankTable[Jack]  = NonTrumpRankJack
	nonTrumpRankTable[Queen] = NonTrumpRankQueen
	nonTrumpRankTable[King]  = NonTrumpRankKing
	nonTrumpRankTable[Ace]   = NonTrumpRankAce
}

// NonTrumpRank returns a numeric rank for a non-trump card within its suit.
// Used to compare cards of the same led suit when no trump is played.
func NonTrumpRank(c Card) int {
	return int(nonTrumpRankTable[c.Rank])
}

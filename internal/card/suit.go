package card

// Suit represents one of the four suits in a pinochle deck.
type Suit int

const (
	Spades Suit = iota
	Clubs
	Hearts
	Diamonds
)

// String returns the single-character suit symbol.
func (s Suit) String() string {
	switch s {
	case Spades:
		return "♠"
	case Clubs:
		return "♣"
	case Hearts:
		return "♥"
	case Diamonds:
		return "♦"
	}
	return "?"
}

// SameColor returns true if two suits share a color.
// Spades/Clubs are black; Hearts/Diamonds are red.
func SameColor(a, b Suit) bool {
	return colorOf(a) == colorOf(b)
}

func colorOf(s Suit) int {
	if s == Spades || s == Clubs {
		return 0 // black
	}
	return 1 // red
}

// PartnerSuit returns the same-color suit that is not s.
// This is the suit whose Jacks become left bowers when s is trump.
func PartnerSuit(s Suit) Suit {
	switch s {
	case Spades:
		return Clubs
	case Clubs:
		return Spades
	case Hearts:
		return Diamonds
	case Diamonds:
		return Hearts
	}
	panic("unknown suit")
}

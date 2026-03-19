package card

import "fmt"

// Card represents a single card in the pinochle deck.
// CopyIndex (0 or 1) distinguishes the two copies of each card.
// When two identical cards are played, CopyIndex 0 is treated as "played first"
// only in the context of trick ordering — actual play order is tracked by the trick.
type Card struct {
	Suit      Suit
	Rank      Rank
	CopyIndex int // 0 or 1
}

func (c Card) String() string {
	return fmt.Sprintf("%s%s", c.Rank, c.Suit)
}

// Equal returns true if two cards have the same suit, rank, and copy index.
func (c Card) Equal(other Card) bool {
	return c.Suit == other.Suit && c.Rank == other.Rank && c.CopyIndex == other.CopyIndex
}

// SameAs returns true if two cards have the same suit and rank (ignoring copy).
func (c Card) SameAs(other Card) bool {
	return c.Suit == other.Suit && c.Rank == other.Rank
}

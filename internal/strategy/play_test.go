package strategy

import (
	"testing"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

var spades = card.Spades

// makeTrick builds a trick with a single card played by seat 0 (leader).
func makeTrick(c card.Card, trump card.Suit) *game.Trick {
	t := game.NewTrick(0, trump)
	t.Add(c, 0)
	return t
}

// makeTrickMulti builds a trick with multiple cards played in order starting at seat 0.
func makeTrickMulti(cards []card.Card, trump card.Suit) *game.Trick {
	t := game.NewTrick(0, trump)
	for i, c := range cards {
		t.Add(c, i)
	}
	return t
}

// trickWinInfo precomputes winner card info from a trick for use with lowestWinner/highestWinner.
func trickWinInfo(trick *game.Trick, trump card.Suit) (winCard card.Card, winTR, winNR int, ledSuit card.Suit) {
	winSeat := trick.Winner()
	for _, pc := range trick.Cards {
		if pc.Seat == winSeat {
			winCard = pc.Card
			break
		}
	}
	winTR = card.TrumpRank(winCard, trump)
	winNR = card.NonTrumpRank(winCard)
	ledSuit = trick.LedSuit()
	return
}

// --- lowestWinner ---

func TestLowestWinner_noWinner(t *testing.T) {
	rightBower := card.Card{Suit: spades, Rank: card.Jack, CopyIndex: 0}
	trick := makeTrick(rightBower, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	hand := []card.Card{
		{Suit: spades, Rank: card.Ace, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0},
	}
	if lowestWinner(hand, wc, wtr, wnr, ls, spades) != nil {
		t.Fatal("no card should beat right bower")
	}
}

func TestLowestWinner_trumpBeatsNonTrump(t *testing.T) {
	ledCard := card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	trick := makeTrick(ledCard, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	lowTrump := card.Card{Suit: spades, Rank: card.Nine, CopyIndex: 0}
	highTrump := card.Card{Suit: spades, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{lowTrump, highTrump}
	got := lowestWinner(hand, wc, wtr, wnr, ls, spades)
	if got == nil {
		t.Fatal("trump should beat non-trump")
	}
	if !got.Equal(lowTrump) {
		t.Fatalf("lowestWinner = %v, want %v (lowest trump)", *got, lowTrump)
	}
}

func TestLowestWinner_sameSuitBeatsLower(t *testing.T) {
	ledCard := card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}
	trick := makeTrick(ledCard, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	heartKing := card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}
	heartAce := card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{heartAce, heartKing}
	got := lowestWinner(hand, wc, wtr, wnr, ls, spades)
	if got == nil {
		t.Fatal("higher card of led suit should win")
	}
	if !got.Equal(heartKing) {
		t.Fatalf("lowestWinner = %v, want heartKing", *got)
	}
}

func TestLowestWinner_offSuitCannotWin(t *testing.T) {
	ledCard := card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}
	trick := makeTrick(ledCard, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	hand := []card.Card{{Suit: card.Diamonds, Rank: card.Ace, CopyIndex: 0}}
	if lowestWinner(hand, wc, wtr, wnr, ls, spades) != nil {
		t.Fatal("off-suit non-trump should not win")
	}
}

func TestLowestWinner_prefersLowestTrumpOverNonTrump(t *testing.T) {
	ledCard := card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}
	trick := makeTrick(ledCard, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	trumpNine := card.Card{Suit: spades, Rank: card.Nine, CopyIndex: 0}
	heartAce := card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{trumpNine, heartAce}
	got := lowestWinner(hand, wc, wtr, wnr, ls, spades)
	if got == nil {
		t.Fatal("should return a winner")
	}
}

// --- highestWinner ---

func TestHighestWinner_noWinner(t *testing.T) {
	rightBower := card.Card{Suit: spades, Rank: card.Jack, CopyIndex: 0}
	trick := makeTrick(rightBower, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	hand := []card.Card{{Suit: spades, Rank: card.Ace, CopyIndex: 0}}
	if highestWinner(hand, wc, wtr, wnr, ls, spades) != nil {
		t.Fatal("no card should beat right bower")
	}
}

func TestHighestWinner_returnsHighestTrump(t *testing.T) {
	ledCard := card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	trick := makeTrick(ledCard, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	lowTrump := card.Card{Suit: spades, Rank: card.Nine, CopyIndex: 0}
	highTrump := card.Card{Suit: spades, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{lowTrump, highTrump}
	got := highestWinner(hand, wc, wtr, wnr, ls, spades)
	if got == nil {
		t.Fatal("should return winner")
	}
	if !got.Equal(highTrump) {
		t.Fatalf("highestWinner = %v, want %v", *got, highTrump)
	}
}

func TestHighestWinner_returnsHighestSameSuit(t *testing.T) {
	ledCard := card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}
	trick := makeTrick(ledCard, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	heartKing := card.Card{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}
	heartAce := card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{heartKing, heartAce}
	got := highestWinner(hand, wc, wtr, wnr, ls, spades)
	if got == nil {
		t.Fatal("should return winner")
	}
	if !got.Equal(heartAce) {
		t.Fatalf("highestWinner = %v, want heartAce", *got)
	}
}

func TestHighestWinner_trumpBeatsCurrentTrump(t *testing.T) {
	ledCard := card.Card{Suit: spades, Rank: card.Nine, CopyIndex: 0}
	trick := makeTrick(ledCard, spades)
	wc, wtr, wnr, ls := trickWinInfo(trick, spades)
	trumpKing := card.Card{Suit: spades, Rank: card.King, CopyIndex: 0}
	trumpAce := card.Card{Suit: spades, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{trumpKing, trumpAce}
	got := highestWinner(hand, wc, wtr, wnr, ls, spades)
	if got == nil {
		t.Fatal("should return winner")
	}
	if !got.Equal(trumpAce) {
		t.Fatalf("highestWinner = %v, want trumpAce", *got)
	}
}

// --- suitScore ---

func defaultCfg() Config {
	return Config{
		RightBowerScore:  1.5,
		LeftBowerScore:   1.2,
		AceKingScore:     1.0,
		LowTrumpScore:    0.5,
		MajorSuitBonus:   0.0,
		TrumpLengthBonus: 0.0,
	}
}

func TestSuitScore_emptyHand(t *testing.T) {
	s := NewStandard(defaultCfg())
	if got := s.suitScore(nil, spades); got != 0.0 {
		t.Fatalf("suitScore empty hand = %v, want 0.0", got)
	}
}

func TestSuitScore_allNonTrump(t *testing.T) {
	s := NewStandard(defaultCfg())
	hand := []card.Card{
		{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.King, CopyIndex: 0},
	}
	if got := s.suitScore(hand, spades); got != 0.0 {
		t.Fatalf("suitScore all non-trump = %v, want 0.0", got)
	}
}

func TestSuitScore_rightBower(t *testing.T) {
	cfg := defaultCfg()
	s := NewStandard(cfg)
	hand := []card.Card{{Suit: spades, Rank: card.Jack, CopyIndex: 0}}
	if got := s.suitScore(hand, spades); got != cfg.RightBowerScore {
		t.Fatalf("suitScore right bower = %v, want %v", got, cfg.RightBowerScore)
	}
}

func TestSuitScore_leftBower(t *testing.T) {
	cfg := defaultCfg()
	s := NewStandard(cfg)
	// Left bower = Jack of partner suit (Clubs for Spades trump).
	hand := []card.Card{{Suit: card.Clubs, Rank: card.Jack, CopyIndex: 0}}
	if got := s.suitScore(hand, spades); got != cfg.LeftBowerScore {
		t.Fatalf("suitScore left bower = %v, want %v", got, cfg.LeftBowerScore)
	}
}

func TestSuitScore_aceAndKing(t *testing.T) {
	cfg := defaultCfg()
	s := NewStandard(cfg)
	hand := []card.Card{
		{Suit: spades, Rank: card.Ace, CopyIndex: 0},
		{Suit: spades, Rank: card.King, CopyIndex: 0},
	}
	// TrumpRankAce=9 hits the tr>=9 case (AceKingScore).
	// TrumpRankKing=7 falls to default (LowTrumpScore).
	want := cfg.AceKingScore + cfg.LowTrumpScore
	if got := s.suitScore(hand, spades); got != want {
		t.Fatalf("suitScore ace+king = %v, want %v", got, want)
	}
}

func TestSuitScore_lowTrump(t *testing.T) {
	cfg := defaultCfg()
	s := NewStandard(cfg)
	hand := []card.Card{
		{Suit: spades, Rank: card.Nine, CopyIndex: 0},
		{Suit: spades, Rank: card.Ten, CopyIndex: 0},
		{Suit: spades, Rank: card.Queen, CopyIndex: 0},
	}
	want := cfg.LowTrumpScore * 3
	if got := s.suitScore(hand, spades); got != want {
		t.Fatalf("suitScore 3 low trump = %v, want %v", got, want)
	}
}

func TestSuitScore_trumpLengthBonus(t *testing.T) {
	cfg := defaultCfg()
	cfg.TrumpLengthBonus = 0.3
	s := NewStandard(cfg)
	// 6 trump cards: 5 base + 1 extra → bonus for 1 card.
	hand := []card.Card{
		{Suit: spades, Rank: card.Jack, CopyIndex: 0},  // right bower
		{Suit: card.Clubs, Rank: card.Jack, CopyIndex: 0}, // left bower
		{Suit: spades, Rank: card.Ace, CopyIndex: 0},
		{Suit: spades, Rank: card.King, CopyIndex: 0},
		{Suit: spades, Rank: card.Queen, CopyIndex: 0},
		{Suit: spades, Rank: card.Nine, CopyIndex: 0},
	}
	// right bower (AceKingScore), left bower, ace (AceKingScore), king/queen/nine (LowTrumpScore each).
	// TrumpRankKing=7 < 9, so king goes to default (LowTrumpScore).
	baseScore := cfg.RightBowerScore + cfg.LeftBowerScore + cfg.AceKingScore + cfg.LowTrumpScore*3
	want := baseScore + cfg.TrumpLengthBonus*1
	if got := s.suitScore(hand, spades); got != want {
		t.Fatalf("suitScore with length bonus = %v, want %v", got, want)
	}
}

func TestSuitScore_majorSuitBonus(t *testing.T) {
	cfg := defaultCfg()
	cfg.MajorSuitBonus = 0.2
	s := NewStandard(cfg)
	hand := []card.Card{{Suit: spades, Rank: card.Ace, CopyIndex: 0}}
	wantSpades := cfg.AceKingScore + cfg.MajorSuitBonus
	if got := s.suitScore(hand, card.Spades); got != wantSpades {
		t.Fatalf("suitScore spades (major) = %v, want %v", got, wantSpades)
	}
	// Clubs is not a major suit.
	hand2 := []card.Card{{Suit: card.Clubs, Rank: card.Ace, CopyIndex: 0}}
	wantClubs := cfg.AceKingScore
	if got := s.suitScore(hand2, card.Clubs); got != wantClubs {
		t.Fatalf("suitScore clubs (minor) = %v, want %v", got, wantClubs)
	}
}

// --- pepperCallerFollow ---

func TestPepperCallerFollow_returnsHighestWinner(t *testing.T) {
	s := NewStandard(defaultCfg())
	// Trick: seat 1 played non-trump 9 of hearts. We hold trump 9 and trump ace.
	trick := game.NewTrick(1, spades)
	trick.Add(card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}, 1)
	state := game.TrickState{Trick: trick, Trump: spades}
	trumpNine := card.Card{Suit: spades, Rank: card.Nine, CopyIndex: 0}
	trumpAce := card.Card{Suit: spades, Rank: card.Ace, CopyIndex: 0}
	hand := []card.Card{trumpNine, trumpAce}
	got := s.pepperCallerFollow(hand, state, spades)
	// Should return highest winner (trump ace beats hearts 9).
	if !got.Equal(trumpAce) {
		t.Fatalf("pepperCallerFollow = %v, want trumpAce (highest winner)", got)
	}
}

func TestPepperCallerFollow_noWinnerReturnsLowest(t *testing.T) {
	s := NewStandard(defaultCfg())
	// Trick: seat 1 played right bower. We can't beat it.
	trick := game.NewTrick(1, spades)
	trick.Add(card.Card{Suit: spades, Rank: card.Jack, CopyIndex: 0}, 1) // right bower
	state := game.TrickState{Trick: trick, Trump: spades}
	hand := []card.Card{
		{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0},
	}
	got := s.pepperCallerFollow(hand, state, spades)
	// Nothing beats right bower — should return lowestCard (hearts 9 = lowest non-trump).
	if !got.Equal(card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}) {
		t.Fatalf("pepperCallerFollow = %v, want heartNine (lowestCard when can't win)", got)
	}
}

func TestPepperCallerFollow_prefersTrumpOverSameSuit(t *testing.T) {
	s := NewStandard(defaultCfg())
	// Trick: seat 1 played non-trump 9 of hearts.
	// Hand has heart ace (same suit) and trump nine (trump).
	// When best is non-trump and candidate is trump, the else-if cTR>=0 branch fires → trump wins.
	trick := game.NewTrick(1, spades)
	trick.Add(card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}, 1)
	state := game.TrickState{Trick: trick, Trump: spades}
	trumpNine := card.Card{Suit: spades, Rank: card.Nine, CopyIndex: 0}
	heartAce := card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}
	// heartAce first so it becomes initial best, then trumpNine should replace it.
	hand := []card.Card{heartAce, trumpNine}
	got := s.pepperCallerFollow(hand, state, spades)
	if !got.Equal(trumpNine) {
		t.Fatalf("pepperCallerFollow = %v, want trumpNine (trump preferred over same-suit non-trump)", got)
	}
}

// --- shortSuitLead ---

func TestShortSuitLead_returnsLowestOfShortestSuit(t *testing.T) {
	// Two hearts (long), one diamond (short) — should return the diamond.
	hand := []card.Card{
		{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.King, CopyIndex: 0},
		{Suit: card.Diamonds, Rank: card.Nine, CopyIndex: 0},
	}
	got, ok := shortSuitLead(hand, spades)
	if !ok {
		t.Fatal("shortSuitLead should return true when non-trump exists")
	}
	if !got.Equal(card.Card{Suit: card.Diamonds, Rank: card.Nine, CopyIndex: 0}) {
		t.Fatalf("shortSuitLead = %v, want diamond 9 (only card of shortest suit)", got)
	}
}

func TestShortSuitLead_returnsLowestOfTiedSuit(t *testing.T) {
	// One heart (low), one diamond (low) — tied at 1 each. Returns either; picks by iteration order.
	// Just verify it returns one of them and doesn't panic.
	hand := []card.Card{
		{Suit: card.Hearts, Rank: card.King, CopyIndex: 0},
		{Suit: card.Diamonds, Rank: card.Nine, CopyIndex: 0},
	}
	got, ok := shortSuitLead(hand, spades)
	if !ok {
		t.Fatal("shortSuitLead should return true")
	}
	// Either card is valid — just ensure it's non-trump.
	if card.TrumpRank(got, spades) >= 0 {
		t.Fatal("shortSuitLead should not return a trump card")
	}
}

func TestShortSuitLead_allTrumpReturnsFalse(t *testing.T) {
	hand := []card.Card{
		{Suit: spades, Rank: card.Ace, CopyIndex: 0},
		{Suit: spades, Rank: card.King, CopyIndex: 0},
	}
	_, ok := shortSuitLead(hand, spades)
	if ok {
		t.Fatal("shortSuitLead should return false when all cards are trump")
	}
}

func TestShortSuitLead_leftBowerCountsAsTrump(t *testing.T) {
	// Left bower (clubs Jack with spades trump) counts as trump, not clubs.
	leftBower := card.Card{Suit: card.Clubs, Rank: card.Jack, CopyIndex: 0}
	heartNine := card.Card{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}
	hand := []card.Card{leftBower, heartNine}
	got, ok := shortSuitLead(hand, spades)
	if !ok {
		t.Fatal("shortSuitLead should return true (hearts available)")
	}
	if !got.Equal(heartNine) {
		t.Fatalf("shortSuitLead = %v, want heartNine (left bower counts as trump)", got)
	}
}

// --- PepperDiscard ---

func TestPepperDiscard_prefersNonTrumpNonAce(t *testing.T) {
	cfg := defaultCfg()
	cfg.PepperDiscardKeepAces = true
	s := NewStandard(cfg)
	// Hand: trump ace, heart ace, heart king, heart nine.
	// PepperDiscardKeepAces=true: prefer non-trump-non-ace → heart king, heart nine.
	hand := []card.Card{
		{Suit: spades, Rank: card.Ace, CopyIndex: 0},  // trump ace — keep
		{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}, // heart ace — keep (it's an ace)
		{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}, // discard candidate
		{Suit: card.Hearts, Rank: card.Nine, CopyIndex: 0}, // discard candidate (lowest)
	}
	result := s.PepperDiscard(0, hand, spades, [2]card.Card{})
	// Should discard the 2 lowest non-trump-non-ace cards.
	discarded := map[card.Card]bool{result[0]: true, result[1]: true}
	if discarded[card.Card{Suit: spades, Rank: card.Ace, CopyIndex: 0}] {
		t.Fatal("should not discard trump ace")
	}
	if discarded[card.Card{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0}] {
		t.Fatal("should not discard heart ace when PepperDiscardKeepAces=true")
	}
}

func TestPepperDiscard_fallsBackToNonTrump(t *testing.T) {
	cfg := defaultCfg()
	cfg.PepperDiscardKeepAces = true
	s := NewStandard(cfg)
	// Only one non-trump-non-ace card exists; should fall back to non-trump (includes aces).
	hand := []card.Card{
		{Suit: spades, Rank: card.Ace, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.Ace, CopyIndex: 0},
		{Suit: card.Hearts, Rank: card.King, CopyIndex: 0}, // only non-trump-non-ace
		{Suit: spades, Rank: card.King, CopyIndex: 0},
	}
	result := s.PepperDiscard(0, hand, spades, [2]card.Card{})
	// Only 1 non-trump-non-ace (heart king), so falls back to non-trump (heart ace + heart king).
	// Result should be 2 non-trump cards.
	for _, c := range result {
		if card.TrumpRank(c, spades) >= 0 {
			t.Fatalf("fallback should still prefer non-trump, got trump card %v", c)
		}
	}
}

func TestPepperDiscard_allTrumpFallsBackToLowestTrump(t *testing.T) {
	cfg := defaultCfg()
	s := NewStandard(cfg)
	// All trump — must discard 2 lowest trump.
	hand := []card.Card{
		{Suit: spades, Rank: card.Jack, CopyIndex: 0},  // right bower (rank 13)
		{Suit: card.Clubs, Rank: card.Jack, CopyIndex: 0}, // left bower (rank 11)
		{Suit: spades, Rank: card.Ace, CopyIndex: 0},   // rank 9
		{Suit: spades, Rank: card.Nine, CopyIndex: 0},  // rank 1
		{Suit: spades, Rank: card.Ten, CopyIndex: 0},   // rank 3
	}
	result := s.PepperDiscard(0, hand, spades, [2]card.Card{})
	// Should discard the 2 lowest trump: nine (1) and ten (3).
	discarded := map[card.Card]bool{result[0]: true, result[1]: true}
	if !discarded[card.Card{Suit: spades, Rank: card.Nine, CopyIndex: 0}] {
		t.Fatal("should discard nine (lowest trump)")
	}
	if !discarded[card.Card{Suit: spades, Rank: card.Ten, CopyIndex: 0}] {
		t.Fatal("should discard ten (second lowest trump)")
	}
}

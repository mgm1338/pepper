package game

import (
	"fmt"
	"strings"

	"github.com/max/pepper/internal/card"
)

// Logger receives game events for display or recording.
// All methods are optional — use NoopLogger for silent play.
type Logger interface {
	OnDeal(hands [6][]card.Card)
	OnBid(seat, bid int, isPepper, isStuck bool)
	OnBidWon(seat, bid int, trump card.Suit, isPepper bool)
	OnPepperExchange(callerSeat int, given [2]card.Card, discarded [2]card.Card)
	OnCardPlayed(trickNum, seat int, c card.Card)
	OnTrickWon(trickNum, seat int)
	OnHandResult(result HandResult, scores [2]int)
	OnGameOver(winner TeamIndex, scores [2]int, rounds int)
}

// NoopLogger discards all events.
type NoopLogger struct{}

func (NoopLogger) OnDeal([6][]card.Card)                                          {}
func (NoopLogger) OnBid(seat, bid int, isPepper, isStuck bool)                   {}
func (NoopLogger) OnBidWon(seat, bid int, trump card.Suit, isPepper bool)        {}
func (NoopLogger) OnPepperExchange(int, [2]card.Card, [2]card.Card)              {}
func (NoopLogger) OnCardPlayed(trickNum, seat int, c card.Card)                  {}
func (NoopLogger) OnTrickWon(trickNum, seat int)                                 {}
func (NoopLogger) OnHandResult(result HandResult, scores [2]int)                 {}
func (NoopLogger) OnGameOver(winner TeamIndex, scores [2]int, rounds int)        {}

// PrintLogger prints a human-readable game trace to stdout.
type PrintLogger struct {
	TeamNames  [2]string
	ShowTricks bool // if false, only show bid and hand result
}

func NewPrintLogger(showTricks bool) *PrintLogger {
	return &PrintLogger{
		TeamNames:  [2]string{"Team0(0,2,4)", "Team1(1,3,5)"},
		ShowTricks: showTricks,
	}
}

func (l *PrintLogger) OnDeal(hands [6][]card.Card) {
	fmt.Println(strings.Repeat("─", 60))
	fmt.Println("DEAL:")
	for seat, hand := range hands {
		team := seat % 2
		fmt.Printf("  Seat %d (Team%d): %s\n", seat, team, formatHand(hand))
	}
}

func (l *PrintLogger) OnBid(seat, bid int, isPepper, isStuck bool) {
	switch {
	case isStuck:
		fmt.Printf("  Seat %d (dealer): STUCK at 3\n", seat)
	case isPepper:
		fmt.Printf("  Seat %d: PEPPER!\n", seat)
	case bid == 0:
		fmt.Printf("  Seat %d: pass\n", seat)
	default:
		fmt.Printf("  Seat %d: %d\n", seat, bid)
	}
}

func (l *PrintLogger) OnBidWon(seat, bid int, trump card.Suit, isPepper bool) {
	if isPepper {
		fmt.Printf("  → Seat %d calls PEPPER, trump = %s\n", seat, trump)
	} else {
		fmt.Printf("  → Seat %d wins bid at %d, calls trump = %s\n", seat, bid, trump)
	}
}

func (l *PrintLogger) OnPepperExchange(callerSeat int, given [2]card.Card, discarded [2]card.Card) {
	fmt.Printf("  Pepper exchange: received %s %s, discarded %s %s\n",
		given[0], given[1], discarded[0], discarded[1])
}

func (l *PrintLogger) OnCardPlayed(trickNum, seat int, c card.Card) {
	if l.ShowTricks {
		fmt.Printf("    Seat %d plays %s\n", seat, c)
	}
}

func (l *PrintLogger) OnTrickWon(trickNum, seat int) {
	if l.ShowTricks {
		fmt.Printf("  → Trick %d won by Seat %d (Team%d)\n", trickNum+1, seat, seat%2)
	}
}

func (l *PrintLogger) OnHandResult(result HandResult, scores [2]int) {
	fmt.Println()
	if result.IsPepper {
		if result.MadeBid {
			fmt.Printf("  PEPPER MADE! Seat %d (Team%d) takes all 8. +16\n",
				result.BidderSeat, int(TeamOf(result.BidderSeat)))
		} else {
			fmt.Printf("  PEPPER MISSED! Seat %d (Team%d) -16, opponents +%d\n",
				result.BidderSeat, int(TeamOf(result.BidderSeat)),
				result.ScoreDelta[1-int(TeamOf(result.BidderSeat))])
		}
	} else {
		bidTeam := int(TeamOf(result.BidderSeat))
		if result.MadeBid {
			fmt.Printf("  Bid %d MADE — Team%d took %d tricks (+%d), Team%d took %d tricks (+%d)\n",
				result.BidAmount,
				bidTeam, result.TricksTaken[bidTeam], result.ScoreDelta[bidTeam],
				1-bidTeam, result.TricksTaken[1-bidTeam], result.ScoreDelta[1-bidTeam])
		} else {
			fmt.Printf("  Bid %d MISSED — Team%d took only %d tricks (-%d), Team%d took %d tricks (+%d)\n",
				result.BidAmount,
				bidTeam, result.TricksTaken[bidTeam], result.BidAmount,
				1-bidTeam, result.TricksTaken[1-bidTeam], result.ScoreDelta[1-bidTeam])
		}
	}
	fmt.Printf("  Score: Team0=%d  Team1=%d\n", scores[0], scores[1])
}

func (l *PrintLogger) OnGameOver(winner TeamIndex, scores [2]int, rounds int) {
	fmt.Println(strings.Repeat("═", 60))
	fmt.Printf("GAME OVER — Team%d wins after %d hands\n", winner, rounds)
	fmt.Printf("Final score: Team0=%d  Team1=%d\n", scores[0], scores[1])
}

func formatHand(hand []card.Card) string {
	parts := make([]string, len(hand))
	for i, c := range hand {
		parts[i] = c.String()
	}
	return strings.Join(parts, " ")
}

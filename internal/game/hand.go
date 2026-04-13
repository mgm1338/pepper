package game

import (
	"math/rand"

	"github.com/max/pepper/internal/card"
)

// Strategy is the interface every bot must implement.
type Strategy interface {
	// Bid returns the player's bid: 0=pass, 8=pepper, else a number in 4–7.
	Bid(seat int, state BidState) int

	// Play returns the card the player wants to play.
	// The engine guarantees only valid plays are accepted.
	Play(seat int, hand []card.Card, state TrickState) card.Card

	// GivePepper returns the best trump card to give the pepper caller.
	GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card

	// PepperDiscard returns 2 cards to remove from the caller's hand after receiving partner cards.
	PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card

	// ChooseTrump returns the trump suit the winner wants to call.
	ChooseTrump(seat int, hand []card.Card) card.Suit
}

// PlayHand runs a complete hand: deal, bid, play tricks, and return the result.
// Pass a NoopLogger{} for silent play, or a PrintLogger for human-readable output.
func PlayHand(gs *GameState, strategies [6]Strategy, rng *rand.Rand, log Logger) HandResult {
	hands := card.Deal(rng)
	log.OnDeal(hands)
	return playHandFrom(gs, strategies, rng, log, hands, BidResult{})
}

// PlayHandFrom runs a hand using pre-dealt cards and a pre-run bid result.
// Use this when you need to inspect the deal/bid before running play.
// If bidResult is zero-value (Winner=0, Amount=0), bidding is re-run internally.
func PlayHandFrom(gs *GameState, strategies [6]Strategy, rng *rand.Rand, log Logger, hands [6][]card.Card, bidResult BidResult) HandResult {
	log.OnDeal(hands)
	return playHandFrom(gs, strategies, rng, log, hands, bidResult)
}

func playHandFrom(gs *GameState, strategies [6]Strategy, rng *rand.Rand, log Logger, hands [6][]card.Card, preBid BidResult) HandResult {

	// --- Bidding ---
	var bidResult BidResult
	if preBid.Amount != 0 || preBid.IsPepper || preBid.IsStuck {
		// Caller provided a pre-run bid result — use it directly.
		bidResult = preBid
	} else {
		bidResult = RunBidding(
			hands,
			gs.Dealer,
			gs.Scores,
			func(seat int, state BidState) int {
				bid := strategies[seat].Bid(seat, state)
				log.OnBid(seat, bid, bid == PepperBid, false)
				return bid
			},
		)
	}
	if bidResult.IsStuck {
		log.OnBid(bidResult.Winner, bidResult.Amount, false, true)
	}

	callerSeat := bidResult.Winner

	// Snapshot the bidder's hand before any pepper exchange for analysis.
	callerOriginalHand := make([]card.Card, len(hands[callerSeat]))
	copy(callerOriginalHand, hands[callerSeat])

	// --- Trump selection ---
	trump := strategies[callerSeat].ChooseTrump(callerSeat, hands[callerSeat])
	log.OnBidWon(callerSeat, bidResult.Amount, trump, bidResult.IsPepper)

	// --- Pepper exchange ---
	var sittingOut [2]int
	pepperActive := false
	if bidResult.IsPepper {
		pepperActive = true
		var given [2]card.Card
		var discarded [2]card.Card
		sittingOut = PepperExchange(
			&hands,
			callerSeat,
			trump,
			func(seat int, hand []card.Card, trump card.Suit) card.Card {
				c := strategies[seat].GivePepper(seat, hand, trump)
				return c
			},
			func(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
				given = received
				d := strategies[seat].PepperDiscard(seat, hand, trump, received)
				discarded = d
				return d
			},
		)
		log.OnPepperExchange(callerSeat, given, discarded)
	}

	// Build the set of active seats for this hand.
	activeSeatSet := buildActiveSeatSet(pepperActive, callerSeat, sittingOut)

	// --- Count pre-play trump distribution ---
	bidderTeam := TeamOf(callerSeat)
	var bidderTrumpCount, oppTrumpCount int
	for seat := 0; seat < 6; seat++ {
		for _, c := range hands[seat] {
			if card.TrumpRank(c, trump) >= 0 {
				if TeamOf(seat) == bidderTeam {
					bidderTrumpCount++
				} else {
					oppTrumpCount++
				}
			}
		}
	}
	trumpStats := TrumpStats{
		TotalTrump:    card.TotalTrumpCards,
		BidderTrump:   bidderTrumpCount,
		OpponentTrump: oppTrumpCount,
	}

	// --- Play tricks ---
	tricksTaken := [6]int{}
	leader := callerSeat
	cumulativeTrumpPlayed := 0
	var history HandHistory

	for t := 0; t < TotalTricks; t++ {
		trick := NewTrick(leader, trump)

		for step := 0; step < len(activeSeatSet); step++ {
			seat := activeSeatSet[(indexInActive(activeSeatSet, leader)+step)%len(activeSeatSet)]
			valid := ValidPlays(hands[seat], trick, trump)
			chosen := strategies[seat].Play(seat, valid, TrickState{
				Trick:       trick,
				Trump:       trump,
				Seat:        seat,
				BidderSeat:  callerSeat,
				BidAmount:   bidResult.Amount,
				TrickNumber: t,
				TricksTaken: tricksTaken,
				Scores:      gs.Scores,
				History:     &history,
				Hand:        hands[seat],
			})
			trick.Add(chosen, seat)
			hands[seat] = removeCard(hands[seat], chosen)
			log.OnCardPlayed(t, seat, chosen)
		}

		// Track trump played this trick.
		if card.EffectiveSuit(trick.Cards[0].Card, trump) == trump {
			trumpStats.TrumpLedsCount++
		}
		for _, pc := range trick.Cards {
			if card.TrumpRank(pc.Card, trump) >= 0 {
				cumulativeTrumpPlayed++
			}
		}
		trumpStats.TrumpPlayedByTrick[t] = cumulativeTrumpPlayed
		history.RecordTrick(trick.Cards)

		winner := trick.Winner()
		tricksTaken[winner]++
		leader = winner
		log.OnTrickWon(t, winner)
	}

	// Aggregate tricks by team.
	var tricksByTeam [2]int
	for seat, count := range tricksTaken {
		tricksByTeam[TeamOf(seat)] += count
	}

	result := ScoreHand(callerSeat, bidResult.Amount, bidResult.IsPepper, tricksByTeam)
	result.IsStuck = bidResult.IsStuck
	result.Trump = trump
	result.BidderHand = callerOriginalHand
	result.TrumpStats = trumpStats

	newScores := [2]int{gs.Scores[0] + result.ScoreDelta[0], gs.Scores[1] + result.ScoreDelta[1]}
	log.OnHandResult(result, newScores)

	return result
}

// buildActiveSeatSet returns the ordered list of seats that will play this hand.
func buildActiveSeatSet(pepperActive bool, callerSeat int, sittingOut [2]int) []int {
	if !pepperActive {
		seats := make([]int, 6)
		for i := range seats {
			seats[i] = i
		}
		return seats
	}
	sitting := map[int]bool{sittingOut[0]: true, sittingOut[1]: true}
	var active []int
	for i := 0; i < 6; i++ {
		if !sitting[i] {
			active = append(active, i)
		}
	}
	return active
}

func indexInActive(active []int, seat int) int {
	for i, s := range active {
		if s == seat {
			return i
		}
	}
	return 0
}

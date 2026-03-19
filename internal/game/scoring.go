package game

import "github.com/max/pepper/internal/card"

// HandResult holds the trick counts and score deltas for one hand.
type HandResult struct {
	BidderSeat  int
	BidAmount   int
	IsPepper    bool
	IsStuck     bool
	TricksTaken [2]int // tricks taken by each team
	ScoreDelta  [2]int // score change for each team this hand
	MadeBid     bool

	// BidderHand is the bidder's hand before any pepper exchange.
	// Used by the sim to build the hand profile lookup table.
	BidderHand []card.Card
	Trump      card.Suit

	// TrumpStats tracks trump depletion across the hand.
	TrumpStats TrumpStats
}

// TrumpStats tracks trump distribution and depletion over a hand.
type TrumpStats struct {
	TotalTrump      int // total trump cards in deck (always 14 for pinochle)
	BidderTrump     int // trump held by bidding team before play
	OpponentTrump   int // trump held by opponents before play
	TrumpLedsCount  int // number of tricks where trump was led
	TrumpPlayedByTrick [8]int // cumulative trump played after each trick
}

// ScoreHand calculates score deltas for a completed hand.
func ScoreHand(
	bidderSeat int,
	bidAmount int,
	isPepper bool,
	tricksByTeam [2]int,
) HandResult {
	bidderTeam := TeamOf(bidderSeat)
	otherTeam := TeamIndex(1 - int(bidderTeam))

	result := HandResult{
		BidderSeat:  bidderSeat,
		BidAmount:   bidAmount,
		IsPepper:    isPepper,
		TricksTaken: tricksByTeam,
	}

	if isPepper {
		callerTricks := tricksByTeam[bidderTeam]
		oppTricks := tricksByTeam[otherTeam]

		if callerTricks == 8 {
			// Perfect pepper: +16 for caller, opponents score 0.
			result.ScoreDelta[bidderTeam] = 16
			result.MadeBid = true
		} else {
			// Missed pepper: caller -16, opponents +2 per trick they took.
			result.ScoreDelta[bidderTeam] = -16
			result.ScoreDelta[otherTeam] = oppTricks * 2
			result.MadeBid = false
		}
		return result
	}

	// Normal hand.
	callerTricks := tricksByTeam[bidderTeam]
	if callerTricks >= bidAmount {
		// Made the bid: score actual tricks taken.
		result.ScoreDelta[bidderTeam] = callerTricks
		result.ScoreDelta[otherTeam] = tricksByTeam[otherTeam]
		result.MadeBid = true
	} else {
		// Missed: bidding team goes negative their bid, other team scores their tricks.
		result.ScoreDelta[bidderTeam] = -bidAmount
		result.ScoreDelta[otherTeam] = tricksByTeam[otherTeam]
		result.MadeBid = false
	}

	return result
}

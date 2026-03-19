package ml

import (
	"math/rand"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// BidCollectRow is one training example: a (bid decision point, candidate bid level) pair
// with the expected score delta for the bidding seat's team averaged over rollouts.
type BidCollectRow struct {
	HandID    int
	Seat      int
	BidLevel  int // 0=pass, 4-7=normal, 8=pepper
	Features  [BidTotalLen]float32
	ScoreDelta float32 // expected score delta for this seat's team
}

// bidPoint captures the state at the moment a seat must decide its bid.
type bidPoint struct {
	seat        int
	hands       [6][]card.Card
	dealer      int
	scores      [2]int
	currentHigh int
	highSeat    int   // seat that holds current high bid (-1 if none)
	seatsLeft   []int // seats that still need to bid after this one (in order)
}

// rollout completes the auction and plays the hand, returning score delta for dp.seat's team.
func (dp bidPoint) rollout(forcedBid int, rolloutStrat [6]game.Strategy, rng *rand.Rand) int {
	currentHigh := dp.currentHigh
	highSeat := dp.highSeat

	// Apply this seat's forced bid.
	if forcedBid == game.PepperBid {
		return dp.playHand(dp.seat, game.PepperBid, true, rolloutStrat, rng)
	}
	if forcedBid != game.PassBid {
		minRequired := game.MinBid
		if currentHigh >= game.MinBid {
			minRequired = currentHigh + 1
		}
		if forcedBid >= minRequired {
			currentHigh = forcedBid
			highSeat = dp.seat
		}
		// If forcedBid is somehow invalid (shouldn't happen), treat as pass.
	}

	// Continue bidding for remaining seats.
	for _, nextSeat := range dp.seatsLeft {
		isDealerTurn := nextSeat == dp.dealer
		if isDealerTurn && highSeat == -1 {
			// Dealer stuck at 3.
			return dp.playHand(dp.dealer, game.StuckBid, false, rolloutStrat, rng)
		}

		bidState := game.BidState{
			Hand:        dp.hands[nextSeat],
			Seat:        nextSeat,
			DealerSeat:  dp.dealer,
			CurrentHigh: currentHigh,
			Scores:      dp.scores,
		}
		bid := rolloutStrat[nextSeat].Bid(nextSeat, bidState)

		if bid == game.PepperBid {
			return dp.playHand(nextSeat, game.PepperBid, true, rolloutStrat, rng)
		}
		if bid != game.PassBid {
			minRequired := game.MinBid
			if currentHigh >= game.MinBid {
				minRequired = currentHigh + 1
			}
			if bid >= minRequired {
				currentHigh = bid
				highSeat = nextSeat
			}
		}
	}

	if highSeat == -1 {
		// Shouldn't happen (dealer stuck prevents this), but guard anyway.
		return 0
	}
	return dp.playHand(highSeat, currentHigh, false, rolloutStrat, rng)
}

// playHand plays a complete hand given the auction result and returns score delta
// for dp.seat's team.
func (dp bidPoint) playHand(
	callerSeat int,
	bidAmount int,
	isPepper bool,
	rolloutStrat [6]game.Strategy,
	rng *rand.Rand,
) int {
	hands := copyHands(dp.hands)

	trump := rolloutStrat[callerSeat].ChooseTrump(callerSeat, hands[callerSeat])

	var sittingOut [2]int
	if isPepper {
		hands, sittingOut = game.PepperExchange(
			hands,
			callerSeat,
			trump,
			func(seat int, hand []card.Card, t card.Suit) card.Card {
				return rolloutStrat[seat].GivePepper(seat, hand, t)
			},
			func(seat int, hand []card.Card, t card.Suit, received [2]card.Card) [2]card.Card {
				return rolloutStrat[seat].PepperDiscard(seat, hand, t, received)
			},
		)
	}

	activeSeats := buildActiveSeats(isPepper, callerSeat, sittingOut)
	tricksTaken := [6]int{}
	leader := callerSeat
	var history game.HandHistory

	for t := 0; t < 8; t++ {
		trick := game.NewTrick(leader, trump)
		for step := 0; step < len(activeSeats); step++ {
			seat := activeSeats[(indexInSlice(activeSeats, leader)+step)%len(activeSeats)]
			valid := game.ValidPlays(hands[seat], trick, trump)
			state := game.TrickState{
				Trick:       trick,
				Trump:       trump,
				Seat:        seat,
				BidderSeat:  callerSeat,
				BidAmount:   bidAmount,
				TrickNumber: t,
				TricksTaken: tricksTaken,
				Scores:      dp.scores,
				History:     &history,
				Hand:        hands[seat],
			}
			chosen := rolloutStrat[seat].Play(seat, valid, state)
			trick.Add(chosen, seat)
			hands[seat] = dropCard(hands[seat], chosen)
		}

		var trickCards []card.Card
		for _, pc := range trick.Cards {
			trickCards = append(trickCards, pc.Card)
		}
		history.Record(trickCards)
		winner := trick.Winner()
		tricksTaken[winner]++
		leader = winner
	}

	var tricksByTeam [2]int
	for s, count := range tricksTaken {
		tricksByTeam[game.TeamOf(s)] += count
	}
	result := game.ScoreHand(callerSeat, bidAmount, isPepper, tricksByTeam)
	return result.ScoreDelta[game.TeamOf(dp.seat)]
}

// CollectBidHand runs one complete hand, intercepting each bid decision and
// evaluating each valid bid option via counterfactual rollouts.
// For each bid decision point, one BidCollectRow is produced per valid bid level.
func CollectBidHand(
	handID int,
	gs *game.GameState,
	strategies [6]game.Strategy,
	rolloutStrat [6]game.Strategy,
	rng *rand.Rand,
	rollouts int,
) []BidCollectRow {
	hands := card.Deal(rng)
	dealer := gs.Dealer
	scores := gs.Scores

	currentHigh := 0
	highSeat := -1
	var rows []BidCollectRow

	for i := 1; i <= 6; i++ {
		seat := (dealer + i) % 6

		isDealerTurn := seat == dealer
		if isDealerTurn && highSeat == -1 {
			// Stuck hand — no useful bidding data, just return what we have.
			break
		}

		// Build the list of seats that still need to bid after this one.
		var seatsLeft []int
		for j := i + 1; j <= 6; j++ {
			seatsLeft = append(seatsLeft, (dealer+j)%6)
		}

		bidState := game.BidState{
			Hand:        hands[seat],
			Seat:        seat,
			DealerSeat:  dealer,
			CurrentHigh: currentHigh,
			Scores:      scores,
		}

		dp := bidPoint{
			seat:        seat,
			hands:       copyHands(hands),
			dealer:      dealer,
			scores:      scores,
			currentHigh: currentHigh,
			highSeat:    highSeat,
			seatsLeft:   seatsLeft,
		}

		// Extract context features once for this decision point.
		ctx := BidContext(seat, hands[seat], dealer, currentHigh, scores)

		// Evaluate each valid bid level via counterfactual rollouts.
		for _, bidLevel := range ValidBidLevels(currentHigh) {
			var totalDelta float64
			for r := 0; r < rollouts; r++ {
				delta := dp.rollout(bidLevel, rolloutStrat, rng)
				totalDelta += float64(delta)
			}
			rows = append(rows, BidCollectRow{
				HandID:     handID,
				Seat:       seat,
				BidLevel:   bidLevel,
				Features:   AppendBidAction(ctx, bidLevel),
				ScoreDelta: float32(totalDelta / float64(rollouts)),
			})
		}

		// Advance bidding state using the base strategy's actual decision.
		bid := strategies[seat].Bid(seat, bidState)
		if bid == game.PepperBid {
			break
		}
		if bid != game.PassBid {
			minRequired := game.MinBid
			if currentHigh >= game.MinBid {
				minRequired = currentHigh + 1
			}
			if bid >= minRequired {
				currentHigh = bid
				highSeat = seat
			}
		}
	}

	return rows
}

package ml

import (
	"math/rand"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// CollectRow is one training example: a (decision point, candidate card) pair with
// the counterfactual outcome of playing that card averaged over rollouts.
type CollectRow struct {
	HandID        int
	TrickNumber   int
	Seat          int
	IsBiddingTeam bool
	Features      [TotalFeatureLen]float32 // full feature vector including card features
	ScoreDelta    float32                  // mean score delta for bidding team across rollouts
	MadeBidRate   float32                  // fraction of rollouts where bid was made
}

// decisionPoint captures a snapshot of game state at the moment a player must play,
// including everything needed to fork and evaluate card choices counterfactually.
type decisionPoint struct {
	seat         int
	hands        [6][]card.Card   // deep copy: all current hands (cards removed as tricks completed)
	trick        []game.PlayedCard // cards already in the current trick (before this seat plays)
	trickNum     int
	tricksTaken  [6]int
	historyCards []card.Card // copy of all cards from completed tricks
	leader       int
	trump        card.Suit
	callerSeat   int
	bidAmount    int
	isPepper     bool
	scores       [2]int
	activeSeats  []int
	seatStep     int // this seat's step index within activeSeats for current trick
}

// rollout simulates the remainder of the hand from this decision point,
// with forcedCard played by this seat, and all other decisions made by rolloutStrat.
// Returns the score delta for the bidding team and whether the bid was made.
func (dp decisionPoint) rollout(forcedCard card.Card, rolloutStrat [6]game.Strategy, rng *rand.Rand) (scoreDelta int, madeBid bool) {
	// Deep copy hands so each rollout is independent.
	hands := copyHands(dp.hands)

	// Reconstruct history from completed tricks.
	var history game.HandHistory
	if len(dp.historyCards) > 0 {
		history.Record(dp.historyCards)
	}

	tricksTaken := dp.tricksTaken

	// --- Finish the current trick ---
	trick := game.NewTrick(dp.leader, dp.trump)

	// Re-add cards already played before this seat.
	for _, pc := range dp.trick {
		trick.Add(pc.Card, pc.Seat)
	}

	// Play the forced card for this seat.
	trick.Add(forcedCard, dp.seat)
	hands[dp.seat] = dropCard(hands[dp.seat], forcedCard)

	// Remaining seats in the current trick (after this seat's step).
	for step := dp.seatStep + 1; step < len(dp.activeSeats); step++ {
		nextSeat := dp.activeSeats[(indexInSlice(dp.activeSeats, dp.leader)+step)%len(dp.activeSeats)]
		valid := game.ValidPlays(hands[nextSeat], trick, dp.trump)
		state := game.TrickState{
			Trick:       trick,
			Trump:       dp.trump,
			Seat:        nextSeat,
			BidderSeat:  dp.callerSeat,
			BidAmount:   dp.bidAmount,
			TrickNumber: dp.trickNum,
			TricksTaken: tricksTaken,
			Scores:      dp.scores,
			History:     &history,
			Hand:        hands[nextSeat],
		}
		chosen := rolloutStrat[nextSeat].Play(nextSeat, valid, state)
		trick.Add(chosen, nextSeat)
		hands[nextSeat] = dropCard(hands[nextSeat], chosen)
	}

	// Record completed trick in history and update state.
	var trickCards []card.Card
	for _, pc := range trick.Cards {
		trickCards = append(trickCards, pc.Card)
	}
	history.Record(trickCards)
	winner := trick.Winner()
	tricksTaken[winner]++
	leader := winner

	// --- Play remaining tricks ---
	for t := dp.trickNum + 1; t < game.TotalTricks; t++ {
		trick = game.NewTrick(leader, dp.trump)

		for step := 0; step < len(dp.activeSeats); step++ {
			s := dp.activeSeats[(indexInSlice(dp.activeSeats, leader)+step)%len(dp.activeSeats)]
			valid := game.ValidPlays(hands[s], trick, dp.trump)
			state := game.TrickState{
				Trick:       trick,
				Trump:       dp.trump,
				Seat:        s,
				BidderSeat:  dp.callerSeat,
				BidAmount:   dp.bidAmount,
				TrickNumber: t,
				TricksTaken: tricksTaken,
				Scores:      dp.scores,
				History:     &history,
				Hand:        hands[s],
			}
			chosen := rolloutStrat[s].Play(s, valid, state)
			trick.Add(chosen, s)
			hands[s] = dropCard(hands[s], chosen)
		}

		trickCards = trickCards[:0]
		for _, pc := range trick.Cards {
			trickCards = append(trickCards, pc.Card)
		}
		history.Record(trickCards)
		winner = trick.Winner()
		tricksTaken[winner]++
		leader = winner
	}

	// Score the hand.
	var tricksByTeam [2]int
	for s, count := range tricksTaken {
		tricksByTeam[game.TeamOf(s)] += count
	}
	result := game.ScoreHand(dp.callerSeat, dp.bidAmount, dp.isPepper, tricksByTeam)
	bidderTeam := game.TeamOf(dp.callerSeat)
	return result.ScoreDelta[bidderTeam], result.MadeBid
}

// CollectHand runs one complete hand and returns counterfactual training rows.
// For every play decision, each legal card is evaluated via `rollouts` simulations
// using rolloutStrat for all other seats. The base strategies handle bidding and
// trump selection; rolloutStrat handles counterfactual play evaluation.
func CollectHand(
	handID int,
	gs *game.GameState,
	strategies [6]game.Strategy,
	rolloutStrat [6]game.Strategy,
	rng *rand.Rand,
	rollouts int,
) []CollectRow {
	// Deal.
	hands := card.Deal(rng)

	// Bid.
	bidResult := game.RunBidding(
		hands,
		gs.Dealer,
		gs.Scores,
		func(seat int, state game.BidState) int {
			return strategies[seat].Bid(seat, state)
		},
	)
	if bidResult.IsStuck {
		return nil // stuck hands don't produce useful play data
	}

	callerSeat := bidResult.Winner

	// Choose trump.
	trump := strategies[callerSeat].ChooseTrump(callerSeat, hands[callerSeat])

	// Pepper exchange.
	var sittingOut [2]int
	isPepper := bidResult.IsPepper
	if isPepper {
		hands, sittingOut = game.PepperExchange(
			hands,
			callerSeat,
			trump,
			func(seat int, hand []card.Card, trump card.Suit) card.Card {
				return strategies[seat].GivePepper(seat, hand, trump)
			},
			func(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
				return strategies[seat].PepperDiscard(seat, hand, trump, received)
			},
		)
	}

	activeSeats := buildActiveSeats(isPepper, callerSeat, sittingOut)

	// Play tricks, intercepting each decision.
	tricksTaken := [6]int{}
	leader := callerSeat
	var history game.HandHistory
	var rows []CollectRow

	for t := 0; t < game.TotalTricks; t++ {
		trick := game.NewTrick(leader, trump)

		for step := 0; step < len(activeSeats); step++ {
			seat := activeSeats[(indexInSlice(activeSeats, leader)+step)%len(activeSeats)]
			valid := game.ValidPlays(hands[seat], trick, trump)

			state := game.TrickState{
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
			}

			// Build decision point snapshot for counterfactual evaluation.
			dp := decisionPoint{
				seat:         seat,
				hands:        copyHands(hands),
				trick:        copyPlayedCards(trick.Cards),
				trickNum:     t,
				tricksTaken:  tricksTaken,
				historyCards: history.Played(),
				leader:       leader,
				trump:        trump,
				callerSeat:   callerSeat,
				bidAmount:    bidResult.Amount,
				isPepper:     isPepper,
				scores:       gs.Scores,
				activeSeats:  activeSeats,
				seatStep:     step,
			}

			// Extract shared context features (uses full hand, not filtered valid).
			ctx := ExtractContext(seat, hands[seat], state, len(activeSeats))

			// Evaluate each legal card via counterfactual rollouts.
			for _, candidate := range valid {
				var totalDelta float64
				madeBidCount := 0
				for r := 0; r < rollouts; r++ {
					delta, made := dp.rollout(candidate, rolloutStrat, rng)
					totalDelta += float64(delta)
					if made {
						madeBidCount++
					}
				}
				rows = append(rows, CollectRow{
					HandID:        handID,
					TrickNumber:   t,
					Seat:          seat,
					IsBiddingTeam: game.TeamOf(seat) == game.TeamOf(callerSeat),
					Features:      AppendCard(ctx, candidate, trump, hands[seat], trick, &history),
					ScoreDelta:    float32(totalDelta / float64(rollouts)),
					MadeBidRate:   float32(madeBidCount) / float32(rollouts),
				})
			}

			// Play the actual card using the base strategy.
			chosen := strategies[seat].Play(seat, valid, state)
			trick.Add(chosen, seat)
			hands[seat] = dropCard(hands[seat], chosen)
		}

		// Record trick in history.
		trickCards := make([]card.Card, len(trick.Cards))
		for j, pc := range trick.Cards {
			trickCards[j] = pc.Card
		}
		history.Record(trickCards)

		winner := trick.Winner()
		tricksTaken[winner]++
		leader = winner
	}

	return rows
}

// --- Helpers ---

func dropCard(hand []card.Card, target card.Card) []card.Card {
	result := make([]card.Card, 0, len(hand))
	removed := false
	for _, c := range hand {
		if !removed && c.Equal(target) {
			removed = true
			continue
		}
		result = append(result, c)
	}
	return result
}

func copyHands(hands [6][]card.Card) [6][]card.Card {
	var result [6][]card.Card
	for i, h := range hands {
		result[i] = make([]card.Card, len(h))
		copy(result[i], h)
	}
	return result
}

func copyPlayedCards(pcs []game.PlayedCard) []game.PlayedCard {
	result := make([]game.PlayedCard, len(pcs))
	copy(result, pcs)
	return result
}

func buildActiveSeats(pepperActive bool, callerSeat int, sittingOut [2]int) []int {
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

func indexInSlice(slice []int, val int) int {
	for i, v := range slice {
		if v == val {
			return i
		}
	}
	return 0
}

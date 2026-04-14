package ml

import (
	"encoding/binary"
	"io"
	"math"
	"math/rand"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// BidCollectRow is one training example...
type BidCollectRow struct {
	HandID     int
	Seat       int
	BidLevel   int
	Features   [BidTotalLen]float32
	ScoreDelta float32
}

// WriteBinary writes the row in a compact binary format without reflection.
func (r BidCollectRow) WriteBinary(w io.Writer) error {
	var buf [6 + 4*BidTotalLen + 4]byte
	binary.LittleEndian.PutUint32(buf[0:4], uint32(r.HandID))
	buf[4] = uint8(r.Seat)
	buf[5] = uint8(r.BidLevel)
	off := 6
	for _, f := range r.Features {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(f))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(r.ScoreDelta))
	_, err := w.Write(buf[:])
	return err
}

// ReadBinary reads a row from a compact binary format without reflection.
func (r *BidCollectRow) ReadBinary(rd io.Reader) error {
	var buf [6 + 4*BidTotalLen + 4]byte
	if _, err := io.ReadFull(rd, buf[:]); err != nil {
		return err
	}
	r.HandID = int(binary.LittleEndian.Uint32(buf[0:4]))
	r.Seat = int(buf[4])
	r.BidLevel = int(buf[5])
	off := 6
	for i := 0; i < BidTotalLen; i++ {
		r.Features[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[off : off+4]))
		off += 4
	}
	r.ScoreDelta = math.Float32frombits(binary.LittleEndian.Uint32(buf[off : off+4]))
	return nil
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
func (dp bidPoint) rollout(forcedBid int, rolloutStrat [6]game.Strategy, rng *rand.Rand, validBuf *[]card.Card) int {
	currentHigh := dp.currentHigh
	highSeat := dp.highSeat

	// Apply this seat's forced bid.
	if forcedBid == game.PepperBid {
		return dp.playHand(dp.hands, dp.seat, game.PepperBid, true, rolloutStrat, rng, validBuf)
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
			return dp.playHand(dp.hands, dp.dealer, game.StuckBid, false, rolloutStrat, rng, validBuf)
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
			return dp.playHand(dp.hands, nextSeat, game.PepperBid, true, rolloutStrat, rng, validBuf)
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
	return dp.playHand(dp.hands, highSeat, currentHigh, false, rolloutStrat, rng, validBuf)
}

// playHand plays a complete hand given the auction result and returns score delta
// for dp.seat's team. dealtHands are the hands for this rollout (already re-sampled).
func (dp bidPoint) playHand(
	dealtHands [6][]card.Card,
	callerSeat int,
	bidAmount int,
	isPepper bool,
	rolloutStrat [6]game.Strategy,
	rng *rand.Rand,
	validBuf *[]card.Card,
) int {
	// Get a pooled buffer and copy hands into it — avoids per-rollout make() calls.
	buf := handBufPool.Get().(*[6][]card.Card)
	var hands [6][]card.Card
	for i, h := range dealtHands {
		n := len(h)
		hands[i] = (*buf)[i][:n]
		copy(hands[i], h)
	}

	trump := rolloutStrat[callerSeat].ChooseTrump(callerSeat, hands[callerSeat])

	var sittingOut [2]int
	if isPepper {
		sittingOut = game.PepperExchange(
			&hands,
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

	// Get a pooled HandHistory — avoids heap escape through TrickState.History interface call.
	history := historyPool.Get().(*game.HandHistory)
	history.Reset()

	trick := trickPool.Get().(*game.Trick)
	trick.Reset(leader, trump)
	for t := 0; t < game.TotalTricks; t++ {
		trick.Reset(leader, trump)
		leaderIdx := indexInSlice(activeSeats, leader)
		for step := 0; step < len(activeSeats); step++ {
			seat := activeSeats[(leaderIdx+step)%len(activeSeats)]
			valid := game.ValidPlaysInto(validBuf, hands[seat], trick, trump)
			state := game.TrickState{
				Trick:       trick,
				Trump:       trump,
				Seat:        seat,
				BidderSeat:  callerSeat,
				BidAmount:   bidAmount,
				TrickNumber: t,
				TricksTaken: tricksTaken,
				Scores:      dp.scores,
				History:     history,
				Hand:        hands[seat],
			}
			chosen := rolloutStrat[seat].Play(seat, valid, state)
			trick.Add(chosen, seat)
			hands[seat] = dropCardInPlace(hands[seat], chosen)
		}

		history.RecordTrick(trick.Cards)
		winner := trick.Winner()
		tricksTaken[winner]++
		leader = winner
	}

	handBufPool.Put(buf)
	trickPool.Put(trick)
	historyPool.Put(history)

	var tricksByTeam [2]int
	for s, count := range tricksTaken {
		tricksByTeam[game.TeamOf(s)] += count
	}
	result := game.ScoreHand(callerSeat, bidAmount, isPepper, tricksByTeam)
	return result.ScoreDelta[game.TeamOf(dp.seat)]
}

// CollectBidHand runs one complete hand, intercepting each bid decision and
// evaluating each valid bid option via counterfactual rollouts.
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
	rows := make([]BidCollectRow, 0, 6*6)

	for i := 1; i <= 6; i++ {
		seat := (dealer + i) % 6
		isDealerTurn := seat == dealer
		if isDealerTurn && highSeat == -1 {
			break
		}

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

		// Capture decision point snapshot.
		pooledHands := copyHandsPooled(hands)
		dp := bidPoint{
			seat:        seat,
			hands:       *pooledHands,
			dealer:      dealer,
			scores:      scores,
			currentHigh: currentHigh,
			highSeat:    highSeat,
			seatsLeft:   seatsLeft,
		}

		ctx := BidContext(seat, hands[seat], dealer, currentHigh, highSeat, len(seatsLeft), scores)

		// Evaluate each bid level sequentially (parallelism is at the worker level).
		validBids := ValidBidLevels(currentHigh)
		var rolloutValidBuf []card.Card
		for _, bidLevel := range validBids {
			var totalDelta float64
			for r := 0; r < rollouts; r++ {
				delta := dp.rollout(bidLevel, rolloutStrat, rng, &rolloutValidBuf)
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
		releaseHands(pooledHands)

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

package mlstrategy

import (
	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
	"github.com/max/pepper/internal/strategy"
	"github.com/max/pepper/ml"
)

// MLPStrategy plays cards using a trained MLP model.
// Bidding uses a separate BidMLP if loaded; otherwise falls back to StandardStrategy.
// Trump selection and pepper decisions always delegate to StandardStrategy.
type MLPStrategy struct {
	model    *ml.MLP
	bidModel *ml.BidMLP // optional; nil = use fallback for bidding
	fallback *strategy.StandardStrategy
}

// NewMLPStrategy creates an MLPStrategy. The fallbackCfg handles all non-play
// decisions (bidding, trump selection, pepper exchange). Balanced is a good default.
func NewMLPStrategy(model *ml.MLP, fallbackCfg strategy.Config) *MLPStrategy {
	return &MLPStrategy{
		model:    model,
		fallback: strategy.NewStandard(fallbackCfg),
	}
}

// WithBidModel attaches a trained bid MLP to the strategy.
func (s *MLPStrategy) WithBidModel(bidModel *ml.BidMLP) *MLPStrategy {
	s.bidModel = bidModel
	return s
}

func (s *MLPStrategy) Bid(seat int, state game.BidState) int {
	if s.bidModel == nil {
		return s.fallback.Bid(seat, state)
	}
	return bidWithMLP(s.bidModel, seat, state)
}

// bidWithMLP uses the bid MLP to choose the best bid level.
// Picks the valid bid level that maximizes expected score delta for this seat's team.
func bidWithMLP(m *ml.BidMLP, seat int, state game.BidState) int {
	ctx := ml.BidContext(seat, state.Hand, state.DealerSeat, state.CurrentHigh, state.Scores)

	bestBid := game.PassBid
	bestScore := float32(-1e30)

	for _, bidLevel := range ml.ValidBidLevels(state.CurrentHigh) {
		features := ml.AppendBidAction(ctx, bidLevel)
		score := m.Score(features)
		if score > bestScore {
			bestScore = score
			bestBid = bidLevel
		}
	}
	return bestBid
}

func (s *MLPStrategy) ChooseTrump(seat int, hand []card.Card) card.Suit {
	return s.fallback.ChooseTrump(seat, hand)
}

func (s *MLPStrategy) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card {
	return s.fallback.GivePepper(seat, hand, trump)
}

func (s *MLPStrategy) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
	return s.fallback.PepperDiscard(seat, hand, trump, received)
}

func (s *MLPStrategy) Play(seat int, validPlays []card.Card, state game.TrickState) card.Card {
	// Pepper play is scripted — fall back to StandardStrategy.
	if state.BidAmount == game.PepperBid {
		return s.fallback.Play(seat, validPlays, state)
	}

	// Use the full hand for context features; fall back to validPlays if not set.
	hand := state.Hand
	if len(hand) == 0 {
		hand = validPlays
	}

	// 4 active players for pepper, 6 for normal — always 6 here (pepper handled above).
	const activePlayers = 6
	ctx := ml.ExtractContext(seat, hand, state, activePlayers)
	isBidder := game.TeamOf(seat) == game.TeamOf(state.BidderSeat)

	best := validPlays[0]
	bestScore := float32(-1e30)

	for _, c := range validPlays {
		features := ml.AppendCard(ctx, c, state.Trump, hand, state.Trick, state.History)
		score := s.model.Score(features)
		// score_delta is from the bidding team's perspective.
		// Defenders want it low, so flip sign.
		if !isBidder {
			score = -score
		}
		if score > bestScore {
			bestScore = score
			best = c
		}
	}
	return best
}

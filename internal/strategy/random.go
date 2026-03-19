package strategy

import (
	"math/rand"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// Random plays and bids completely randomly. Used only for smoke testing the engine.
type Random struct {
	rng *rand.Rand
}

func NewRandom(rng *rand.Rand) *Random {
	return &Random{rng: rng}
}

func (r *Random) Bid(seat int, state game.BidState) int {
	minBid := game.MinBid
	if state.CurrentHigh >= game.MinBid {
		minBid = state.CurrentHigh + 1
	}
	if minBid > 8 {
		return game.PassBid
	}
	if r.rng.Float64() < 0.4 {
		return game.PassBid
	}
	return minBid + r.rng.Intn(8-minBid+1)
}

func (r *Random) Play(seat int, validPlays []card.Card, state game.TrickState) card.Card {
	return validPlays[r.rng.Intn(len(validPlays))]
}

func (r *Random) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card {
	best, found := game.BestTrump(hand, trump)
	if found {
		return best
	}
	return hand[r.rng.Intn(len(hand))]
}

func (r *Random) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
	var nonTrump []card.Card
	for _, c := range hand {
		if card.TrumpRank(c, trump) < 0 {
			nonTrump = append(nonTrump, c)
		}
	}
	pool := hand
	if len(nonTrump) >= 2 {
		pool = nonTrump
	}
	cp := make([]card.Card, len(pool))
	copy(cp, pool)
	r.rng.Shuffle(len(cp), func(i, j int) { cp[i], cp[j] = cp[j], cp[i] })
	return [2]card.Card{cp[0], cp[1]}
}

func (r *Random) ChooseTrump(seat int, hand []card.Card) card.Suit {
	suits := []card.Suit{card.Spades, card.Clubs, card.Hearts, card.Diamonds}
	best := card.Spades
	bestCount := -1
	for _, s := range suits {
		count := 0
		for _, c := range hand {
			if card.TrumpRank(c, s) >= 0 {
				count++
			}
		}
		if count > bestCount {
			bestCount = count
			best = s
		}
	}
	return best
}

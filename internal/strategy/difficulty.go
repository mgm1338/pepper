package strategy

import (
	"math/rand"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// DifficultyConfig controls how a DifficultyStrategy blends MLP and Balanced play.
type DifficultyConfig struct {
	PlayMixRate float64 // probability of using MLP for card play (vs Balanced)
	BidMixRate  float64 // probability of using MLP for bidding (vs Balanced)
	PlayEps     float64 // probability of playing a random valid card
	BidNoise    float64 // σ of Gaussian noise added to MLP bid scores
}

// DifficultyLevels defines the 5 preset difficulty levels.
// Level 1 = beginner (index 0), Level 5 = full AI (index 4).
var DifficultyLevels = [5]DifficultyConfig{
	{0.00, 0.00, 0.12, 2.0}, // Level 1: sloppy Balanced, overbids
	{0.00, 0.00, 0.05, 1.0}, // Level 2: slightly less sloppy Balanced
	{0.00, 0.00, 0.00, 0.0}, // Level 3: clean Balanced (serviceable)
	{0.70, 0.00, 0.00, 0.0}, // Level 4: MLP card play, Balanced bid (sharp)
	{1.00, 1.00, 0.00, 0.0}, // Level 5: full MLP (AI)
}

// Level45Config is a half step between level 4 and level 5:
// full MLP card play, Balanced bidding (no overbid).
var Level45Config = DifficultyConfig{
	PlayMixRate: 1.00,
	BidMixRate:  0.00,
	PlayEps:     0.00,
	BidNoise:    0.0,
}

// NewLevel45 creates a level 4.5 strategy.
func NewLevel45(mlp game.Strategy, rng *rand.Rand) *DifficultyStrategy {
	return &DifficultyStrategy{
		mlp:      mlp,
		balanced: NewStandard(Balanced),
		cfg:      Level45Config,
		rng:      rng,
	}
}

// Level46Config: the "bidder" 4.5 — full MLP bidding, Balanced card play.
var Level46Config = DifficultyConfig{
	PlayMixRate: 0.00,
	BidMixRate:  1.00,
	PlayEps:     0.00,
	BidNoise:    0.0,
}

// NewLevel46 creates the bidder-focused 4.5 strategy.
func NewLevel46(mlp game.Strategy, rng *rand.Rand) *DifficultyStrategy {
	return &DifficultyStrategy{
		mlp:      mlp,
		balanced: NewStandard(Balanced),
		cfg:      Level46Config,
		rng:      rng,
	}
}

// CautiousConfig: level 3-ish play, underbids.
var CautiousConfig = DifficultyConfig{
	PlayMixRate: 0.00,
	BidMixRate:  0.00,
	PlayEps:     0.02,
	BidNoise:    -1.0,
}

// AggressiveConfig: sharp play, overbids like Peter but harder.
var AggressiveConfig = DifficultyConfig{
	PlayMixRate: 0.70,
	BidMixRate:  0.00,
	PlayEps:     0.00,
	BidNoise:    1.5,
}

// RiskyConfig: random spicy play, normal bids.
var RiskyConfig = DifficultyConfig{
	PlayMixRate: 0.30,
	BidMixRate:  0.00,
	PlayEps:     0.15,
	BidNoise:    0.8,
}

// NewPersonality builds a named personality strategy.
func NewPersonality(name string, mlp game.Strategy, rng *rand.Rand) *DifficultyStrategy {
	var cfg DifficultyConfig
	switch name {
	case "aggressive":
		cfg = AggressiveConfig
	case "cautious":
		cfg = CautiousConfig
	case "risky":
		cfg = RiskyConfig
	case "scientist":
		cfg = DifficultyLevels[4] // same as level 5
	default:
		cfg = DifficultyLevels[2]
	}
	return &DifficultyStrategy{
		mlp:      mlp,
		balanced: NewStandard(Balanced),
		cfg:      cfg,
		rng:      rng,
	}
}

// PeterConfig is a special personality: sharp card play but overbids by ~1 point.
// "Peter" always thinks his hand is a little better than it is.
var PeterConfig = DifficultyConfig{
	PlayMixRate: 0.70, // sharp card play like level 4
	BidMixRate:  0.00, // uses Balanced bid as base
	PlayEps:     0.00,
	BidNoise:    1.0,  // positive-biased noise: overbids
}

// NewPeter creates the special "Peter" bot personality.
func NewPeter(mlp game.Strategy, rng *rand.Rand) *DifficultyStrategy {
	return &DifficultyStrategy{
		mlp:      mlp,
		balanced: NewStandard(Balanced),
		cfg:      PeterConfig,
		rng:      rng,
	}
}

// DifficultyStrategy wraps an MLP strategy and a Balanced fallback,
// blending them according to a DifficultyConfig.
type DifficultyStrategy struct {
	mlp      game.Strategy // the full MLP strategy (level 5)
	balanced game.Strategy // the Balanced rule-based strategy
	cfg      DifficultyConfig
	rng      *rand.Rand
}

// NewDifficulty creates a DifficultyStrategy at the given level (1-5).
// mlp is the full MLP strategy; if nil, all levels behave as Balanced.
func NewDifficulty(level int, mlp game.Strategy, rng *rand.Rand) *DifficultyStrategy {
	if level < 1 {
		level = 1
	}
	if level > 5 {
		level = 5
	}
	return &DifficultyStrategy{
		mlp:      mlp,
		balanced: NewStandard(Balanced),
		cfg:      DifficultyLevels[level-1],
		rng:      rng,
	}
}

func (d *DifficultyStrategy) Bid(seat int, state game.BidState) int {
	if d.mlp != nil && d.cfg.BidMixRate > 0 && d.rng.Float64() < d.cfg.BidMixRate {
		return d.mlp.Bid(seat, state)
	}
	bid := d.balanced.Bid(seat, state)
	if d.cfg.BidNoise > 0 {
		bid = d.noisyBid(bid, state)
	}
	return bid
}

// noisyBid adjusts a Balanced bid by adding random noise to the decision.
// Positive noise makes the player overbid; negative makes them underbid/pass.
func (d *DifficultyStrategy) noisyBid(baseBid int, state game.BidState) int {
	if baseBid == game.PepperBid {
		return baseBid // don't mess with pepper calls
	}
	noise := d.rng.NormFloat64() * d.cfg.BidNoise
	// Peter-style: if this is a positive-bias config, use abs(noise) to always push up.
	if d.cfg == PeterConfig {
		if noise < 0 {
			noise = -noise
		}
	}
	adjusted := float64(baseBid) + noise

	// Convert back to a valid bid level.
	if adjusted < float64(game.MinBid)-0.5 {
		return game.PassBid
	}
	level := int(adjusted + 0.5)
	if level < game.MinBid {
		return game.PassBid
	}
	if level > 7 {
		level = 7
	}
	// Must exceed current high to be valid.
	if state.CurrentHigh >= game.MinBid && level <= state.CurrentHigh {
		return game.PassBid
	}
	return level
}

func (d *DifficultyStrategy) Play(seat int, validPlays []card.Card, state game.TrickState) card.Card {
	// Random play chance (levels 1-2).
	if d.cfg.PlayEps > 0 && d.rng.Float64() < d.cfg.PlayEps {
		return validPlays[d.rng.Intn(len(validPlays))]
	}
	// MLP mix (levels 4-5).
	if d.mlp != nil && d.cfg.PlayMixRate > 0 && d.rng.Float64() < d.cfg.PlayMixRate {
		return d.mlp.Play(seat, validPlays, state)
	}
	return d.balanced.Play(seat, validPlays, state)
}

func (d *DifficultyStrategy) ChooseTrump(seat int, hand []card.Card) card.Suit {
	return d.balanced.ChooseTrump(seat, hand)
}

func (d *DifficultyStrategy) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card {
	return d.balanced.GivePepper(seat, hand, trump)
}

func (d *DifficultyStrategy) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
	return d.balanced.PepperDiscard(seat, hand, trump, received)
}

package strategy

import (
	"math/rand"

	"github.com/max/pepper/internal/card"
	"github.com/max/pepper/internal/game"
)

// TrumpMemoryLevel controls how much trump history Balanced tracks.
type TrumpMemoryLevel string

const (
	TrumpMemoryNone    TrumpMemoryLevel = "none"    // always assumes opponents still hold trump
	TrumpMemoryBowers  TrumpMemoryLevel = "bowers"  // tracks right bowers only
	TrumpMemoryHigh    TrumpMemoryLevel = "high"    // tracks right + left bowers
	TrumpMemoryFull    TrumpMemoryLevel = "full"    // exact remaining count
)

func (l TrumpMemoryLevel) toInt() int {
	switch l {
	case TrumpMemoryBowers:
		return 1
	case TrumpMemoryHigh:
		return 2
	case TrumpMemoryFull:
		return 3
	default:
		return 0
	}
}

// AceMemoryLevel controls whether Balanced tracks which off-suit aces have dropped.
type AceMemoryLevel string

const (
	AceMemoryNone AceMemoryLevel = "none" // only cashes aces held in hand
	AceMemoryAces AceMemoryLevel = "aces" // knows when off-suit aces have dropped; leads kings as tops
)

// BotConfig controls how a bot blends MLP and Balanced play.
type BotConfig struct {
	PlayStyle   float64          // 0 = pure Balanced card play, 1 = pure MLP
	BidStyle    float64          // 0 = pure Balanced bidding, 1 = pure MLP
	Slop        float64          // 0 = no mistakes, 1 = maximum randomness and bid noise
	TrumpMemory TrumpMemoryLevel // how much trump history Balanced uses
	AceMemory   AceMemoryLevel   // whether Balanced tracks which off-suit aces have dropped
}

const maxPlayEps  = 0.15 // Slop=1 → 15% random card plays
const maxBidNoise = 2.0  // Slop=1 → 2.0σ bid noise

// ConfiguredBot blends an MLP strategy and Balanced fallback according to a BotConfig.
// If mlp is nil all decisions use Balanced regardless of PlayStyle/BidStyle.
type ConfiguredBot struct {
	mlp      game.Strategy
	balanced game.Strategy
	cfg      BotConfig
}

// NewBot creates a ConfiguredBot with the given config and optional MLP strategy.
func NewBot(cfg BotConfig, mlp game.Strategy) *ConfiguredBot {
	balancedCfg := Balanced
	balancedCfg.TrumpMemory = cfg.TrumpMemory.toInt()
	balancedCfg.AceMemory = cfg.AceMemory == AceMemoryAces
	return &ConfiguredBot{
		mlp:      mlp,
		balanced: NewStandard(balancedCfg),
		cfg:      cfg,
	}
}

func (b *ConfiguredBot) Bid(seat int, state *game.BidState) int {
	if b.mlp != nil && b.cfg.BidStyle > 0 && rand.Float64() < b.cfg.BidStyle {
		return b.mlp.Bid(seat, state)
	}
	bid := b.balanced.Bid(seat, state)
	if b.cfg.Slop > 0 {
		bid = b.noisyBid(bid, state)
	}
	return bid
}

func (b *ConfiguredBot) noisyBid(baseBid int, state *game.BidState) int {
	if baseBid == game.PepperBid {
		return baseBid
	}
	noise := rand.NormFloat64() * (b.cfg.Slop * maxBidNoise)
	adjusted := float64(baseBid) + noise
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
	if state.CurrentHigh >= game.MinBid && level <= state.CurrentHigh {
		return game.PassBid
	}
	return level
}

func (b *ConfiguredBot) Play(seat int, validPlays []card.Card, state *game.TrickState) card.Card {
	if b.cfg.Slop > 0 && rand.Float64() < b.cfg.Slop*maxPlayEps {
		return validPlays[rand.Intn(len(validPlays))]
	}
	if b.mlp != nil && b.cfg.PlayStyle > 0 && rand.Float64() < b.cfg.PlayStyle {
		return b.mlp.Play(seat, validPlays, state)
	}
	return b.balanced.Play(seat, validPlays, state)
}

func (b *ConfiguredBot) ChooseTrump(seat int, hand []card.Card) card.Suit {
	return b.balanced.ChooseTrump(seat, hand)
}

func (b *ConfiguredBot) GivePepper(seat int, hand []card.Card, trump card.Suit) card.Card {
	return b.balanced.GivePepper(seat, hand, trump)
}

func (b *ConfiguredBot) PepperDiscard(seat int, hand []card.Card, trump card.Suit, received [2]card.Card) [2]card.Card {
	return b.balanced.PepperDiscard(seat, hand, trump, received)
}
